"""act_a collect のデモで LeRobot の ACT を学習する（観測は状態だけ。画像なし）。

LeRobot の学習ループ（lerobot-train）は LeRobotDataset 形式と全依存を要るので使わず、モデル（ACTPolicy）だけを使う。
正規化（平均・標準偏差）はここで行い、checkpoint に stats.npz として残す（act_a.ActPolicy が読む）。

デモの扱い
  - 置き場所に平らに置けたエピソード（success）だけを使う
  - エピソードの終わりより先の行動は、最後の行動（待機姿勢）を繰り返す。「終わったら止まる」も学ばせるため、pad 扱いにしない

使い方: python -m pai_book_intake.act_train --data /workspace/outputs/act_collect --steps 5000
出力: /workspace/outputs/act_train/<run_id>/（step_<k>/ と final/ に config.json・model.safetensors・stats.npz・meta.json）
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from .runlog import classify, new_run, save_result

STD_FLOOR = 1e-3


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, nargs="+", default=[Path("/workspace/outputs/act_collect")],
                    help="act_collect の run のフォルダ（親を渡すと、中の run を全部使う）")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--chunk", type=int, default=100, help="一度に出す行動の数（50 Hz なので 100 = 2 s）")
    ap.add_argument("--kl-weight", type=float, default=10.0)
    ap.add_argument("--save-every", type=int, default=0, help="0 なら最後だけ保存")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def demo_files(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        files += sorted(p.glob("demos_*.npz")) if any(p.glob("demos_*.npz")) else sorted(p.glob("*/demos_*.npz"))
    return files


def load_episodes(files: list[Path]) -> tuple[list[dict], dict]:
    eps = []
    n_all = 0
    for f in files:
        z = np.load(f)
        # 配列はファイルごとに 1 回だけ読む。z[...] は読むたびに全環境ぶんを展開し、[i] の切り出しはその全体を
        # 抱えたままになる（seed 1〜6 で学習のコンテナが 10 GB を超えた）。切り出しは copy して全体を手放す
        state, env, action = z["state"], z["env"], z["action"]
        # 置けて、しかも本が 30 mm 以上上がったものだけ。10 冊の山の上の本では、持ち上げの高さが箱の壁で固定されていた版の
        # デモ（seed 1 の上 2 冊）が、下へ動いて置いていた（act_collect 20261027T205846-60198d。p2_line.carry_z で直した）
        lifted = (env[:, :, 2].max(1) - env[:, 0, 2]) > 0.03
        ok = z["success"] & lifted
        n_all += len(ok)
        for i in np.nonzero(ok)[0]:
            eps.append({"state": state[i].copy(), "env": env[i].copy(), "action": action[i].copy(),
                        "book_id": str(z["book_id"])})
    return eps, {"files": len(files), "episodes_all": n_all, "episodes_used": len(eps)}


def stats_of(eps: list[dict]) -> dict:
    out = {}
    for k in ("state", "env", "action"):
        x = np.concatenate([e[k] for e in eps], 0).astype(np.float64)
        out[f"{k}_mean"] = x.mean(0).astype(np.float32)
        out[f"{k}_std"] = np.maximum(x.std(0), STD_FLOOR).astype(np.float32)
    return out


def main() -> None:
    args = parse_args()
    run_id, out, meta = new_run("act_train", {k: (str(v) if isinstance(v, Path) else [str(x) for x in v]
                                                   if isinstance(v, list) else v) for k, v in vars(args).items()})
    result: dict = {}
    error = None
    try:
        result = train(args, out)
    except BaseException as ex:
        error = ex
        result = {"error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()}
        raise
    finally:
        result["status"] = classify(bool(result.get("acceptance_ok", False)), error)
        save_result(out, meta, result)
        print(f"run_id={run_id} status={result['status']} out={out}")


def train(args, out: Path) -> dict:
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    files = demo_files(args.data)
    eps, info = load_episodes(files)
    if not eps:
        raise RuntimeError(f"使えるデモが無い: {info}")
    st = stats_of(eps)
    dev = "cuda"
    C = args.chunk
    lens = np.array([len(e["action"]) for e in eps])
    T = int(lens.max())
    ds, de, da = eps[0]["state"].shape[1], eps[0]["env"].shape[1], eps[0]["action"].shape[1]

    def pad(x, n):  # 最後の値を繰り返して長さ n に
        return np.concatenate([x, np.repeat(x[-1:], n - len(x), 0)], 0)

    S = torch.tensor(np.stack([(pad(e["state"], T) - st["state_mean"]) / st["state_std"] for e in eps]), device=dev)
    E = torch.tensor(np.stack([(pad(e["env"], T) - st["env_mean"]) / st["env_std"] for e in eps]), device=dev)
    A = torch.tensor(np.stack([(pad(e["action"], T + C) - st["action_mean"]) / st["action_std"] for e in eps]), device=dev)
    L = torch.tensor(lens, device=dev)

    cfg = ACTConfig(input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(ds,)),
                                    "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(de,))},
                    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(da,))},
                    chunk_size=C, n_action_steps=C, kl_weight=args.kl_weight, device=dev)
    pol = ACTPolicy(cfg).to(dev)
    pol.train()
    opt = torch.optim.AdamW(pol.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    ar = torch.arange(C, device=dev)
    meta = {"episode_ctrl_steps": int(T + 25), "ctrl_hz": 50, "chunk": C, "dims": {"state": ds, "env": de, "action": da},
            "data": [str(p) for p in args.data], **info}
    log = []
    t0 = time.time()

    def save(tag: str) -> Path:
        d = out / tag
        pol.save_pretrained(str(d))
        np.savez(d / "stats.npz", **st)
        (d / "meta.json").write_text(json.dumps({**meta, "step": step}, indent=2))
        return d

    step = 0
    for step in range(1, args.steps + 1):
        ei = torch.tensor(rng.integers(0, len(eps), args.batch), device=dev)
        ti = (torch.rand(args.batch, device=dev) * L[ei]).long()
        batch = {"observation.state": S[ei, ti], "observation.environment_state": E[ei, ti],
                 "action": A[ei[:, None], ti[:, None] + ar[None]],
                 "action_is_pad": torch.zeros(args.batch, C, dtype=torch.bool, device=dev)}
        loss, li = pol.forward(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 10.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1:
            row = {"step": step, "loss": float(loss), "l1": float(li["l1_loss"]), "kld": float(li.get("kld_loss", 0.0)),
                   "wall_s": round(time.time() - t0, 1)}
            log.append(row)
            print(json.dumps(row), flush=True)
        if args.save_every and step % args.save_every == 0:
            save(f"step_{step:06d}")
    final = save("final")
    (out / "loss.jsonl").write_text("\n".join(json.dumps(r) for r in log))
    return {"acceptance": "最後まで学習して checkpoint を保存した", "acceptance_ok": True, "final": str(final),
            "loss_first": log[0], "loss_last": log[-1], "wall_s": time.time() - t0, **meta}


if __name__ == "__main__":
    main()
