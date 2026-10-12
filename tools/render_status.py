"""開発ステータスの 1 枚図（配置・役割・処理フロー・動作状況・残タスク）を SVG に描く。

元データは docs/status/line_status.yaml。図は手で直さず、YAML を直して描き直す。
色は状態にだけ使い、形（●◐◇○✕）と文字でも状態を示す（モノクロ印刷でも読めるように）。

使い方: python tools/render_status.py [--src docs/status/line_status.yaml] --out docs/status/line_status.svg [--out 別の場所.svg]
"""

from __future__ import annotations

import argparse
import math
import unicodedata
from html import escape
from pathlib import Path

import yaml

W, H = 1680, 1060
FONT = "'Yu Gothic UI','Meiryo','Noto Sans JP','Hiragino Sans',sans-serif"
INK, SUB, LINE, PANEL = "#1f2328", "#57606a", "#d0d7de", "#f6f8fa"
STATUS = {  # 色・記号・言葉
    "done": ("#1a7f37", "完成"),
    "doing": ("#0969da", "作業中"),
    "proposed": ("#9a6700", "提案・未確認"),
    "todo": ("#6e7781", "未着手"),
    "blocked": ("#cf222e", "廃止予定・停止"),
}


def text_w(s: str, size: float) -> float:
    """文字列のおおよその幅（全角 1・半角 0.55）。"""
    return sum(size if unicodedata.east_asian_width(c) in "WFA" else size * 0.55 for c in s)


def wrap(s: str, width: float, size: float) -> list[str]:
    lines, cur = [], ""
    for c in s:
        if text_w(cur + c, size) > width and cur:
            lines.append(cur)
            cur = c
        else:
            cur += c
    return lines + ([cur] if cur else [])


def t(x, y, s, size=13, weight="normal", fill=INK, anchor="start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{escape(str(s))}</text>')


def glyph(x, y, status, r=6) -> str:
    """状態の記号（色だけに頼らない）。x, y は中心。"""
    c = STATUS[status][0]
    if status == "done":
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}"/>'
    if status == "doing":
        return (f'<circle cx="{x}" cy="{y}" r="{r}" fill="#fff" stroke="{c}" stroke-width="2"/>'
                f'<path d="M{x},{y - r} A{r},{r} 0 0 1 {x},{y + r} Z" fill="{c}"/>')
    if status == "proposed":
        return (f'<path d="M{x},{y - r - 1} L{x + r + 1},{y} L{x},{y + r + 1} L{x - r - 1},{y} Z" '
                f'fill="#fff" stroke="{c}" stroke-width="2"/>')
    if status == "todo":
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="#fff" stroke="{c}" stroke-width="2"/>'
    return (f'<path d="M{x - r},{y - r} L{x + r},{y + r} M{x + r},{y - r} L{x - r},{y + r}" '
            f'stroke="{c}" stroke-width="2.5"/>')


def badge(x, y, status, size=12) -> str:
    c, word = STATUS[status]
    return glyph(x + 6, y - size * 0.35, status) + t(x + 17, y, word, size, "bold", c)


def dash(status) -> str:
    return {"proposed": ' stroke-dasharray="7 4"', "todo": ' stroke-dasharray="2 4"',
            "blocked": ' stroke-dasharray="3 3"'}.get(status, "")


def render(d: dict) -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="{FONT}" style="font-variant-numeric: tabular-nums">',
           f'<rect width="{W}" height="{H}" fill="#fff"/>',
           '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
           f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{SUB}"/></marker></defs>']
    m = d["meta"]
    out.append(t(24, 40, m["title"], 24, "bold"))
    out.append(t(24, 66, f'更新 {m["updated"]}　｜　次の節目: {m["milestone_next"]}　｜　{m["deadline"]}', 13, fill=SUB))
    lx = 1120  # 凡例
    for i, (k, (c, word)) in enumerate(STATUS.items()):
        out.append(glyph(lx + i * 108, 36, k) + t(lx + i * 108 + 12, 41, word, 12, fill=INK))

    # ---- 配置（左） ----
    px0, py0, pw, ph = 20, 90, 600, 640
    out.append(f'<rect x="{px0}" y="{py0}" width="{pw}" height="{ph}" rx="6" fill="{PANEL}" stroke="{LINE}"/>')
    out.append(t(px0 + 14, py0 + 24, "配置と役割（上から見た図）", 15, "bold"))
    b = d["layout"]["bounds"]
    sc = min((pw - 40) / (b["xmax"] - b["xmin"]), (ph - 60) / (b["ymax"] - b["ymin"]))

    def P(x, y):  # 世界座標 → 図（世界の +x を右、+y を上に）
        return px0 + 20 + (x - b["xmin"]) * sc, py0 + 40 + (b["ymax"] - y) * sc

    for o in d["layout"]["objects"]:
        c = STATUS[o["status"]][0]
        cx, cy = P(o["x"], o["y"])
        if o["kind"] == "camera":
            out.append(f'<path d="M{cx},{cy - 9} L{cx + 9},{cy + 7} L{cx - 9},{cy + 7} Z" fill="#fff" stroke="{c}" stroke-width="2"/>')
            out.append(t(cx + 14, cy + 6, o["label"], 11, fill=SUB))
            continue
        # 図では world の x 幅 = w、y 幅 = h とする
        x0, y0 = P(o["x"] - o["w"] / 2, o["y"] + o["h"] / 2)
        ww, hh = o["w"] * sc, o["h"] * sc
        fill = "#fff" if o["status"] != "blocked" else "none"
        out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{ww:.1f}" height="{hh:.1f}" fill="{fill}" '
                   f'stroke="{c}" stroke-width="2"{dash(o["status"])}/>')
        if o["id"] == "table_old":  # 同じ場所の新しい点検台と重なるので、説明は図の下端にまとめる
            out.append(glyph(px0 + 22, py0 + ph - 18, o["status"], 5)
                       + t(px0 + 34, py0 + ph - 14, f'点線の赤枠＝{o["label"]}: {o.get("note", "")}（同じ場所を点検台にする）', 11, fill=c))
            continue
        out.append(glyph(x0 + 10, y0 + 14, o["status"], 5))
        for i, line in enumerate(wrap(o["label"], ww - 26, 12)):
            out.append(t(x0 + 20, y0 + 18 + i * 15, line, 12, "bold"))
        if o.get("note"):
            out.append(t(x0 + 6, y0 + hh - 8, o["note"], 11, fill=c))
    for r in d["layout"]["robots"]:
        c = STATUS[r["status"]][0]
        cx, cy = P(r["x"], r["y"])
        rad = 0.11 * sc
        a = math.radians(r["facing_deg"])
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rad:.1f}" fill="#fff" stroke="{c}" stroke-width="3"/>')
        out.append(f'<line x1="{cx + math.cos(a) * rad:.1f}" y1="{cy - math.sin(a) * rad:.1f}" x2="{cx + math.cos(a) * rad * 1.9:.1f}" '
                   f'y2="{cy - math.sin(a) * rad * 1.9:.1f}" stroke="{c}" stroke-width="3" marker-end="url(#arr)"/>')
        out.append(t(cx, cy + 5, r["label"].replace("ロボット ", ""), 15, "bold", c, "middle"))
        ty = cy + rad + 18 if r["y"] < 0.5 else cy - rad - 40
        out.append(t(cx - 150 if r["x"] < 0.3 else cx, ty, f'{r["label"]}（{r["task"]}）', 12, "bold"))
        for i, line in enumerate(wrap(r["role"], 290, 11)):
            out.append(t(cx - 150 if r["x"] < 0.3 else cx, ty + 15 + i * 14, line, 11, fill=SUB))

    # ---- 処理フロー（右上） ----
    fx0, fy0, fw, fh = 640, 90, 1020, 560
    out.append(f'<rect x="{fx0}" y="{fy0}" width="{fw}" height="{fh}" rx="6" fill="{PANEL}" stroke="{LINE}"/>')
    out.append(t(fx0 + 14, fy0 + 24, "処理フローと開発ステータス", 15, "bold"))
    nodes = {n["id"]: n for n in d["flow"]["nodes"]}
    cw, ch = 188, 210
    pos = {}
    row = ["pick", "place", "scan", "appraise", "sort"]
    for i, nid in enumerate(row):
        pos[nid] = (fx0 + 14 + i * (cw + 14), fy0 + 48)
    pos["flip"] = (pos["scan"][0], fy0 + 48 + ch + 46)
    for nid, (x, y) in pos.items():
        n = nodes[nid]
        c = STATUS[n["status"]][0]
        hh = ch if nid != "flip" else 150
        out.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{hh}" rx="6" fill="#fff" stroke="{c}" '
                   f'stroke-width="2"{dash(n["status"])}/>')
        out.append(t(x + 10, y + 22, n["label"], 14, "bold"))
        out.append(t(x + 10, y + 40, n["actor"], 11, fill=SUB))
        out.append(badge(x + 8, y + 60, n["status"]))
        yy = y + 80
        for it in n["items"]:
            for j, line in enumerate(wrap(it, cw - 24, 11)):
                out.append(t(x + 12 if j else x + 10, yy, ("・" if j == 0 else "　") + line, 11))
                yy += 14
            yy += 2
    for e in d["flow"]["edges"]:
        a, z, lab = (e[0], e[1], "") if isinstance(e, list) else (e["from"], e["to"], e.get("label", ""))
        (ax, ay), (zx, zy) = pos[a], pos[z]
        if ay == zy:  # 横に並んだカード
            y = ay + 22
            out.append(f'<line x1="{ax + cw}" y1="{y}" x2="{zx}" y2="{y}" stroke="{SUB}" stroke-width="2" marker-end="url(#arr)"/>')
            if lab:  # カードの間は狭いので、カードの上の余白に書く
                out.append(t((ax + cw + zx) / 2, ay - 6, lab, 11, "bold", SUB, "middle"))
        else:  # 点検 ⇄ 裏返し（縦）
            down = zy > ay
            x = ax + (cw * 0.35 if down else cw * 0.65)
            y1, y2 = (ay + ch, zy) if down else (ay, zy + ch)
            out.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{SUB}" stroke-width="2" marker-end="url(#arr)"/>')
            out.append(t(x + (-6 if down else 6), (y1 + y2) / 2 + 4, lab, 11, "bold", SUB, "end" if down else "start"))
    # オブザーバ（全体を監視）
    ob = nodes["observe"]
    oy = fy0 + 48 + ch + 46
    ox = fx0 + 14
    c = STATUS[ob["status"]][0]
    out.append(f'<rect x="{ox}" y="{oy}" width="{2 * cw + 14}" height="150" rx="6" fill="#fff" stroke="{c}" stroke-width="2"{dash(ob["status"])}/>')
    out.append(t(ox + 10, oy + 22, ob["label"], 14, "bold"))
    out.append(t(ox + 10, oy + 40, f'{ob["actor"]}（全工程のメッセージを購読）', 11, fill=SUB))
    out.append(badge(ox + 8, oy + 60, ob["status"]))
    for i, it in enumerate(ob["items"]):
        out.append(t(ox + 10, oy + 82 + i * 16, "・" + it, 11))
    # 裏返しの右に、記録
    sx = pos["appraise"][0]
    out.append(f'<rect x="{sx}" y="{oy}" width="{2 * cw + 14}" height="150" rx="6" fill="#fff" stroke="{LINE}"/>')
    out.append(t(sx + 10, oy + 22, "記録（全工程 → bus.jsonl）", 14, "bold"))
    for i, s in enumerate(["メッセージは run_id 付きで bus.jsonl に先に書く", "修正依頼・好評価は前工程の records へ"]):
        out.append(t(sx + 10, oy + 46 + i * 18, "・" + s, 11))

    # ---- 動作状況と残タスク（下） ----
    by0 = 670
    out.append(f'<rect x="640" y="{by0}" width="1020" height="370" rx="6" fill="{PANEL}" stroke="{LINE}"/>')
    out.append(t(654, by0 + 24, "動作状況（直近の実行）", 15, "bold"))
    for i, r in enumerate(d["runs"]):
        y = by0 + 52 + i * 26
        out.append(glyph(662, y - 4, r["status"]) + t(676, y, r["label"], 12) + t(900, y, r["result"], 12, "bold"))
    out.append(t(1000, by0 + 24, "残タスク（期日順）", 15, "bold"))
    out.append(t(1000, by0 + 46, "期日", 11, "bold", SUB) + t(1052, by0 + 46, "担い手", 11, "bold", SUB)
               + t(1116, by0 + 46, "内容", 11, "bold", SUB))
    for i, k in enumerate(d["tasks"]):
        y = by0 + 70 + i * 28
        out.append(t(1000, y, k["due"], 12) + t(1052, y, k["owner"], 12, fill=SUB) + glyph(1124, y - 4, k["status"])
                   + t(1138, y, k["label"], 12) + t(1650, y, STATUS[k["status"]][1], 11, "bold", STATUS[k["status"]][0], "end"))
    out.append(t(24, H - 16, "元データ: docs/status/line_status.yaml ／ 描画: tools/render_status.py（図を手で直さない）", 11, fill=SUB))
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="docs/status/line_status.yaml")
    ap.add_argument("--out", action="append", required=True)
    a = ap.parse_args()
    d = yaml.safe_load(Path(a.src).read_text(encoding="utf-8"))
    bad = [s for s in [o["status"] for o in d["layout"]["objects"] + d["layout"]["robots"] + d["flow"]["nodes"]
                       + d["runs"] + d["tasks"]] if s not in STATUS]
    if bad:
        raise SystemExit(f"不明な status: {bad}（{list(STATUS)} のどれか）")
    svg = render(d)
    for o in a.out:
        Path(o).parent.mkdir(parents=True, exist_ok=True)
        Path(o).write_text(svg, encoding="utf-8", newline="\n")
        print(o)


if __name__ == "__main__":
    main()
