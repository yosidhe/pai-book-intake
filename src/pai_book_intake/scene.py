"""買取ラインのシーン（Genesis）: ロボット 2 台・段ボール箱と本の山・受け渡し台・本棚・廃棄箱・カメラ 2 台。

配置の数値は layout.py にまとめてある。

受け渡し部はコンベアではなく固定の台にした（P1 の判断）
  - 裏返し（試行2-7）は「台の縁からのはみ出し」を前提にしており、固定の台の縁で置き方の標準を決めるほうが早い
  - 動くコンベアは、Genesis では運動学的に動かす板か、本へ速度を与える仕掛けが要り、受け渡しの精度の検証が増える
  - バーコードの撮影も、止まった本を真上から撮るほうが確実（試行2-8 は静止した本で 95%）
  コンベアは、台の受け渡しが通しで安定してから（P2 以降）差し替える候補として残す。

描画は環境 0 だけ（rendered_envs_idx=[0]）。WSL2 の Docker では描画が CPU（mesa）になるため。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import genesis as gs

from . import layout as L
from .book_assets import make_book_assets
from .books import SURFACES, BookSpec
from .geometry import quat_mul, quat_to_mat, quat_x, quat_z
from .tool import HOME, Robot, World, build_robot_xml

DT = 0.01
CARD = (0.72, 0.56, 0.38)
WOOD = (0.62, 0.45, 0.28)
GRAY = (0.80, 0.80, 0.78)
BIN_COLOR = (0.35, 0.37, 0.40)


@dataclass(frozen=True)
class PileConfig:
    """箱の中の本の山。flat は下から上の順（最後が一番上）。standing は +y の壁から順に立てかける。"""

    flat: tuple[BookSpec, ...]
    standing: tuple[BookSpec, ...] = ()
    face_down: frozenset[str] = frozenset()  # 裏向き（表紙が下）に入っている本の book_id
    # 初期配置のばらつき（試行2-6 の範囲 ±4 cm / ±3 cm / ±20° の内側）
    jitter_xy: tuple[float, float] = (0.03, 0.02)
    jitter_yaw_deg: float = 15.0
    lean_deg: float = 12.0  # 立てた本の傾き（鉛直から）
    lean_jitter_deg: float = 3.0


@dataclass(frozen=True)
class SceneConfig:
    n_envs: int = 2
    barcode_res: tuple[int, int] = (1920, 1440)
    overview_res: tuple[int, int] = (960, 540)
    tool_v2: bool = True


@dataclass
class BookBody:
    spec: BookSpec
    entity: object
    role: str  # "flat" / "standing"
    index: int  # 山の中の順番


@dataclass
class Line:
    scene: object
    world: World
    robot_a: Robot
    robot_b: Robot
    books: list[BookBody]
    pile: PileConfig
    barcode_cam: object
    overview_cam: object
    cfg: SceneConfig
    meta: dict = field(default_factory=dict)

    def book(self, book_id: str) -> BookBody:
        for b in self.books:
            if b.spec.book_id == book_id:
                return b
        raise KeyError(book_id)


def _add_box(scene, size, pos, color, euler=(0.0, 0.0, 0.0)):
    return scene.add_entity(gs.morphs.Box(size=tuple(size), pos=tuple(float(v) for v in pos), euler=euler, fixed=True),
                            surface=gs.surfaces.Default(color=color))


def _add_local_box(scene, frame, size_local, pos_local, color):
    """B の座標で書いた箱（向きは B に合わせる）を置く。"""
    p = frame.to_world(np.asarray(pos_local, dtype=float))
    return _add_box(scene, size_local, p, color, euler=(0.0, 0.0, math.degrees(frame.yaw)))


def build_line(cfg: SceneConfig, pile: PileConfig, out_dir: Path) -> Line:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=4),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0], show_world_frame=False),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    # 段ボール箱（試行2・3 と同じ寸法・位置）
    cx, cy = L.BOX_C
    ix, iy = L.BOX_IN
    h, t, fl = L.BOX_WALL_H, L.BOX_WALL_T, L.BOX_FLOOR
    for size, pos in [
        ((ix + 2 * t, iy + 2 * t, fl), (cx, cy, fl / 2)),
        ((t, iy + 2 * t, h), (cx - ix / 2 - t / 2, cy, h / 2)),
        ((t, iy + 2 * t, h), (cx + ix / 2 + t / 2, cy, h / 2)),
        ((ix, t, h), (cx, cy - iy / 2 - t / 2, h / 2)),
        ((ix, t, h), (cx, cy + iy / 2 + t / 2, h / 2)),
    ]:
        _add_box(scene, size, pos, CARD)

    fb = L.FRAME_B
    # 受け渡し台（B の座標: 縁 TABLE_EDGE_B から奥へ TABLE_DEPTH）
    _add_local_box(scene, fb, (L.TABLE_DEPTH, 2 * L.TABLE_HALF_W, L.TABLE_TOP),
                   (L.TABLE_EDGE_B + L.TABLE_DEPTH / 2, 0.0, L.TABLE_TOP / 2), GRAY)

    # 本棚: 棚板（下は箱で埋める）・背板・側板。上は開いている（上から平置きする 1 段）
    x0, x1 = L.SHELF_SLOT_X[0] - L.SHELF_SLOT_HALF, L.SHELF_SLOT_X[-1] + L.SHELF_SLOT_HALF
    y0, y1 = L.SHELF_DEPTH_C - L.SHELF_DEPTH / 2, L.SHELF_DEPTH_C + L.SHELF_DEPTH / 2
    pt = 0.015
    _add_local_box(scene, fb, (x1 - x0, y1 - y0, L.SHELF_BOARD_TOP), ((x0 + x1) / 2, (y0 + y1) / 2, L.SHELF_BOARD_TOP / 2),
                   WOOD)
    ph = L.SHELF_PANEL_H
    _add_local_box(scene, fb, (x1 - x0, pt, ph), ((x0 + x1) / 2, y1 + pt / 2, L.SHELF_BOARD_TOP + ph / 2), WOOD)
    for xs in (x0 - pt / 2, x1 + pt / 2):
        _add_local_box(scene, fb, (pt, y1 - y0 + pt, ph), (xs, (y0 + y1) / 2 + pt / 2, L.SHELF_BOARD_TOP + ph / 2), WOOD)

    # 廃棄箱（上が開いた箱）
    bx, by = L.BIN_C
    bw, bd = L.BIN_IN
    bh, bt = L.BIN_WALL_H, 0.01
    for size, pos in [
        ((bw + 2 * bt, bd + 2 * bt, L.BIN_FLOOR_T), (bx, by, L.BIN_FLOOR_T / 2)),
        ((bt, bd + 2 * bt, bh), (bx - bw / 2 - bt / 2, by, bh / 2)),
        ((bt, bd + 2 * bt, bh), (bx + bw / 2 + bt / 2, by, bh / 2)),
        ((bw, bt, bh), (bx, by - bd / 2 - bt / 2, bh / 2)),
        ((bw, bt, bh), (bx, by + bd / 2 + bt / 2, bh / 2)),
    ]:
        _add_local_box(scene, fb, size, pos, BIN_COLOR)

    # 本（表紙にバーコードを印字したメッシュ）。初期位置は reset_pile で環境ごとに決め直す
    books: list[BookBody] = []
    for role, specs in (("flat", pile.flat), ("standing", pile.standing)):
        for i, spec in enumerate(specs):
            obj, tex = make_book_assets(spec, out_dir)
            s = SURFACES[spec.surface]
            ent = scene.add_entity(
                gs.morphs.Mesh(file=str(obj), pos=(cx, cy, 0.5 + 0.05 * len(books))),
                material=gs.materials.Rigid(rho=spec.density, friction=s["friction"]),
                surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(tex))),
            )
            books.append(BookBody(spec, ent, role, i))

    # ロボット A（箱の前）と B（台をはさんで向かい）
    xml = build_robot_xml(out_dir, rounded=cfg.tool_v2)
    ent_a = scene.add_entity(gs.morphs.MJCF(file=str(xml), pos=(*L.ROBOT_A_BASE, 0.0),
                                            euler=(0.0, 0.0, math.degrees(L.ROBOT_A_YAW))))
    ent_b = scene.add_entity(gs.morphs.MJCF(file=str(xml), pos=(*L.ROBOT_B_BASE, 0.0),
                                            euler=(0.0, 0.0, math.degrees(L.ROBOT_B_YAW))))

    # カメラ: バーコード用（台の真上、試行2-8 と同じ高さ）と、全体を撮る動画用
    hp = L.handoff_pose(0.13, 0.02).center_world
    barcode_cam = scene.add_camera(res=tuple(cfg.barcode_res), pos=(hp[0], hp[1], L.TABLE_TOP + L.BARCODE_CAM_H),
                                   lookat=(hp[0], hp[1], L.TABLE_TOP), up=(1.0, 0.0, 0.0), fov=40, env_idx=0)
    overview_cam = scene.add_camera(res=tuple(cfg.overview_res), pos=(1.55, -0.65, 1.85), lookat=(0.05, 0.62, 0.15),
                                    fov=48, env_idx=0)

    scene.build(n_envs=cfg.n_envs)
    world = World(scene, cfg.n_envs, DT)
    home_b = HOME.copy()
    home_b[0] = L.B_PARK_JOINT1
    robot_a = Robot(world, ent_a, "robot-a-pick", base_xy=L.ROBOT_A_BASE)
    robot_b = Robot(world, ent_b, "robot-b-sort", home=home_b, base_xy=L.ROBOT_B_BASE)
    robot_a.setup()
    robot_b.setup()
    return Line(scene, world, robot_a, robot_b, books, pile, barcode_cam, overview_cam, cfg,
                meta={"conveyor": False, "handoff": "static_table"})


def _standing_poses(specs, lean, n, rng, lean_jitter) -> list[tuple[np.ndarray, np.ndarray]]:
    """立てた本を +y の壁から順に、平行に立てかけた姿勢（環境ごと）。
    傾きのばらつきは環境ごとに 1 つ（同じ箱の本は平行。本ごとに変えると上端が隣に食い込む）。"""
    cx, cy = L.BOX_C
    th = lean + rng.uniform(-lean_jitter, lean_jitter, n)
    wall_y = cy + L.BOX_IN[1] / 2
    out = []
    prev_plane = None  # 前の本の表紙側の面: n_b·p = prev_plane
    for spec in specs:
        # 本の y（天地）を鉛直に、z（厚み）を -y へ向けてから、上端を +y の壁へ th だけ倒す
        q = quat_x(math.pi / 2 - th)
        R = quat_to_mat(q)
        w, hgt, t = spec.size_m
        corners = np.array([[sx * w / 2, sy * hgt / 2, sz * t / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        cw = np.einsum("nij,kj->nki", R, corners)  # (n, 8, 3)
        z = L.BOX_FLOOR + 0.001 - cw[:, :, 2].min(axis=1)
        nb = R[:, :, 2] * -1  # 裏表紙側（+y を向く面）の法線
        if prev_plane is None:
            y = wall_y - 0.002 - cw[:, :, 1].max(axis=1)
        else:
            # 裏表紙の面を、前の本の表紙の面から 2 mm 離す: nb·c = prev_plane - 0.002 - t/2
            target = prev_plane - 0.002 - t / 2
            y = (target - nb[:, 2] * z) / nb[:, 1]
        c = np.stack([np.full(n, cx), y, z], -1)
        c[:, 0] += rng.uniform(-0.02, 0.02, n)
        prev_plane = np.einsum("ni,ni->n", nb, c) - t / 2
        out.append((c, q))
    return out


def reset_pile(line: Line, rng: np.random.Generator) -> None:
    """本を箱の中へ置き直す（環境ごとにばらつかせる）。"""
    n = line.cfg.n_envs
    p = line.pile
    sx, sy = L.FLAT_STACK_C
    z = np.full(n, L.BOX_FLOOR)
    jx, jy = p.jitter_xy
    jyaw = math.radians(p.jitter_yaw_deg)
    flat = [b for b in line.books if b.role == "flat"]
    for i, b in enumerate(flat):
        t = b.spec.thickness_m
        zc = z + t / 2 + 0.003
        pos = np.stack([sx + rng.uniform(-jx, jx, n), sy + rng.uniform(-jy, jy, n), zc], -1)
        q = quat_z(rng.uniform(-jyaw, jyaw, n))
        if b.spec.book_id in p.face_down:
            q = quat_mul(q, np.broadcast_to(np.array([0.0, 1.0, 0.0, 0.0]), (n, 4)))  # x 軸まわり 180°
        b.entity.set_pos(pos)
        b.entity.set_quat(q)
        z = zc + t / 2
    standing = [b for b in line.books if b.role == "standing"]
    poses = _standing_poses([b.spec for b in standing], math.radians(p.lean_deg), n, rng,
                            math.radians(p.lean_jitter_deg))
    for b, (c, q) in zip(standing, poses):
        b.entity.set_pos(c)
        b.entity.set_quat(q)
