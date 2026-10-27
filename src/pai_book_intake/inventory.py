"""本棚に置いた本の記録（ローカル SQLite）。Genesis に依存しない。

1 冊を棚に置くたびに 1 行足す。置いた後に確かめた結果（立っているか・棚の中か・背表紙の出入り）も同じ行に残す。
置き方（pose）は standing（立てて差し込む。B4）か flat（平らに置く・積む。B8 の回答 c）。flat では stack_level が
下から何冊目か（0 = 棚の床）、top_z_m が置いた本の上面の高さ（棚の座標）。
記録は実行の出力先（named volume）に置く。本棚の中身を後で問い合わせられるように、本棚・位置で引ける形にする。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS shelf_placements (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  env INTEGER NOT NULL,
  box_id TEXT,
  shelf_id TEXT NOT NULL,
  book_id TEXT NOT NULL,
  isbn TEXT,
  title TEXT,
  value_jpy INTEGER,
  x_lo_m REAL NOT NULL,
  x_hi_m REAL NOT NULL,
  fingers TEXT NOT NULL,
  free_space_source TEXT NOT NULL,
  placed_sim_s REAL NOT NULL,
  upright INTEGER,
  inside INTEGER,
  spine_y_m REAL,
  tilt_deg REAL,
  pose TEXT NOT NULL DEFAULT 'standing',
  stack_level INTEGER,
  top_z_m REAL,
  spine_front INTEGER,
  note TEXT
);
CREATE INDEX IF NOT EXISTS idx_shelf ON shelf_placements(shelf_id, x_lo_m);
"""


@dataclass
class Placement:
    run_id: str
    env: int
    shelf_id: str
    book_id: str
    x_lo_m: float
    x_hi_m: float
    fingers: list[int]
    free_space_source: str
    placed_sim_s: float
    box_id: str | None = None
    isbn: str | None = None
    title: str | None = None
    value_jpy: int | None = None
    upright: bool | None = None
    inside: bool | None = None
    spine_y_m: float | None = None
    tilt_deg: float | None = None
    pose: str = "standing"
    stack_level: int | None = None
    top_z_m: float | None = None
    spine_front: bool | None = None
    note: str = ""


class Inventory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)

    def add(self, p: Placement) -> int:
        row = asdict(p)
        row["fingers"] = json.dumps(p.fingers)
        for k in ("upright", "inside", "spine_front"):
            row[k] = None if row[k] is None else int(bool(row[k]))
        cols = ",".join(row)
        cur = self.db.execute(f"INSERT INTO shelf_placements ({cols}) VALUES ({','.join('?' * len(row))})",
                              tuple(row.values()))
        self.db.commit()
        return int(cur.lastrowid)

    def on_shelf(self, shelf_id: str, env: int | None = None) -> list[dict]:
        q = "SELECT * FROM shelf_placements WHERE shelf_id = ?"
        args: list = [shelf_id]
        if env is not None:
            q += " AND env = ?"
            args.append(env)
        cur = self.db.execute(q + " ORDER BY env, x_lo_m", args)
        names = [c[0] for c in cur.description]
        out = []
        for r in cur.fetchall():
            d = dict(zip(names, r))
            d["fingers"] = json.loads(d["fingers"])
            out.append(d)
        return out

    def close(self) -> None:
        self.db.close()
