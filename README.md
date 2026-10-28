# pai-book-intake

中古書店の買取の作業（段ボールから本を取り出す → バーコードで査定する → 本棚か廃棄箱へ仕分ける）を、
複数のロボットが工程を分けて進めるシミュレーション（Genesis）。後工程は前工程の仕上がりを確かめて修正を頼み、
オブザーバが全体を見て、おかしいときは箱ごと止めて報告する。Physical AI 応用1講座の最終課題。

映像と数値はすべてシミュレーションのもの。

## 構成

| 担当 | 役目 | コード |
|---|---|---|
| ロボット A（固定） | 箱の平積みの一番上の本を取り、査定台に置く。手は「背をゴムで挟んで数 mm 浮かせ、すき間に爪を入れ、上から板で押さえる」形 | `pinch.py`・`hand_design.py` |
| 査定台とカメラ | 真上のカメラで本のバーコードを読む（zxing-cpp） | `vision.py`・`inspection.py` |
| 査定 | ISBN → カタログ（10 冊）→ 本棚か廃棄か。ルールで決める（LLM は使わない） | `appraisal.py`・`release.py` |
| ロボット B と台車 | 運んでよい本を取り、廃棄箱へ平らに置く。本棚なら台車で移動し、深さカメラで空きを見て置き、記録する | `p2_line.py`・`inventory.py` |
| 本棚 | 指で 1 冊ずつ支える棚（ブックキーパー）を横に長くしたもの。締切版は平らに積む | `keeper.py` |
| オブザーバ | 全部のメッセージを見る。止める条件（バーコードが続けて読めない・続けて失敗・本棚が満杯）と箱ごとの報告書 | `observer.py`・`bus.py` |
| A の方策の学習 | スクリプトの動きをデモにして、LeRobot の ACT（模倣学習）で A の取り出しを学習する。観測は状態だけ（画像なし） | `act_a.py`・`act_train.py` |
| 改善の材料 | 実行の記録を数えて、LLM に「次に直すこと」「次に学習させる場面」を頼むための材料を作る | `improve.py` |

## 実行

Docker（NVIDIA GPU）で動かす。出力は Docker の named volume `pai-outputs` の `<試行>/<run_id>/` に入る
（`result.json`・`bus.jsonl`・報告書・画像・動画）。

```bash
docker compose build

# 買取ライン（1 環境・4 冊・動画つき）。A と B は並行して動く（--sequential で 1 冊ずつ）
docker compose run --rm sim python -m pai_book_intake.p2_line --n-envs 1 --books 4 --video

# 10 冊。裏向きの本を混ぜる
docker compose run --rm sim python -m pai_book_intake.p2_line --n-envs 1 --books 10 --face-down book-06 --video \
    --pile book-10,book-09,book-08,book-07,book-01,book-04,book-03,book-05,book-06,book-02

# わざと失敗させる: 2 冊目から点検カメラの画像を汚す → 修正依頼 → 人の判断待ち → オブザーバが箱ごと止める
docker compose run --rm sim python -m pai_book_intake.p2_line --n-envs 1 --books 5 --inject-camera-dirty-from 1 --video \
    --pile book-09,book-04,book-10,book-03,book-02

# ACT: デモを集める → 学習 → スクリプト版と同じ条件で評価
docker compose run --rm sim python -m pai_book_intake.act_a collect --n-envs 32 --seed 1 --books 10
docker compose run --rm sim python -m pai_book_intake.act_train --data /workspace/outputs/act_collect/<run_id> --steps 150000 --save-every 10000
docker compose run --rm sim python -m pai_book_intake.act_a eval --policy scripted --n-envs 50
docker compose run --rm sim python -m pai_book_intake.act_a eval --policy act --ckpt /workspace/outputs/act_train/<run_id>/final --n-envs 50

# 改善の材料（digest.json と LLM への頼み方 prompt.md）
docker compose run --rm sim python -m pai_book_intake.improve --runs /workspace/outputs/p2_line/<run_id> /workspace/outputs/act_eval/<run_id>
```

評価の条件: 箱の中の本の初期位置を ±10 mm・±5° ばらつかせる（seed 1000・50 環境）。デモ集めは ±20 mm・±10°。

## 判定

- 各 run の `result.json` の `status` は `succeeded`（受入の条件を満たした）・`rejected`（最後まで動いたが満たさなかった）・`failed`（例外）の 3 つ
- 工程ごとの成功率（取り出し・読取・査定・運搬・空き認識・棚に置く・廃棄）と全体は `stage_rates_env0` に出る

## 制約

- 描画は CPU（WSL2 の Docker）なので、カメラを使うのは環境 0 だけ。環境 0 以外は、バーコードと本棚の空きを正解値から決めている（`result.json` に `source` として残す）
- ACT の観測は状態だけ。本の位置と向きはシミュの正解値を使う（スクリプト版も同じ）
- まだ無いもの（`not_implemented`）: 裏返し・箱の中の縦置きの本・本棚に立てて差し込む

## ライセンス

外部の資産と依存は `THIRD_PARTY_NOTICES.md` に書いてある。
