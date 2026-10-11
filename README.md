# pai-book-intake

中古書店の買取フロー（段ボールから取り出す → 査定する → 本棚か廃棄箱へ仕分ける）を、
複数のロボットが互いの工程を評価し合いながら進めるシミュレーション。
Physical AI 応用1講座の最終課題。

> 開発中。現在はシミュレータ（Genesis）の試行段階。

## 実行

```bash
docker compose build
docker compose run --rm sim python spikes/genesis_trial.py
```

出力（動画・ログ）は Docker の named volume `pai-outputs` に保存される。
