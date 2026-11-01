# Third-party notices

このリポジトリが利用している外部のソフトウェア・データ・アセット。採否とライセンスの確認は 2026-10-11 に実施。

| 名前 | 用途 | ライセンス | 入手元 | 備考 |
|---|---|---|---|---|
| Genesis (genesis-world 1.4.3) | 物理シミュレーション・描画 | Apache-2.0 | https://github.com/Genesis-Embodied-AI/Genesis | pip で導入。リポジトリには同梱しない |
| Franka Emika Panda モデル（Genesis 同梱 `xml/franka_emika_panda/`） | ロボットアーム | Apache-2.0（同フォルダの README による） | Genesis パッケージ内 | 実行時に読み込む。メッシュはコピーしない。手首の先は自作のグリッパに置き換える |
| LeRobot 0.4.4（ACT のモデルだけ） | ロボット A の取り出しの模倣学習（ACT） | Apache-2.0 | https://github.com/huggingface/lerobot | 2026-10-28 に追加。pip で `--no-deps` で導入し、import に要る依存だけを `docker/requirements-act.txt` に固定（einops MIT・draccus MIT・huggingface-hub / safetensors / datasets / accelerate / diffusers Apache-2.0・deepdiff MIT・termcolor MIT・jsonlines BSD-3-Clause・pyserial BSD-3-Clause・packaging Apache-2.0/BSD）。学習済みの重み・データセットは使わず、デモはこのシーンで自分で集めた |
| torchvision 0.23.0 | ACT の依存（画像の部品は使わない） | BSD-3-Clause | https://github.com/pytorch/vision | 2026-10-28 に追加 |
| zxing-cpp 3.1.1 | バーコードの生成・読み取り | Apache-2.0 | https://github.com/zxing-cpp/zxing-cpp | pip で導入 |
| PyTorch 2.8.0 | 計算基盤 | BSD-3-Clause | https://pytorch.org | pip で導入 |
| Poly Haven のアセット（予定） | 箱・棚・回収箱の見た目 | CC0 | https://polyhaven.com/license | 採用したアセット名をここに追記する |
| Kenney Conveyor Kit（予定） | コンベアの見た目 | CC0 | https://kenney-assets.itch.io/conveyor-kit | 搬送の物理は自作 |
| 国立国会図書館書誌データ（国立国会図書館サーチ API） | 書籍10冊の書誌（`assets/catalog/books_v1.json`） | 国立国会図書館ウェブサイトのコンテンツ利用規約に基づき、営利・非営利を問わず無償で利用可能（[書誌データ等の利用について](https://www.ndl.go.jp/jp/data/data_service/quickguide/index.html)。2026-11-01 に確認） | https://ndlsearch.ndl.go.jp/api/sru （SRU, dpid=iss-ndl-opac） | 出典：国立国会図書館書誌データ（2026年10月11日に取得）。国立国会図書館サーチのAPIを利用。題名・著者・出版者・出版年・ページ数・大きさを抜き出して整形し、寸法・推定重さは本プロジェクトが計算した（国立国会図書館が作成したものではない）。NDL 以外の機関が作成した書誌は使わない。価格・状態・本当の重さは架空。表紙は自作のダミーで、実在の書影は使わない |
