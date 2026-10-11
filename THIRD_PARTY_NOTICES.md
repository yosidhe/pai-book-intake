# Third-party notices

このリポジトリが利用している外部のソフトウェア・データ・アセット。採否とライセンスの確認は 2026-10-11 に実施。

| 名前 | 用途 | ライセンス | 入手元 | 備考 |
|---|---|---|---|---|
| Genesis (genesis-world 1.4.3) | 物理シミュレーション・描画 | Apache-2.0 | https://github.com/Genesis-Embodied-AI/Genesis | pip で導入。リポジトリには同梱しない |
| Franka Emika Panda モデル（Genesis 同梱 `xml/franka_emika_panda/`） | ロボットアーム | Apache-2.0（同フォルダの README による） | Genesis パッケージ内 | 実行時に読み込む。メッシュはコピーしない。手首の先は自作のグリッパに置き換える |
| zxing-cpp 3.1.1 | バーコードの生成・読み取り | Apache-2.0 | https://github.com/zxing-cpp/zxing-cpp | pip で導入 |
| PyTorch 2.8.0 | 計算基盤 | BSD-3-Clause | https://pytorch.org | pip で導入 |
| Poly Haven のアセット（予定） | 箱・棚・回収箱の見た目 | CC0 | https://polyhaven.com/license | 採用したアセット名をここに追記する |
| Kenney Conveyor Kit（予定） | コンベアの見た目 | CC0 | https://kenney-assets.itch.io/conveyor-kit | 搬送の物理は自作 |
| 国立国会図書館サーチの書誌データ（予定） | 書籍10冊の書誌 | NDL 作成の書誌は CC BY 4.0 互換 | https://ndlsearch.ndl.go.jp/help/api | 出典を表示する。価格は架空。表紙は自作のダミーで、実在の書影は使わない |
