# 書籍カタログ（v1）

ライン全体（取り出し → バーコード読み取り → 査定 → 棚／回収箱）で使う 10 冊のデータ。

| ファイル | 中身 |
|---|---|
| `books_v1.json` | 完成したカタログ（生成物。手で直さない） |
| `pricing_v1.yaml` | 手書きの入力。架空の買取価格・状態・表面・重さの誤差と、査定の規則 |

## 作り方

```bash
docker compose run --rm sim python tools/fetch_ndl.py      # NDL サーチから書誌を取得 → outputs/ndl_raw/
docker compose run --rm sim python tools/build_catalog.py  # 検証して books_v1.json を書く
docker compose run --rm sim python tools/make_covers.py    # ダミー表紙 → outputs/covers/（読み戻し確認つき）
docker compose run --rm sim python -m pytest tests/test_catalog.py -q
```

## 出典（書誌データ）

出典：国立国会図書館書誌データ（2026年10月11日に取得）。国立国会図書館サーチの API（SRU, `https://ndlsearch.ndl.go.jp/api/sru`、データプロバイダ「国立国会図書館蔵書」`iss-ndl-opac`）を利用した。

- 国立国会図書館が作成し提供する書誌データの二次利用条件は、クリエイティブ・コモンズ 表示 4.0 国際 パブリック・ライセンス（CC BY 4.0）と互換性がある（[API 提供対象データプロバイダ一覧](https://ndlsearch.ndl.go.jp/help/api/provider)、[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ja)）。
- 書誌から題名・著者・出版者・出版年・ページ数・大きさを抜き出して整形した。寸法・推定重さはこのプロジェクトが計算したもので、国立国会図書館が作成したものではない。
- 国立国会図書館以外の機関が作成した書誌（`R100000002` 以外）は使っていない（`build_catalog.py` が検査する）。

## 架空の値

**買取価格・状態・表面・`weight_g_true` はすべて架空。** 特定の店舗やマーケットプレイスの価格は参照していない。
中古書の一般的な傾向（流通量の多い古い文庫は値が付きにくい、全集や大型本は値が残りやすい、水濡れ・書き込みは買い取れない）を、一般知識として大まかに写しただけ。
表紙もプロジェクトで作るダミーで、実在の書影は使わない。

## 各項目

| 項目 | 意味 |
|---|---|
| `id` | ライン内の識別子（`book-01`〜`book-10`） |
| `isbn13` / `ean13` | ISBN-13。表紙のバーコード（EAN-13）は同じ数字。NDL に ISBN-10 しか無い本は変換した |
| `bib.*` | NDL の書誌そのまま: 題名・巻・シリーズ・著者（典拠形）・責任表示・出版者・出版年・形態（`extent`、例 `563p ; 15cm`）・NDL の書誌 URL。`raw_file` / `raw_sha256` は取得した生応答 |
| `physical.pages` / `height_cm_ndl` | `extent` から読んだページ数と高さ（cm、NDL の慣行で切り上げ） |
| `physical.format` / `format_ja` / `binding` | 高さから当てはめた判型と製本（soft=並製 / hard=上製） |
| `physical.dims_mm` | 高さ・幅・厚さ [mm] |
| `physical.weight_g_estimated` | **つかむ前に**判型とページ数だけで出す推定重さ [g]。把持力はこれで決める |
| `physical.weight_g_true` | 架空の「本当の重さ」。持ち上げた後に推定と比べる |
| `physical.weight_true_error` | `weight_g_true = weight_g_estimated × (1 + この値)`（±0.05〜0.15） |
| `physical.surface` | 表紙の表面。`smooth`（つや・PP 加工）/ `rough`（布・マット紙） |
| `appraisal.condition` | A 美品 / B 良好 / C 可（ヤケ・スレ） / D 難あり（水濡れ・書き込み） |
| `appraisal.base_buyback_yen` | 美品だったときの架空の買取価格 [円] |
| `appraisal.buyback_yen` | 状態を掛けた架空の買取価格 [円] |
| `appraisal.expected_decision` | 正解の仕分け。`shelf`（棚）/ `discard`（回収箱） |
| `appraisal.borderline` | しきい値から ±12 円以内（境界ぎりぎりの例） |

## 査定の規則

```
buyback_yen = round(base_buyback_yen × 状態係数)   状態係数: A=1.0, B=0.8, C=0.5, D=0.0
buyback_yen >= 50 円 なら shelf、未満なら discard
```

規則は `pricing_v1.yaml` の `decision_rule` と `tools/catalog_model.py` の `appraise()` にあり、`books_v1.json` にも同じものを写している。

## 寸法と推定重さの式

式の実体は `tools/catalog_model.py`（`FORMATS`, `WEIGHT_MODEL`, `estimate_dims_weight()`）。`books_v1.json` の `weight_model` / `format_table` にも同じ値を写している。

1. **判型**: NDL の高さ（cm）から一般的な判型を当てはめる（仮定）。15cm → 文庫 148×105、16cm → 文庫（やや大）152×106、18cm → 新書判 173×105、19cm → 四六判並製 188×127、20cm → 四六判上製 194×133、21cm → A5判上製 210×148、26cm → B5判並製 257×182（mm）
2. **厚さ**: `本文の厚さ = ceil(ページ数 / 2) × 紙 1 枚の厚さ`。紙 1 枚は文庫・新書 0.075 mm、それ以外 0.090 mm（仮定）。表紙 1 枚は並製 0.3 mm、上製 2.5 mm（仮定）。`厚さ = 本文の厚さ + 2 × 表紙`
3. **推定重さ**: `本文 = 高さ × 幅 × 本文の厚さ × 紙の密度 0.75 g/cm³`、`表紙 = 2 × 高さ × 幅 × 面密度`（並製 250 g/m²、上製 1200 g/m²）、`推定重さ = 本文 + 表紙`（仮定）

例: 吾輩は猫である（563p, 15cm）→ 282 枚 × 0.075 mm = 21.15 mm、14.8 × 10.5 × 2.115 cm³ × 0.75 = 246.5 g、表紙 7.8 g → **254.3 g**、厚さ 21.8 mm。

紙の厚さ・密度・判型の当てはめは一般的な値を置いた仮定で、実物を量ってはいない。
