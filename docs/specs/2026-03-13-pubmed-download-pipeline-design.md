# PubMed Data Download & Sampling Pipeline Design

Status: Draft
Date: 2026-03-13
Owner: Yasuhiro Okamoto
Related: [ADR 0002 - Data Scope](../adr/0002-data-scope.md)

## Overview

PubMed/MEDLINE データをダウンロードし、ADR 0002 に基づいてフィルタリング・年層化サンプリング・MeSH カテゴリ最低カバレッジを適用して JSONL 形式で出力するパイプライン。

まずは playground 上で HuggingFace データセット経由の toy program として実装し、後に NLM FTP ベースライン (XML) 版に差し替え可能な設計とする。

## Architecture

2スクリプト構成のパイプライン。中間データ（raw JSONL）を契約としてスクリプト間を疎結合にする。

```
[1. Download]  -->  raw JSONL  -->  [2. Sample/Filter]  -->  final JSONL + audit log
 (download_hf.py)   (data/raw/)      (sample.py)            (data/processed/)
```

FTP 版への移行時は `download_ftp.py` + `convert_xml_to_jsonl.py` を追加して同じ `data/raw/*.jsonl` に出力すれば、`sample.py` は変更不要。

## Directory Structure

```
data_pipeline/
  pubmed_pipeline/
    download_hf.py          # HuggingFace DL script
    sample.py               # Filter + stratified sampling script
    config.yaml             # Pipeline configuration
    data/
      raw/                  # download_hf.py output
      processed/            # sample.py output
    README.md
```

## Intermediate JSONL Schema (Contract Between Scripts)

DL スクリプトの出力 = 整形スクリプトの入力。この形式がスクリプト間の契約となる。

```json
{
  "pmid": "12345678",
  "title": "Example Title",
  "abstract": "Example abstract text...",
  "authors": ["Author A", "Author B"],
  "publication_date": "2023-05-15",
  "mesh_terms": ["Neoplasms", "Lung Neoplasms"],
  "keywords": ["keyword1", "keyword2"],
  "publication_types": ["Journal Article", "Randomized Controlled Trial"],
  "language": "eng",
  "journal": "Journal Name"
}
```

Note: `publication_types` は下流の study type フィルタリング（statements.md Requirement 1）で必要になるため含める。

## Script 1: download_hf.py

### Purpose

HuggingFace の `ncbi/pubmed` データセットをダウンロードし、中間 JSONL に変換して `data/raw/` に出力する。

### Behavior

1. `datasets.load_dataset("ncbi/pubmed", streaming=True)` でストリーミングロード（全量をメモリに載せない）
2. config の `years` に該当する年のレコードのみ早期フィルタして書き出し（raw JSONL のサイズを抑える）
3. 各レコードを中間 JSONL スキーマにマッピング
4. `data/raw/pubmed_raw.jsonl` に書き出し（既存ファイルは上書き）

### HuggingFace Dataset Field Mapping

`ncbi/pubmed` データセットの各レコードには `MedlineCitation` XML が含まれる。`download_hf.py` 内で `xml.etree.ElementTree` を使ってパースし、以下のマッピングを行う:

| Output Field | XML Path / Source |
|---|---|
| `pmid` | `MedlineCitation/PMID` |
| `title` | `MedlineCitation/Article/ArticleTitle` |
| `abstract` | `MedlineCitation/Article/Abstract/AbstractText` (複数セクションは結合) |
| `authors` | `MedlineCitation/Article/AuthorList/Author` (LastName + ForeName) |
| `publication_date` | `MedlineCitation/Article/Journal/JournalIssue/PubDate` (Year-Month-Day, 欠損時は Year のみ) |
| `mesh_terms` | `MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName` |
| `keywords` | `MedlineCitation/KeywordList/Keyword` |
| `publication_types` | `MedlineCitation/Article/PublicationTypeList/PublicationType` |
| `language` | `MedlineCitation/Article/Language` |
| `journal` | `MedlineCitation/Article/Journal/Title` |

XML パースロジックはこのスクリプト内に閉じる。FTP 版の `convert_xml_to_jsonl.py` も同じマッピングを使うが、入力形式が異なるだけ。

### Data Volume Estimate

全 PubMed は 36M+ レコード。ストリーミング + 年フィルタ (2021-2025) により、DL対象は約 7-8M レコードに絞られる。raw JSONL はアブストラクト有無に関わらず書き出すため、推定 10-15GB 程度。ディスク容量に注意。

toy program として小規模テスト（例: 最初の 10,000 件のみ）で動作確認してからフル実行することを推奨。`--limit N` オプションで件数制限可能とする。

### Error Handling

- ネットワークエラー時は処理済み件数をログ出力して終了（部分ファイルが残る）
- 再実行時はファイルを上書き（idempotent）

## Script 2: sample.py

### Purpose

中間 JSONL を読み込み、ADR 0002 に従ってフィルタリング・サンプリングを行い、最終 JSONL と監査ログを出力する。

### Processing Flow (ADR 0002 Order of Operations)

1. `data/raw/*.jsonl` を読み込み
2. Filter: `publication_date` の年が対象年 (2021-2025) に含まれるか
3. Filter: `language == "eng"`
4. Filter: `abstract` が非空
5. 年ごとにグループ化し、各年の母数 (population) を算出・記録
6. 各年・各 MeSH トップカテゴリ (10カテゴリ) で最低 `min_coverage.per_category_per_year` (500) 件を確保
   - カテゴリ候補が不足する場合は全件取得しログに記録（エラーにしない）
7. 残り枠をランダムサンプリングで埋めて各年 `n_max / len(years)` (20,000) 件に
8. (Optional) クロス年 top-up: 特定の年が quota 未達の場合、余剰のある年から追加サンプリングして合計 `n_max` に近づける。toy 実装では省略可。
9. 固定シード (42) で再現性確保
10. 出力:
    - `data/processed/sampled.jsonl` — 最終サンプリング結果
    - `data/processed/audit_log.json` — 監査ログ

### MeSH Category Matching

MeSH terms はツリー構造を持つ。toy program では **部分文字列マッチ** で判定する（例: MeSH term "Lung Neoplasms" はカテゴリ "Neoplasms" に含まれると判定）。具体的には、レコードの `mesh_terms` のいずれかにカテゴリ名が部分文字列として含まれているかで判定する。

厳密な MeSH ツリー階層の走査は本格実装で対応。

### Audit Log Contents

```json
{
  "config": { "...pipeline_config..." },
  "timestamp": "2026-03-13T12:00:00Z",
  "per_year": {
    "2021": {
      "population": 150000,
      "selected": 20000,
      "per_category": {
        "Neoplasms": { "available": 8000, "selected": 500 },
        "...": "..."
      }
    }
  },
  "total_selected": 100000,
  "shortfalls": [
    { "year": 2023, "category": "Urogenital Diseases", "requested": 500, "available": 320 }
  ],
  "sampled_pmids": ["12345678", "12345679", "..."]
}
```

Note: `sampled_pmids` は ADR 0002 の要件。全 PMID をリストとして含める。

## Configuration (config.yaml)

```yaml
years: [2021, 2022, 2023, 2024, 2025]
language: "eng"
require_abstract: true
sampling:
  n_max: 100000
  seed: 42
  allocation: "equal_per_year"  # Only equal_per_year is implemented. Other strategies (e.g., proportional) are deferred.
  min_coverage:
    enabled: true
    per_category_per_year: 500
    mesh_categories:
      - "Neoplasms"
      - "Cardiovascular Diseases"
      - "Infectious Diseases"
      - "Nervous System Diseases"
      - "Respiratory Tract Diseases"
      - "Digestive System Diseases"
      - "Urogenital Diseases"
      - "Musculoskeletal Diseases"
      - "Nutritional and Metabolic Diseases"
      - "Immune System Diseases"

paths:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
```

## Future: FTP Baseline Migration Path

FTP 版に移行する際の追加スクリプト:

1. `download_ftp.py` — NLM FTP サーバーからベースライン `.xml.gz` をダウンロード
2. `convert_xml_to_jsonl.py` — PubMed XML を中間 JSONL スキーマに変換して `data/raw/` に出力

`sample.py` と `config.yaml` は変更不要。

## Dependencies

- Python 3.10+
- `datasets` (HuggingFace)
- `pyyaml`
- Standard library: `json`, `random`, `collections`, `datetime`, `pathlib`, `xml.etree.ElementTree`

## Success Criteria

- `download_hf.py` が HuggingFace から PubMed データを取得し `data/raw/pubmed_raw.jsonl` を生成できる
- `sample.py` が config に従ってフィルタ・サンプリングし `data/processed/sampled.jsonl` + `audit_log.json` を出力できる
- 監査ログで年別・カテゴリ別のカバレッジおよび sampled PMIDs を確認できる
- 固定シードで再実行時に同一結果が得られる
- `--limit N` オプションで小規模テストが可能
