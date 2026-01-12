# クイックスタートガイド

## 1. 環境セットアップ

```bash
# 依存関係のインストール
pip install -r requirements.txt

# セットアップテスト
python src/test_setup.py
```

## 2. データセットの作成

### PDFからデータセットを作成

1. PDFファイルを `data/pdfs/` ディレクトリに配置
2. データセットを作成:

```bash
python -c "
from src.dataset_creator import DatasetCreator
creator = DatasetCreator()
dataset = creator.create_from_pdfs('data/pdfs', 'data/custom_dataset')
print(f'データセットサイズ: {len(dataset)}')
"
```

### テキストからデータセットを作成

```python
from src.dataset_creator import DatasetCreator

creator = DatasetCreator()
texts = ["文書1の内容...", "文書2の内容..."]
sources = ["doc1.txt", "doc2.txt"]
dataset = creator.create_from_texts(texts, sources, "data/custom_dataset")
```

## 3. モデルの学習

```bash
# 基本的な学習
python src/domain_specialized_llm.py --mode train \
    --dataset_path data/custom_dataset

# PDFから直接学習
python src/domain_specialized_llm.py --mode train \
    --pdf_dir data/pdfs

# ドメインキーワードを指定
python src/domain_specialized_llm.py --mode train \
    --dataset_path data/custom_dataset \
    --domain_keywords 医療 診断 治療 患者　♯医療系領域特化型の場合
```

## 4. モデルの評価

```bash
python src/domain_specialized_llm.py --mode evaluate \
    --dataset_path data/custom_dataset
```

## 5. テキスト生成

```bash
python src/domain_specialized_llm.py --mode generate \
    --prompt "領域特化LLMについて説明してください。"
```

## 設定の調整

メモリが不足する場合、`src/config.py`で以下を調整:

```python
# バッチサイズを小さく
grpo_config.per_device_train_batch_size = 1

# 勾配累積を増やす
grpo_config.gradient_accumulation_steps = 16

# 生成数を減らす
grpo_config.num_generations = 4

# 4bit量子化を使用
model_config.load_in_4bit = True
```

## トラブルシューティング

### メモリ不足

- `src/config.py`で`load_in_4bit=True`を設定
- `per_device_train_batch_size`を1に設定
- `num_generations`を4に減らす

### モデルの読み込みエラー

- Unslothが対応していない場合、自動的にTransformersにフォールバック
- `src/config.py`で`use_unsloth=False`を設定して明示的にTransformersを使用

### データセットのエラー

- PDFファイルが読み込めない場合、`pdfplumber`をインストール: `pip install pdfplumber`
- データセットが空の場合、`chunk_size`や`min_chunk_size`を調整
