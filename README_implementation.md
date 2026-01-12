# gpt-oss-20b領域特化LLM実装ガイド

## 概要

この実装は、`openai/gpt-oss-20b`モデルを使用して、GRPO（Group Relative Policy Optimization）による領域特化LLMを作成するためのコードです。

## 特徴

- **gpt-oss-20b対応**: OpenAIのgpt-oss-20bモデルを使用
- **カスタムデータセット作成**: PDFやテキストからデータセットを自動生成
- **GRPO学習**: 強化学習による領域特化
- **Harmony format対応**: gpt-oss-20bのHarmony formatを自動処理
- **柔軟な報酬関数**: ドメイン固有の報酬関数を定義可能

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. ディレクトリ構造の作成

```bash
mkdir -p data/pdfs
mkdir -p outputs
```

## 使用方法

### 1. データセットの作成

#### PDFからデータセットを作成

```python
from src.dataset_creator import DatasetCreator

creator = DatasetCreator(
    chunk_size=512,
    chunk_overlap=50,
    min_chunk_size=100
)

# PDFディレクトリからデータセットを作成
dataset = creator.create_from_pdfs(
    pdf_dir="data/pdfs",
    output_path="data/custom_dataset"
)
```

#### テキストリストからデータセットを作成

```python
texts = ["文書1の内容...", "文書2の内容..."]
sources = ["doc1.txt", "doc2.txt"]

dataset = creator.create_from_texts(
    texts=texts,
    sources=sources,
    output_path="data/custom_dataset"
)
```

### 2. モデルの学習

#### コマンドラインから実行

```bash
# PDFからデータセットを作成して学習
python src/domain_specialized_llm.py --mode train --pdf_dir data/pdfs

# 既存のデータセットで学習
python src/domain_specialized_llm.py --mode train --dataset_path data/custom_dataset

# ドメイン固有のキーワードを指定
python src/domain_specialized_llm.py --mode train \
    --dataset_path data/custom_dataset \
    --domain_keywords 医療 診断 治療 患者
```

#### Pythonコードから実行

```python
from src.domain_specialized_llm import DomainSpecializedLLM

# モデルの初期化
llm = DomainSpecializedLLM()
llm.load_model(apply_lora=True)

# データセットの準備
dataset = llm.prepare_dataset(
    dataset_path="data/custom_dataset"
)

# ドメイン固有のキーワード
domain_keywords = ["医療", "診断", "治療", "患者"] #医療系領域特化型の場合

# 学習
trainer = llm.train_with_grpo(
    train_dataset=dataset,
    domain_keywords=domain_keywords
)
```

### 3. モデルの評価

```bash
python src/domain_specialized_llm.py --mode evaluate \
    --dataset_path data/custom_dataset
```

### 4. テキスト生成

```bash
python src/domain_specialized_llm.py --mode generate \
    --prompt "領域特化LLMについて説明してください。"
```

## カスタム報酬関数

### デフォルトの報酬関数

- **FormatReward**: 指定された形式で出力できているか
- **LengthReward**: 適切な長さの回答か
- **QualityReward**: テキストの品質（理由、例、まとめなど）
- **DomainSpecificReward**: ドメイン固有のキーワードを含んでいるか

### カスタム報酬関数の作成

```python
from src.reward_functions import DomainRewardFunction

class MyCustomReward(DomainRewardFunction):
    def __call__(self, prompts, completions, **kwargs):
        rewards = []
        for completion in completions:
            # カスタムロジックで報酬を計算
            reward = 0.0
            if "重要なポイント" in completion:
                reward += 0.5
            rewards.append(reward)
        return rewards

# 使用
reward_functions = [
    FormatReward(),
    LengthReward(),
    MyCustomReward()
]

trainer = llm.train_with_grpo(
    train_dataset=dataset,
    reward_functions=reward_functions
)
```

## 設定のカスタマイズ

`src/config.py`を編集して設定を変更できます：

```python
# モデル設定
model_config.model_name = "openai/gpt-oss-20b"
model_config.max_seq_length = 2048

# LoRA設定
lora_config.r = 32
lora_config.lora_alpha = 64

# GRPO設定
grpo_config.learning_rate = 5e-6
grpo_config.max_steps = 100
grpo_config.num_generations = 8
```

## 注意事項

### gpt-oss-20bの特徴

1. **Harmony format必須**: gpt-oss-20bはHarmony formatで動作します。コード内で自動的に適用されます。
2. **MoEアーキテクチャ**: Mixture of Expertsアーキテクチャを使用しています。
3. **メモリ要件**: MXFP4量子化済みで、16GBメモリで動作可能です。

### Unslothの互換性

gpt-oss-20bはUnslothで直接サポートされていない可能性があります。その場合、自動的にTransformersベースの実装にフォールバックします。

### データセットの形式

GRPO用のデータセットは以下の形式が必要です：

```python
{
    "prompt": "プロンプトテキスト",
    "answer": None,  # GRPOでは生成時に評価されるためNoneでも可
    "source": "ソースファイル名",
    "chunk_id": 0,
    "metadata": {}
}
```

## トラブルシューティング

### メモリ不足エラー

- `per_device_train_batch_size`を小さくする（例: 1）
- `gradient_accumulation_steps`を増やす（例: 8）
- `num_generations`を減らす（例: 4）
- `load_in_4bit=True`または`load_in_8bit=True`を設定

### モデルの読み込みエラー

- Unslothが対応していない場合、自動的にTransformersにフォールバックします
- `src/config.py`で`use_unsloth=False`を設定して明示的にTransformersを使用

### データセットのエラー

- PDFファイルが正しく読み込めない場合、`pdfplumber`または`PyPDF2`をインストール
- データセットが空の場合は、`chunk_size`や`min_chunk_size`を調整

## ファイル構成

```
domain_specified_LLM/
├── src/                          # ソースコード
│   ├── __init__.py
│   ├── domain_specialized_llm.py  # メイン実装
│   ├── dataset_creator.py        # データセット作成
│   ├── reward_functions.py       # 報酬関数
│   ├── config.py                 # 設定
│   ├── example_usage.py          # 使用例
│   ├── extract_pdf_text.py       # PDFテキスト抽出
│   └── test_setup.py             # セットアップテスト
├── requirements.txt              # 依存関係
├── README_implementation.md     # このファイル
├── outputs/                      # 学習結果
└── data/                         # データセット
    ├── pdfs/                     # PDFファイル
    └── custom_dataset/           # 作成されたデータセット
```

## 参考資料

- [gpt-oss-20b Model Card](https://huggingface.co/openai/gpt-oss-20b)
- [GRPO解説記事](https://lancelqf.github.io/note/llm_post_training/)
- [Unsloth Documentation](https://unsloth.ai/)
