# gpt-oss-20b 領域特化LLM

`openai/gpt-oss-20b`モデルを使用して、GRPO（Group Relative Policy Optimization）による領域特化LLMを作成するための実装です。

## 概要

このプロジェクトは、汎用LLMである`gpt-oss-20b`を特定のドメインに特化させるための実装を提供します。カスタムデータセットから学習し、強化学習（GRPO）を用いてドメイン固有の知識を獲得します。

## 主な特徴

- **gpt-oss-20b対応**: OpenAIのgpt-oss-20bモデルを使用
- **カスタムデータセット作成**: PDFやテキストからデータセットを自動生成
- **GRPO学習**: 強化学習による領域特化
- **Harmony format対応**: gpt-oss-20bのHarmony formatを自動処理
- **柔軟な報酬関数**: ドメイン固有の報酬関数を定義可能
- **Unsloth/Transformers対応**: メモリ効率的な学習をサポート

## クイックスタート

詳細な手順は [QUICKSTART.md](QUICKSTART.md) を参照してください。

### 1. セットアップ

```bash
# 依存関係のインストール
pip install -r requirements.txt

# セットアップテスト
python src/test_setup.py
```

### 2. データセットの作成

```bash
# PDFからデータセットを作成
python -c "
from src.dataset_creator import DatasetCreator
creator = DatasetCreator()
dataset = creator.create_from_pdfs('data/pdfs', 'data/custom_dataset')
print(f'データセットサイズ: {len(dataset)}')
"
```

### 3. モデルの学習

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
    --domain_keywords 医療 診断 治療 患者
```

### 4. モデルの評価と生成

```bash
# 評価
python src/domain_specialized_llm.py --mode evaluate \
    --dataset_path data/custom_dataset

# テキスト生成
python src/domain_specialized_llm.py --mode generate \
    --prompt "領域特化LLMについて説明してください。"
```

## プロジェクト構成

```
domain_specified_LLM/
├── README.md                    # このファイル
├── QUICKSTART.md                # クイックスタートガイド
├── README_implementation.md     # 詳細な実装ガイド
├── LECTURE_MATERIALS.md         # 講義資料の紹介（参考）
│
├── src/                         # ソースコード
│   ├── __init__.py
│   ├── config.py                # 設定ファイル
│   ├── domain_specialized_llm.py  # メイン実装
│   ├── dataset_creator.py       # データセット作成
│   ├── reward_functions.py      # 報酬関数
│   ├── example_usage.py         # 使用例
│   ├── extract_pdf_text.py      # PDFテキスト抽出
│   └── test_setup.py            # セットアップテスト
│
├── data/                        # データディレクトリ
│   ├── pdfs/                    # PDFファイル格納
│   └── custom_dataset/          # 作成されたデータセット
│
├── outputs/                     # 学習結果の出力
│   └── gpt_oss_20b_domain_specialized/
│
└── requirements.txt            # 依存関係
```

## 主な機能

### データセット作成

- **PDFからデータセット作成**: PDFファイルを自動的にチャンク化してデータセットを生成
- **テキストからデータセット作成**: テキストリストから直接データセットを作成
- **カスタマイズ可能なチャンクサイズ**: ドメインに応じて最適なチャンクサイズを設定

### 学習機能

- **GRPO学習**: Group Relative Policy Optimizationによる強化学習
- **LoRA対応**: メモリ効率的なパラメータ効率的な微調整
- **カスタム報酬関数**: ドメイン固有の評価基準を定義可能
- **ドメインキーワード指定**: 特定のキーワードを含む回答を促進

### 評価・生成機能

- **モデル評価**: データセットに対する性能評価
- **テキスト生成**: カスタマイズされたモデルによるテキスト生成
- **Harmony format自動処理**: gpt-oss-20bの特殊なフォーマットを自動処理

## 設定のカスタマイズ

`src/config.py`を編集して設定を変更できます：

```python
# モデル設定
model_config.model_name = "openai/gpt-oss-20b"
model_config.max_seq_length = 2048
model_config.load_in_4bit = False  # メモリ不足時はTrueに

# LoRA設定
lora_config.r = 32
lora_config.lora_alpha = 64

# GRPO設定
grpo_config.learning_rate = 5e-6
grpo_config.max_steps = 100
grpo_config.num_generations = 8
```

詳細は [README_implementation.md](README_implementation.md) を参照してください。

## 使用例

詳細な使用例は `src/example_usage.py` を参照してください。

```bash
# 使用例の実行
python src/example_usage.py create_pdf    # PDFからデータセット作成
python src/example_usage.py create_text   # テキストからデータセット作成
python src/example_usage.py train         # 学習
python src/example_usage.py evaluate      # 評価
python src/example_usage.py generate      # 生成
python src/example_usage.py custom_reward # カスタム報酬関数
```

## 注意事項

### gpt-oss-20bの特徴

1. **Harmony format必須**: gpt-oss-20bはHarmony formatで動作します。コード内で自動的に適用されます。
2. **MoEアーキテクチャ**: Mixture of Expertsアーキテクチャを使用しています。
3. **メモリ要件**: MXFP4量子化済みで、16GBメモリで動作可能です。

### システム要件

- Python 3.8以上
- CUDA対応GPU（推奨、16GB以上推奨）
- 十分なディスク容量（モデルダウンロード用）

### トラブルシューティング

メモリ不足が発生した場合：

- `src/config.py`で`load_in_4bit=True`を設定
- `per_device_train_batch_size`を1に設定
- `num_generations`を4に減らす
- `gradient_accumulation_steps`を増やす

詳細は [QUICKSTART.md](QUICKSTART.md) のトラブルシューティングセクションを参照してください。

## 参考資料

- [gpt-oss-20b Model Card](https://huggingface.co/openai/gpt-oss-20b)
- [GRPO解説記事](https://lancelqf.github.io/note/llm_post_training/)
- [Unsloth Documentation](https://unsloth.ai/)
- [TRL Documentation](https://huggingface.co/docs/trl/)

## ライセンス

このプロジェクトのライセンスについては、各ファイルのヘッダーを参照してください。

---

**最終更新**: 2026年
