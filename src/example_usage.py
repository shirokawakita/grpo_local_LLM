"""
使用例: gpt-oss-20b領域特化LLMの使用方法
"""

from src.domain_specialized_llm import DomainSpecializedLLM
from src.dataset_creator import DatasetCreator
from src.reward_functions import create_domain_reward_functions
from src.config import dataset_config


def example_create_dataset_from_pdfs():
    """PDFからデータセットを作成する例"""
    print("=== PDFからデータセット作成 ===")
    
    creator = DatasetCreator(
        chunk_size=512,
        chunk_overlap=50,
        min_chunk_size=100
    )
    
    # PDFディレクトリからデータセットを作成
    pdf_dir = "data/pdfs"  # PDFファイルが格納されているディレクトリ
    output_path = "data/custom_dataset"
    
    dataset = creator.create_from_pdfs(
        pdf_dir=pdf_dir,
        output_path=output_path
    )
    
    print(f"データセットサイズ: {len(dataset)}")
    print(f"最初のサンプル: {dataset[0]}")


def example_create_dataset_from_texts():
    """テキストリストからデータセットを作成する例"""
    print("=== テキストリストからデータセット作成 ===")
    
    creator = DatasetCreator()
    
    # サンプルテキスト
    texts = [
        """
        領域特化LLMは、特定のドメインに特化した大規模言語モデルです。
        汎用LLMと比較して、特定の領域での性能が向上します。
        主な手法として、Fine-tuning、LoRA、GRPOなどがあります。
        """,
        """
        GRPO（Group Relative Policy Optimization）は、
        強化学習を用いてLLMの性能を向上させる手法です。
        PPOと比較して、メモリ効率が良いことが特徴です。
        """
    ]
    
    sources = ["doc1.txt", "doc2.txt"]
    
    dataset = creator.create_from_texts(
        texts=texts,
        sources=sources,
        output_path="data/text_dataset"
    )
    
    print(f"データセットサイズ: {len(dataset)}")


def example_train():
    """学習の例"""
    print("=== GRPO学習 ===")
    
    # モデルの初期化
    llm = DomainSpecializedLLM()
    llm.load_model(apply_lora=True)
    
    # データセットの準備（既存のデータセットを使用）
    dataset = llm.prepare_dataset(
        dataset_path="data/custom_dataset"
    )
    
    # ドメイン固有のキーワード（例：医療ドメイン）
    domain_keywords = [
        "診断", "治療", "症状", "患者", "疾患",
        "薬剤", "臨床", "医学", "医療"
    ]
    
    # 学習
    trainer = llm.train_with_grpo(
        train_dataset=dataset,
        domain_keywords=domain_keywords
    )
    
    # 学習履歴をプロット
    llm.plot_training_history(trainer)


def example_evaluate():
    """評価の例"""
    print("=== モデル評価 ===")
    
    # モデルの初期化
    llm = DomainSpecializedLLM()
    llm.load_model(apply_lora=False)  # 評価時はLoRA不要
    
    # データセットの準備
    dataset = llm.prepare_dataset(
        dataset_path="data/custom_dataset"
    )
    
    # 評価
    results = llm.evaluate(
        test_dataset=dataset,
        num_samples=50  # 50サンプルで評価
    )
    
    print(f"正解率: {results['accuracy']:.4f}")


def example_generate():
    """テキスト生成の例"""
    print("=== テキスト生成 ===")
    
    # モデルの初期化
    llm = DomainSpecializedLLM()
    llm.load_model(apply_lora=False)
    
    # プロンプト
    prompt = """
    以下の文書を読んで、重要なポイントを要約してください。
    
    文書:
    領域特化LLMは、特定のドメインに特化した大規模言語モデルです。
    汎用LLMと比較して、特定の領域での性能が向上します。
    主な手法として、Fine-tuning、LoRA、GRPOなどがあります。
    """
    
    # 生成
    generated = llm.generate(prompt)
    
    print("--- 生成結果 ---")
    print(generated)


def example_custom_reward():
    """カスタム報酬関数の例"""
    print("=== カスタム報酬関数 ===")
    
    from src.reward_functions import (
        FormatReward,
        LengthReward,
        QualityReward,
        DomainSpecificReward
    )
    
    # カスタム報酬関数を作成
    reward_functions = [
        FormatReward(pattern=r"要約|まとめ", reward=0.2),  # 要約を含む場合
        LengthReward(min_length=100, max_length=1000, optimal_length=500),
        QualityReward(),
        DomainSpecificReward(
            domain_keywords=["領域", "特化", "LLM", "ドメイン"],
            reward_per_keyword=0.1
        )
    ]
    
    # モデルの初期化
    llm = DomainSpecializedLLM()
    llm.load_model(apply_lora=True)
    
    # データセットの準備
    dataset = llm.prepare_dataset(
        dataset_path="data/custom_dataset"
    )
    
    # カスタム報酬関数で学習
    trainer = llm.train_with_grpo(
        train_dataset=dataset,
        reward_functions=reward_functions
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        
        if example_name == "create_pdf":
            example_create_dataset_from_pdfs()
        elif example_name == "create_text":
            example_create_dataset_from_texts()
        elif example_name == "train":
            example_train()
        elif example_name == "evaluate":
            example_evaluate()
        elif example_name == "generate":
            example_generate()
        elif example_name == "custom_reward":
            example_custom_reward()
        else:
            print(f"不明な例: {example_name}")
    else:
        print("使用例:")
        print("  python src/example_usage.py create_pdf    # PDFからデータセット作成")
        print("  python src/example_usage.py create_text   # テキストからデータセット作成")
        print("  python src/example_usage.py train         # 学習")
        print("  python src/example_usage.py evaluate      # 評価")
        print("  python src/example_usage.py generate      # 生成")
        print("  python src/example_usage.py custom_reward # カスタム報酬関数")
