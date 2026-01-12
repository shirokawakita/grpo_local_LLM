"""
セットアップテストスクリプト
環境が正しく設定されているか確認する
"""

import sys

def test_imports():
    """必要なライブラリがインポートできるか確認"""
    print("=== ライブラリのインポートテスト ===")
    
    required_modules = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("trl", "TRL"),
        ("peft", "PEFT"),
    ]
    
    optional_modules = [
        ("unsloth", "Unsloth"),
        ("pdfplumber", "pdfplumber"),
        ("PyPDF2", "PyPDF2"),
    ]
    
    print("\n必須ライブラリ:")
    all_ok = True
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} - インストールが必要です")
            all_ok = False
    
    print("\nオプションライブラリ:")
    for module_name, display_name in optional_modules:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  - {display_name} - オプション（推奨）")
    
    return all_ok


def test_gpu():
    """GPUの利用可能性を確認"""
    print("\n=== GPU確認 ===")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA利用可能")
            print(f"    デバイス数: {torch.cuda.device_count()}")
            print(f"    デバイス名: {torch.cuda.get_device_name(0)}")
            print(f"    メモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("  - CUDA利用不可（CPUモードで実行されます）")
    except ImportError:
        print("  ✗ PyTorchがインストールされていません")


def test_model_access():
    """モデルへのアクセスを確認"""
    print("\n=== モデルアクセステスト ===")
    
    try:
        from transformers import AutoTokenizer
        
        model_name = "openai/gpt-oss-20b"
        print(f"  モデル: {model_name}")
        print("  トークナイザーの読み込みをテスト中...")
        
        # トークナイザーのみ読み込み（軽量）
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        print("  ✓ トークナイザーの読み込み成功")
        print(f"    語彙サイズ: {len(tokenizer)}")
        
        # 簡単なテスト
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"  テストエンコード/デコード: {test_text} -> {decoded}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        print("    モデルへのアクセスに問題があります")
        return False


def test_local_modules():
    """ローカルモジュールのインポートを確認"""
    print("\n=== ローカルモジュールテスト ===")
    
    modules = [
        "src.config",
        "src.dataset_creator",
        "src.reward_functions",
        "src.domain_specialized_llm",
    ]
    
    all_ok = True
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            all_ok = False
    
    return all_ok


def main():
    """メインテスト"""
    print("=" * 50)
    print("gpt-oss-20b領域特化LLM セットアップテスト")
    print("=" * 50)
    
    results = []
    
    # ライブラリテスト
    results.append(("ライブラリ", test_imports()))
    
    # GPUテスト
    test_gpu()
    
    # ローカルモジュールテスト
    results.append(("ローカルモジュール", test_local_modules()))
    
    # モデルアクセステスト（オプション、時間がかかる場合がある）
    if "--skip-model-test" not in sys.argv:
        results.append(("モデルアクセス", test_model_access()))
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("テスト結果サマリー")
    print("=" * 50)
    
    for name, result in results:
        status = "✓ 成功" if result else "✗ 失敗"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ すべてのテストが成功しました！")
        print("  python domain_specialized_llm.py --help で使用方法を確認してください")
    else:
        print("\n✗ 一部のテストが失敗しました")
        print("  requirements.txtのライブラリをインストールしてください:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
