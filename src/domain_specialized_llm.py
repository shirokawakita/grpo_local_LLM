"""
gpt-oss-20b領域特化LLM実装
GRPOを使用して領域特化LLMを作成する
"""

import os
import sys
import torch
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    TrainingArguments
)
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# ローカルモジュール
from src.config import (
    model_config,
    lora_config,
    grpo_config,
    dataset_config,
    generation_config
)
from src.dataset_creator import DatasetCreator
from src.reward_functions import (
    get_default_reward_functions,
    create_domain_reward_functions,
    extract_boxed_answer
)


class HarmonyFormatHandler:
    """Harmony formatの処理クラス"""
    
    @staticmethod
    def apply_harmony_format(messages: List[Dict[str, str]]) -> str:
        """
        Harmony formatを適用
        
        Args:
            messages: [{"role": "user", "content": "..."}]形式のメッセージリスト
            
        Returns:
            Harmony formatに変換されたテキスト
        """
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted += f"<|system|>\n{content}\n<|end|>\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}\n<|end|>\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n<|end|>\n"
        
        return formatted
    
    @staticmethod
    def create_prompt(user_content: str, system_content: Optional[str] = None) -> str:
        """プロンプトを作成"""
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        return HarmonyFormatHandler.apply_harmony_format(messages)
    
    @staticmethod
    def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
        """
        tokenizerのchat templateを使用してHarmony formatを適用
        （gpt-oss-20bのtokenizerがchat templateをサポートしている場合）
        """
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # フォールバック: 手動でHarmony formatを適用
            return HarmonyFormatHandler.apply_harmony_format(messages)


class BoxedStoppingCriteria(StoppingCriteria):
    """\\boxed{*}が出力されたら停止するStoppingCriteria"""
    
    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.pattern = re.compile(r"\\boxed\{.*?\S+.*?\}")
    
    def __call__(self, input_ids, scores, **kwargs):
        generated_tokens = input_ids[0][self.prompt_length:]
        if len(generated_tokens) < 5:
            return False
        
        last_token_text = self.tokenizer.decode(
            generated_tokens[-1:],
            skip_special_tokens=True
        )
        if "}" not in last_token_text:
            return False
        
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return bool(self.pattern.search(text))


class DomainSpecializedLLM:
    """領域特化LLMクラス"""
    
    def __init__(
        self,
        model_name: str = None,
        use_unsloth: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model_name: モデル名（Noneの場合はconfigから取得）
            use_unsloth: Unslothを使用するか
            device: デバイス
        """
        self.model_name = model_name or model_config.model_name
        self.use_unsloth = use_unsloth and model_config.use_unsloth
        self.device = device
        self.model = None
        self.tokenizer = None
        self.harmony_handler = HarmonyFormatHandler()
        
        print(f"モデル: {self.model_name}")
        print(f"Unsloth使用: {self.use_unsloth}")
        print(f"デバイス: {self.device}")
    
    def load_model(self, apply_lora: bool = True):
        """モデルを読み込む"""
        print("モデルを読み込んでいます...")
        
        if self.use_unsloth:
            try:
                from unsloth import FastLanguageModel
                from unsloth import PatchFastRL
                
                PatchFastRL("GRPO", FastLanguageModel)
                
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_name,
                    max_seq_length=model_config.max_seq_length,
                    dtype=None,
                    load_in_4bit=model_config.load_in_4bit,
                    load_in_8bit=model_config.load_in_8bit,
                )
                
                if apply_lora:
                    self.model = FastLanguageModel.get_peft_model(
                        self.model,
                        r=lora_config.r,
                        lora_alpha=lora_config.lora_alpha,
                        target_modules=lora_config.target_modules,
                        use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
                        random_state=lora_config.random_state,
                    )
                
                print("Unslothでモデルを読み込みました")
                return
                
            except Exception as e:
                print(f"警告: Unslothでの読み込みに失敗: {e}")
                print("Transformersベースの実装にフォールバックします")
                self.use_unsloth = False
        
        # Transformersベースの実装
        print("Transformersでモデルを読み込んでいます...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # pad_tokenが設定されていない場合
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # モデルの読み込み
        load_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        if model_config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif model_config.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )
        
        # LoRAを適用
        if apply_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=0.1,
            )
            self.model = get_peft_model(self.model, peft_config)
        
        print("Transformersでモデルを読み込みました")
    
    def prepare_dataset(
        self,
        dataset_path: Optional[str] = None,
        pdf_dir: Optional[str] = None,
        texts: Optional[List[str]] = None,
        sources: Optional[List[str]] = None
    ) -> Dataset:
        """
        データセットを準備
        
        Args:
            dataset_path: 既存のデータセットパス
            pdf_dir: PDFディレクトリ
            texts: テキストリスト
            sources: ソースリスト
            
        Returns:
            準備されたデータセット
        """
        creator = DatasetCreator(
            chunk_size=dataset_config.chunk_size,
            chunk_overlap=dataset_config.chunk_overlap,
            min_chunk_size=dataset_config.min_chunk_size
        )
        
        if dataset_path and os.path.exists(dataset_path):
            print(f"既存のデータセットを読み込みます: {dataset_path}")
            return creator.load_dataset(dataset_path)
        
        elif pdf_dir:
            print(f"PDFディレクトリからデータセットを作成します: {pdf_dir}")
            return creator.create_from_pdfs(
                pdf_dir,
                dataset_config.output_dataset_path
            )
        
        elif texts and sources:
            print("テキストリストからデータセットを作成します")
            return creator.create_from_texts(
                texts,
                sources,
                dataset_config.output_dataset_path
            )
        
        else:
            raise ValueError(
                "データセットのソースを指定してください: "
                "dataset_path, pdf_dir, または texts+sources"
            )
    
    def train_with_grpo(
        self,
        train_dataset: Dataset,
        reward_functions: Optional[List] = None,
        domain_keywords: Optional[List[str]] = None
    ):
        """
        GRPOで学習
        
        Args:
            train_dataset: 学習データセット
            reward_functions: 報酬関数のリスト（Noneの場合はデフォルト）
            domain_keywords: ドメイン固有のキーワードリスト
        """
        if self.model is None:
            raise ValueError("モデルを先に読み込んでください")
        
        # 報酬関数の準備
        if reward_functions is None:
            reward_functions = create_domain_reward_functions(
                domain_keywords=domain_keywords,
                has_ground_truth=False  # カスタムデータセットでは正解がない場合が多い
            )
        
        print(f"使用する報酬関数数: {len(reward_functions)}")
        
        # GRPO設定
        training_args = GRPOConfig(
            output_dir=grpo_config.output_dir,
            learning_rate=grpo_config.learning_rate,
            adam_beta1=grpo_config.adam_beta1,
            adam_beta2=grpo_config.adam_beta2,
            weight_decay=grpo_config.weight_decay,
            warmup_ratio=grpo_config.warmup_ratio,
            lr_scheduler_type=grpo_config.lr_scheduler_type,
            logging_steps=grpo_config.logging_steps,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            per_device_train_batch_size=grpo_config.per_device_train_batch_size,
            gradient_accumulation_steps=grpo_config.gradient_accumulation_steps,
            num_generations=grpo_config.num_generations,
            max_prompt_length=grpo_config.max_prompt_length,
            max_completion_length=grpo_config.max_completion_length,
            max_steps=grpo_config.max_steps,
            save_steps=grpo_config.save_steps,
            report_to=grpo_config.report_to,
            use_vllm=grpo_config.use_vllm,
            generation_kwargs={
                "do_sample": generation_config.do_sample,
                "temperature": generation_config.temperature,
                "top_p": generation_config.top_p,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
            },
        )
        
        # GRPO Trainer
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_functions,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        print("GRPO学習を開始します...")
        trainer.train()
        
        print(f"学習完了。モデルは {grpo_config.output_dir} に保存されました")
        
        return trainer
    
    def evaluate(
        self,
        test_dataset: Dataset,
        num_samples: Optional[int] = None,
        use_stopping_criteria: bool = True
    ) -> Dict[str, Any]:
        """
        モデルを評価
        
        Args:
            test_dataset: 評価データセット
            num_samples: 評価サンプル数（Noneの場合は全て）
            use_stopping_criteria: StoppingCriteriaを使用するか
            
        Returns:
            評価結果の辞書
        """
        if self.model is None:
            raise ValueError("モデルを先に読み込んでください")
        
        self.model.eval()
        
        if num_samples:
            test_dataset = test_dataset.shuffle(seed=42).select(range(num_samples))
        
        correct = 0
        total = len(test_dataset)
        
        results = []
        
        print(f"{total}サンプルを評価します...")
        
        with torch.no_grad():
            for i, example in enumerate(tqdm(test_dataset, total=total)):
                prompt = example.get("prompt", "")
                ground_truth = example.get("answer", None)
                
                # Harmony formatを適用
                if not prompt.startswith("<|"):
                    # tokenizerのchat templateを使用（利用可能な場合）
                    messages = [{"role": "user", "content": prompt}]
                    prompt = self.harmony_handler.apply_chat_template(
                        self.tokenizer,
                        messages
                    )
                
                # トークナイズ
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=grpo_config.max_prompt_length
                ).to(self.device)
                
                prompt_length = inputs.input_ids.shape[1]
                
                # StoppingCriteria
                stopping_criteria = None
                if use_stopping_criteria:
                    stopping_criteria = StoppingCriteriaList([
                        BoxedStoppingCriteria(self.tokenizer, prompt_length)
                    ])
                
                # 生成
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_config.max_new_tokens,
                    temperature=generation_config.temperature,
                    top_p=generation_config.top_p,
                    do_sample=generation_config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                )
                
                # デコード
                generated_text = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                # 回答を抽出
                predicted = extract_boxed_answer(generated_text)
                
                # 正解判定
                is_correct = False
                if ground_truth is not None and predicted is not None:
                    is_correct = (predicted == ground_truth)
                    if is_correct:
                        correct += 1
                
                results.append({
                    "prompt": prompt[:100] + "...",
                    "generated": generated_text[len(prompt):][:200] + "...",
                    "predicted": predicted,
                    "ground_truth": ground_truth,
                    "correct": is_correct
                })
        
        accuracy = correct / total if total > 0 else 0.0
        
        print(f"\n--- 評価結果 ---")
        print(f"正解数: {correct}/{total}")
        print(f"正解率: {accuracy:.4f}")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = None
    ) -> str:
        """
        テキストを生成
        
        Args:
            prompt: プロンプト
            max_new_tokens: 最大生成トークン数
            temperature: 温度パラメータ
            top_p: top_pパラメータ
            
        Returns:
            生成されたテキスト
        """
        if self.model is None:
            raise ValueError("モデルを先に読み込んでください")
        
        self.model.eval()
        
        # Harmony formatを適用
        if not prompt.startswith("<|"):
            # tokenizerのchat templateを使用（利用可能な場合）
            messages = [{"role": "user", "content": prompt}]
            prompt = self.harmony_handler.apply_chat_template(
                self.tokenizer,
                messages
            )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=grpo_config.max_prompt_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or generation_config.max_new_tokens,
                temperature=temperature or generation_config.temperature,
                top_p=top_p or generation_config.top_p,
                do_sample=generation_config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # プロンプト部分を除去
        return generated_text[len(prompt):].strip()
    
    def plot_training_history(self, trainer):
        """学習履歴をプロット"""
        if not hasattr(trainer, 'state') or not hasattr(trainer.state, 'log_history'):
            print("学習履歴がありません")
            return
        
        history = trainer.state.log_history
        
        steps = np.array([x['step'] for x in history if 'reward' in x])
        rewards = np.array([x['reward'] for x in history if 'reward' in x])
        
        if len(steps) == 0:
            print("報酬データがありません")
            return
        
        window = 5
        ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ma_steps = steps[window - 1:]
        
        plt.figure(figsize=(10, 4))
        plt.plot(steps, rewards, marker='.', alpha=0.5, label='Reward')
        plt.plot(ma_steps, ma, linewidth=2, label='5-step Moving Average')
        plt.title('Training Reward Over Steps')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        output_path = os.path.join(grpo_config.output_dir, "training_history.png")
        plt.savefig(output_path)
        print(f"学習履歴を保存しました: {output_path}")
        plt.close()


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="gpt-oss-20b領域特化LLM")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "generate"],
        default="train",
        help="実行モード"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="データセットパス（既存のデータセット）"
    )
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default=None,
        help="PDFディレクトリ（新規データセット作成）"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="学習済みモデルのパス（評価・生成時）"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="生成用のプロンプト（generateモード時）"
    )
    parser.add_argument(
        "--domain_keywords",
        type=str,
        nargs="+",
        default=None,
        help="ドメイン固有のキーワードリスト"
    )
    
    args = parser.parse_args()
    
    # モデルの初期化
    llm = DomainSpecializedLLM()
    llm.load_model(apply_lora=True)
    
    if args.mode == "train":
        # データセットの準備
        dataset = llm.prepare_dataset(
            dataset_path=args.dataset_path,
            pdf_dir=args.pdf_dir
        )
        
        print(f"データセットサイズ: {len(dataset)}")
        
        # 学習
        trainer = llm.train_with_grpo(
            train_dataset=dataset,
            domain_keywords=args.domain_keywords
        )
        
        # 学習履歴をプロット
        llm.plot_training_history(trainer)
    
    elif args.mode == "evaluate":
        # データセットの準備
        dataset = llm.prepare_dataset(dataset_path=args.dataset_path)
        
        # 評価
        results = llm.evaluate(dataset, num_samples=100)
        
        # 結果を保存
        import json
        output_path = os.path.join(grpo_config.output_dir, "evaluation_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"評価結果を保存しました: {output_path}")
    
    elif args.mode == "generate":
        if args.prompt is None:
            args.prompt = input("プロンプトを入力してください: ")
        
        generated = llm.generate(args.prompt)
        print("\n--- 生成結果 ---")
        print(generated)


if __name__ == "__main__":
    main()
