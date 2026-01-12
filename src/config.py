"""
設定ファイル: gpt-oss-20b領域特化LLM用の設定
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """モデル設定"""
    model_name: str = "openai/gpt-oss-20b"
    max_seq_length: int = 2048
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_unsloth: bool = True  # Unslothが対応していない場合はFalseに変更
    
@dataclass
class LoRAConfig:
    """LoRA設定"""
    r: int = 32  # ランク
    lora_alpha: int = 64  # スケーリング係数
    target_modules: Optional[list] = None  # Noneの場合は自動検出
    use_gradient_checkpointing: bool = True
    random_state: int = 42
    
    def __post_init__(self):
        if self.target_modules is None:
            # gpt-oss-20bのMoEアーキテクチャに適したモジュール
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

@dataclass
class GRPOConfig:
    """GRPO学習設定"""
    output_dir: str = "outputs/gpt_oss_20b_domain_specialized"
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 1
    per_device_train_batch_size: int = 1  # 20Bモデルなので小さめに
    gradient_accumulation_steps: int = 8
    num_generations: int = 8  # VRAM消費抑制のため低く設定。理想16以上
    max_prompt_length: int = 512
    max_completion_length: int = 1024
    max_steps: int = 100
    save_steps: int = 50
    report_to: str = "none"
    use_vllm: bool = False
    
@dataclass
class DatasetConfig:
    """データセット設定"""
    data_dir: str = "data"
    output_dataset_path: str = "data/custom_dataset"
    chunk_size: int = 512  # チャンクサイズ（トークン数）
    chunk_overlap: int = 50  # チャンクのオーバーラップ
    min_chunk_size: int = 100  # 最小チャンクサイズ
    
@dataclass
class GenerationConfig:
    """生成設定"""
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 1024

# デフォルト設定インスタンス
model_config = ModelConfig()
lora_config = LoRAConfig()
grpo_config = GRPOConfig()
dataset_config = DatasetConfig()
generation_config = GenerationConfig()
