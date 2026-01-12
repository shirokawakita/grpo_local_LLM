"""
報酬関数モジュール
GRPO学習用のカスタム報酬関数を定義
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np


def extract_boxed_answer(text: str) -> Optional[int]:
    """
    \\boxed{123}形式の回答を抽出
    
    Args:
        text: 生成されたテキスト
        
    Returns:
        抽出された数値、見つからない場合はNone
    """
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if match is None:
        return None
    
    raw_value = match.group(1).replace(",", "").strip()
    try:
        return int(raw_value)
    except ValueError:
        return None


def extract_hash_answer(text: str) -> Optional[int]:
    """
    ### 123形式の回答を抽出
    
    Args:
        text: 生成されたテキスト
        
    Returns:
        抽出された数値、見つからない場合はNone
    """
    match = re.search(r"####\s*(-?\d+)", text)
    if match is None:
        return None
    
    try:
        return int(match.group(1))
    except ValueError:
        return None


def extract_last_number(text: str) -> Optional[float]:
    """
    テキストから最後の数値を抽出（フォールバック用）
    
    Args:
        text: 生成されたテキスト
        
    Returns:
        抽出された数値、見つからない場合はNone
    """
    # 数値パターンを検索（整数・小数）
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


class DomainRewardFunction:
    """ドメイン特化用の報酬関数基底クラス"""
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        報酬を計算
        
        Args:
            prompts: プロンプトのリスト
            completions: 生成されたテキストのリスト
            **kwargs: その他の引数（answer, metadata等）
            
        Returns:
            報酬のリスト
        """
        raise NotImplementedError


class FormatReward(DomainRewardFunction):
    """フォーマット報酬: 指定された形式で出力できているか"""
    
    def __init__(self, pattern: str = r"\\boxed\{.+?\}", reward: float = 0.1):
        """
        Args:
            pattern: 検索する正規表現パターン
            reward: フォーマットが正しい場合の報酬
        """
        self.pattern = re.compile(pattern)
        self.reward = reward
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        rewards = []
        for completion in completions:
            if self.pattern.search(completion):
                rewards.append(self.reward)
            else:
                rewards.append(0.0)
        return rewards


class CorrectnessReward(DomainRewardFunction):
    """正解報酬: 正解と一致しているか"""
    
    def __init__(self, extract_func=None):
        """
        Args:
            extract_func: 回答を抽出する関数（デフォルト: extract_boxed_answer）
        """
        self.extract_func = extract_func or extract_boxed_answer
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        answer: Optional[List[Any]] = None,
        **kwargs
    ) -> List[float]:
        """
        Args:
            answer: 正解のリスト（Noneの場合は正解報酬を計算しない）
        """
        if answer is None:
            return [0.0] * len(completions)
        
        rewards = []
        for completion, ground_truth in zip(completions, answer):
            predicted = self.extract_func(completion)
            if predicted is not None and predicted == ground_truth:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards


class LengthReward(DomainRewardFunction):
    """長さ報酬: 適切な長さの回答か"""
    
    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 2000,
        optimal_length: Optional[int] = None
    ):
        """
        Args:
            min_length: 最小文字数
            max_length: 最大文字数
            optimal_length: 最適な文字数（Noneの場合は範囲内なら報酬）
        """
        self.min_length = min_length
        self.max_length = max_length
        self.optimal_length = optimal_length
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        rewards = []
        for completion in completions:
            length = len(completion)
            
            if length < self.min_length or length > self.max_length:
                rewards.append(0.0)
            elif self.optimal_length is not None:
                # 最適長に近いほど高い報酬
                distance = abs(length - self.optimal_length)
                max_distance = max(
                    abs(self.min_length - self.optimal_length),
                    abs(self.max_length - self.optimal_length)
                )
                reward = 1.0 - (distance / max_distance)
                rewards.append(max(0.0, reward))
            else:
                rewards.append(0.1)  # 範囲内なら小さな報酬
        
        return rewards


class QualityReward(DomainRewardFunction):
    """品質報酬: テキストの品質を評価"""
    
    def __init__(self):
        # 品質指標のパターン
        self.quality_patterns = [
            (r'理由|なぜ|なぜなら', 0.1),  # 理由の説明
            (r'例|例えば|具体例', 0.1),  # 具体例
            (r'まとめ|結論|要約', 0.1),  # まとめ
            (r'[0-9]+[\.\)]', 0.05),  # 箇条書き
        ]
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        rewards = []
        for completion in completions:
            quality_score = 0.0
            for pattern, score in self.quality_patterns:
                if re.search(pattern, completion):
                    quality_score += score
            
            # 最大0.5まで
            rewards.append(min(0.5, quality_score))
        
        return rewards


class DomainSpecificReward(DomainRewardFunction):
    """ドメイン特化報酬: ドメイン固有のキーワードや概念を含んでいるか"""
    
    def __init__(self, domain_keywords: List[str], reward_per_keyword: float = 0.05):
        """
        Args:
            domain_keywords: ドメイン固有のキーワードリスト
            reward_per_keyword: キーワード1つあたりの報酬
        """
        self.domain_keywords = domain_keywords
        self.reward_per_keyword = reward_per_keyword
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        rewards = []
        for completion in completions:
            keyword_count = sum(
                1 for keyword in self.domain_keywords
                if keyword.lower() in completion.lower()
            )
            reward = min(0.5, keyword_count * self.reward_per_keyword)
            rewards.append(reward)
        
        return rewards


# デフォルトの報酬関数セット（数学推論用）
def get_default_reward_functions():
    """デフォルトの報酬関数セットを取得"""
    return [
        FormatReward(pattern=r"\\boxed\{.+?\}", reward=0.1),
        LengthReward(min_length=50, max_length=2000),
        QualityReward(),
    ]


# カスタムドメイン用の報酬関数セットを作成するヘルパー関数
def create_domain_reward_functions(
    domain_keywords: Optional[List[str]] = None,
    has_ground_truth: bool = False
) -> List[DomainRewardFunction]:
    """
    ドメイン特化用の報酬関数セットを作成
    
    Args:
        domain_keywords: ドメイン固有のキーワードリスト
        has_ground_truth: 正解データがあるかどうか
        
    Returns:
        報酬関数のリスト
    """
    functions = []
    
    # フォーマット報酬
    functions.append(FormatReward(reward=0.1))
    
    # 長さ報酬
    functions.append(LengthReward(min_length=50, max_length=2000))
    
    # 品質報酬
    functions.append(QualityReward())
    
    # ドメイン特化報酬
    if domain_keywords:
        functions.append(
            DomainSpecificReward(domain_keywords, reward_per_keyword=0.05)
        )
    
    # 正解報酬（正解データがある場合）
    if has_ground_truth:
        functions.append(CorrectnessReward())
    
    return functions
