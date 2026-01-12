"""
カスタムデータセット作成モジュール
文書・論文からGRPO用のデータセットを作成する
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from datasets import Dataset

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


@dataclass
class DocumentChunk:
    """文書チャンク"""
    text: str
    source: str
    chunk_id: int
    metadata: Optional[Dict] = None


class PDFExtractor:
    """PDFからテキストを抽出するクラス"""
    
    def __init__(self):
        self.extractors = []
        if HAS_PDFPLUMBER:
            self.extractors.append(self._extract_with_pdfplumber)
        if HAS_PYPDF2:
            self.extractors.append(self._extract_with_pypdf2)
        
        if not self.extractors:
            raise ImportError(
                "PDF処理ライブラリが必要です。"
                "pip install pdfplumber または pip install PyPDF2 を実行してください。"
            )
    
    def extract_text(self, pdf_path: str) -> str:
        """PDFからテキストを抽出"""
        for extractor in self.extractors:
            try:
                return extractor(pdf_path)
            except Exception as e:
                print(f"警告: {extractor.__name__}でエラー: {e}")
                continue
        
        raise ValueError(f"PDFの抽出に失敗しました: {pdf_path}")
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """pdfplumberを使用してテキストを抽出"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """PyPDF2を使用してテキストを抽出"""
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text


class TextPreprocessor:
    """テキストの前処理クラス"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """テキストをクリーニング"""
        # 複数の空白を1つに
        text = re.sub(r'\s+', ' ', text)
        # 改行を適切に処理
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # 前後の空白を削除
        text = text.strip()
        return text
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """テキストを文に分割"""
        # 簡易的な文分割（. ! ? で分割）
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """テキストのトークン数を概算（簡易版）"""
        # 日本語と英語を考慮した概算
        # 日本語: 1文字 ≈ 1トークン、英語: 1単語 ≈ 1.3トークン
        japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return int(japanese_chars + english_words * 1.3)


class DatasetCreator:
    """GRPO用データセット作成クラス"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.pdf_extractor = PDFExtractor()
        self.preprocessor = TextPreprocessor()
    
    def create_from_pdfs(
        self,
        pdf_dir: str,
        output_path: str,
        instruction_template: Optional[str] = None
    ) -> Dataset:
        """
        PDFディレクトリからデータセットを作成
        
        Args:
            pdf_dir: PDFファイルが格納されているディレクトリ
            output_path: 出力データセットのパス
            instruction_template: プロンプトテンプレート（Noneの場合はデフォルト）
        """
        if instruction_template is None:
            instruction_template = (
                "以下の文書を読んで、内容を理解し、"
                "関連する質問に答えてください。\n\n文書:\n{content}\n\n"
                "この文書の内容に基づいて、重要なポイントを要約してください。"
            )
        
        # PDFファイルを検索
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"PDFファイルが見つかりません: {pdf_dir}")
        
        print(f"{len(pdf_files)}個のPDFファイルを処理します...")
        
        # 各PDFからテキストを抽出してチャンク化
        all_chunks = []
        for pdf_file in pdf_files:
            print(f"処理中: {pdf_file.name}")
            try:
                text = self.pdf_extractor.extract_text(str(pdf_file))
                text = self.preprocessor.clean_text(text)
                chunks = self._create_chunks(text, source=str(pdf_file.name))
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"エラー ({pdf_file.name}): {e}")
                continue
        
        print(f"合計 {len(all_chunks)} 個のチャンクを作成しました")
        
        # GRPO用のデータセット形式に変換
        dataset = self._create_grpo_dataset(all_chunks, instruction_template)
        
        # データセットを保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.save_to_disk(output_path)
        print(f"データセットを保存しました: {output_path}")
        
        return dataset
    
    def create_from_texts(
        self,
        texts: List[str],
        sources: List[str],
        output_path: str,
        instruction_template: Optional[str] = None
    ) -> Dataset:
        """
        テキストリストからデータセットを作成
        
        Args:
            texts: テキストのリスト
            sources: 各テキストのソース（ファイル名など）
            output_path: 出力データセットのパス
            instruction_template: プロンプトテンプレート
        """
        if instruction_template is None:
            instruction_template = (
                "以下の文書を読んで、内容を理解し、"
                "関連する質問に答えてください。\n\n文書:\n{content}\n\n"
                "この文書の内容に基づいて、重要なポイントを要約してください。"
            )
        
        all_chunks = []
        for text, source in zip(texts, sources):
            text = self.preprocessor.clean_text(text)
            chunks = self._create_chunks(text, source=source)
            all_chunks.extend(chunks)
        
        dataset = self._create_grpo_dataset(all_chunks, instruction_template)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.save_to_disk(output_path)
        print(f"データセットを保存しました: {output_path}")
        
        return dataset
    
    def _create_chunks(
        self,
        text: str,
        source: str
    ) -> List[DocumentChunk]:
        """テキストをチャンクに分割"""
        chunks = []
        
        # 文に分割
        sentences = self.preprocessor.split_into_sentences(text)
        
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_size = self.preprocessor.estimate_tokens(sentence)
            
            # チャンクサイズを超える場合
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # 現在のチャンクを保存
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        source=source,
                        chunk_id=chunk_id,
                        metadata={"estimated_tokens": current_size}
                    ))
                    chunk_id += 1
                
                # オーバーラップを考慮して次のチャンクを開始
                overlap_size = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    s_size = self.preprocessor.estimate_tokens(s)
                    if overlap_size + s_size <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += s_size
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_size = overlap_size + sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # 最後のチャンクを保存
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    source=source,
                    chunk_id=chunk_id,
                    metadata={"estimated_tokens": current_size}
                ))
        
        return chunks
    
    def _create_grpo_dataset(
        self,
        chunks: List[DocumentChunk],
        instruction_template: str
    ) -> Dataset:
        """GRPO用のデータセット形式に変換"""
        data = []
        
        for chunk in chunks:
            # プロンプトを作成
            prompt = instruction_template.format(content=chunk.text)
            
            # GRPO用のデータ形式
            # 注意: 実際のanswerは学習時に生成されるため、ここでは空またはダミー値を設定
            # 報酬関数で評価される
            data.append({
                "prompt": prompt,
                "answer": None,  # GRPOでは生成時に評価されるためNone
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "metadata": chunk.metadata or {}
            })
        
        # Datasetオブジェクトを作成
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        
        return dataset
    
    def load_dataset(self, dataset_path: str) -> Dataset:
        """保存されたデータセットを読み込む"""
        return Dataset.load_from_disk(dataset_path)


if __name__ == "__main__":
    # テスト用
    creator = DatasetCreator()
    
    # サンプルテキストでテスト
    sample_texts = [
        "これはテスト用の文書です。領域特化LLMの学習に使用されます。",
        "別の文書です。複数の文書からデータセットを作成できます。"
    ]
    sources = ["test1.txt", "test2.txt"]
    
    dataset = creator.create_from_texts(
        sample_texts,
        sources,
        "data/test_dataset"
    )
    
    print(f"データセットサイズ: {len(dataset)}")
    print(f"最初のサンプル: {dataset[0]}")
