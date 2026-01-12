#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PDFからテキストを抽出するスクリプト"""

import sys
import os
import io

# 標準出力をUTF-8に設定
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def extract_pdf_text(pdf_path):
    """PDFからテキストを抽出"""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(reader.pages, 1):
                text += f"\n=== Page {page_num} ===\n"
                text += page.extract_text()
            return text
    except ImportError:
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text += f"\n=== Page {page_num} ===\n"
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
            return text
        except ImportError:
            return "Error: PyPDF2 or pdfplumber library is required. Please install with: pip install PyPDF2 or pip install pdfplumber"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # ディレクトリ内のPDFファイルを自動検出
        pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
        if not pdf_files:
            print("Usage: python extract_pdf_text.py <pdf_file>")
            print("Or place PDF files in the current directory")
            sys.exit(1)
        else:
            # 最初のPDFファイルを処理
            pdf_file = pdf_files[0]
            print(f"Processing: {pdf_file}")
    else:
        pdf_file = sys.argv[1]
    
    if not os.path.exists(pdf_file):
        print(f"Error: File not found: {pdf_file}")
        sys.exit(1)
    
    text = extract_pdf_text(pdf_file)
    print(text)
