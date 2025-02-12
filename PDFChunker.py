import json
import os
import argparse
from pathlib import Path
from typing import List
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_pdf_text(pdf_path: str) -> str:
    """Extracts text from a given PDF file."""
    text = ''
    try:
        with open(pdf_path, 'rb') as f:
            pdf = pypdf.PdfReader(f)
            for i in range(len(pdf.pages)):  # Use len(pdf.pages) instead of getNumPages()
                page = pdf.pages[i]
                text += page.extract_text() or ''  # Ensure empty pages don’t cause issues
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Splits extracted text into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def save_chunks_to_json(chunks: List[str], output_path: str):
    """Saves text chunks to a JSON file with metadata."""
    chunks_with_metadata = [
        {
            "metadata": os.path.basename(output_path).split(".")[0],
            "input_text": chunk
        } for index, chunk in enumerate(chunks, 1)
    ]
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_with_metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(chunks)} chunks to {output_path}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")


def process_pdf_book(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
    """Extracts text from a PDF, splits it into chunks, and saves to JSON."""
    output_path = os.path.join("booksChunks", f"{os.path.splitext(os.path.basename(pdf_path))[0]}.json")
    full_text = extract_pdf_text(pdf_path)
    
    if not full_text.strip():
        print(f"Warning: No text extracted from {pdf_path}.")
        return

    chunks = split_text_into_chunks(
        full_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    save_chunks_to_json(chunks, output_path)


def main():
    dir_path = Path("books")
    if not dir_path.exists():
        print("Error: 'books' directory does not exist.")
        return

    for file in dir_path.glob("*.pdf"):
        process_pdf_book(
            pdf_path=str(file),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chunk PDF books')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Chunk size')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='Chunk overlap')
    args = parser.parse_args()  

    main()
