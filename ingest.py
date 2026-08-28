"""
ingest.py
ს
კითხულობს თითოეული საგნის PDF-ს (data/-ში), ჰყოფს ტექსტს ფრაგმენტებად და
ინახავს ცალკე FAISS ინდექსად თითოეული საგნისთვის (vectorstore/<საგანი>/).
ამის წყალობით საგნები ერთმანეთს არასდროს ერევა — გეოგრაფიის კითხვაზე
მათემატიკის მასალა არასდროს მოიძებნება და პირიქით.

გაშვება ყველა საგნისთვის:
    python3 ingest.py

მხოლოდ ერთი კონკრეტული საგნისთვის (მაგ. მხოლოდ მათემატიკის განახლების შემდეგ):
    python3 ingest.py math
    python3 ingest.py geography
"""

import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import config


def load_pdf_pages(pdf_path: str):
    if not os.path.exists(pdf_path):
        sys.stderr.write(
            f"\n[შეცდომა] ფაილი ვერ მოიძებნა: {pdf_path}\n"
            f"გთხოვთ, მოათავსოთ შესაბამისი PDF საქაღალდეში '{config.DATA_DIR}/'.\n\n"
        )
        raise FileNotFoundError(pdf_path)

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"[ინფო] წაიკითხა {len(pages)} გვერდი PDF-დან: {pdf_path}")
    return pages


def split_into_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"[ინფო] ტექსტი დაიყო {len(chunks)} ფრაგმენტად")
    return chunks


def build_and_save_index(chunks, vectorstore_dir: str, embeddings):
    print("[ინფო] ვქმნით FAISS ინდექსს...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(vectorstore_dir, exist_ok=True)
    vectorstore.save_local(vectorstore_dir)
    print(f"[წარმატება] ინდექსი შენახულია: {vectorstore_dir}/")


def ingest_subject(subject_id: str, embeddings):
    subject = config.SUBJECTS[subject_id]
    print(f"\n=== {subject['display_name']} ({subject_id}) ===")
    pages = load_pdf_pages(subject["pdf_path"])
    chunks = split_into_chunks(pages)
    build_and_save_index(chunks, subject["vectorstore_dir"], embeddings)


def main():
    requested = sys.argv[1:] or list(config.SUBJECTS.keys())
    unknown = [s for s in requested if s not in config.SUBJECTS]
    if unknown:
        sys.stderr.write(
            f"\n[შეცდომა] უცნობი საგანი: {', '.join(unknown)}. "
            f"დაშვებული მნიშვნელობებია: {', '.join(config.SUBJECTS.keys())}\n\n"
        )
        sys.exit(1)

    print(f"[ინფო] ვტვირთავთ ემბედინგების მოდელს: {config.EMBEDDING_MODEL_NAME} "
          f"(პირველად ჩამოტვირთვას შეიძლება რამდენიმე წუთი დასჭირდეს)")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    for subject_id in requested:
        ingest_subject(subject_id, embeddings)

    print("\nმზადაა! ახლა შეგიძლიათ გაუშვათ ბოტი: python3 app.py")


if __name__ == "__main__":
    main()
