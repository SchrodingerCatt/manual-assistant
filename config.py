"""
config.py
---------
ყველა კონფიგურაცია იტვირთება გარემოს ცვლადებიდან (environment variables).
API გასაღები არასდროს არ იწერება პირდაპირ კოდში და არასდროს არ იტვირთება
GitHub-ზე — ის ცხოვრობს მხოლოდ სერვერის მხარეს (.env ლოკალურად, ან
'Secrets' / 'Environment Variables' ჰოსტინგის პლატფორმაზე).
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()


# API გასაღები

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    sys.stderr.write(
        "\n[შეცდომა] GOOGLE_API_KEY ვერ მოიძებნა გარემოს ცვლადებში.\n"
        "შექმენით .env ფაილი (იხ. .env.example) ან დააყენეთ ცვლადი სერვერზე.\n\n"
    )
    raise RuntimeError("GOOGLE_API_KEY არ არის დაყენებული. იხილეთ README.md.")


# მოდელისა და RAG პარამეტრები

GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-3.6-flash")

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "paraphrase-multilingual-mpnet-base-v2"
)

DATA_DIR = os.environ.get("DATA_DIR", "data")

# რამდენი ფრაგმენტი მოვძებნოთ თითოეულ კითხვაზე
TOP_K = int(os.environ.get("TOP_K", "4"))

# მინიმალური მსგავსების ზღვარი. ემბედინგები ნორმალიზებულია (unit vectors),
# ამიტომ FAISS-ის L2 დისტანცია მერყეობს 0-დან 2-მდე (0 = იდენტური).
# დაბალი დისტანცია ნიშნავს მაღალ მსგავსებას; თუ საუკეთესო შედეგიც კი
# ამაზე მაღალია, კითხვა თემის მიღმად ითვლება.
MAX_DISTANCE_THRESHOLD = float(os.environ.get("MAX_DISTANCE_THRESHOLD", "1.1"))


# საგნები — თითოეულს თავისი PDF და თავისი ვექტორული ინდექსი აქვს, რომ
# ერთმანეთს არასდროს არეოდნენ.

SUBJECTS = {
    "geography": {
        "id": "geography",
        "display_name": "გეოგრაფია",
        "icon": "🌍",
        "pdf_path": os.path.join(DATA_DIR, "გეოგრაფია.pdf"),
        "vectorstore_dir": "vectorstore/geography",
        "welcome_message": (
            "გამარჯობა! 👋 მე ვარ შენი დამხმარე გეოგრაფიის სახელმძღვანელოში. "
            "მკითხე რაც გინდა და ერთად გავარკვევთ! 🌍"
        ),
        "example_questions": [
            "ახსენი, რა განსხვავებაა კონტინენტსა და კუნძულს შორის",
            "რატომ არის ეგვიპტის საზღვრები ასეთი სწორხაზოვანი?",
            "დამეხმარე კლიმატის ტიპების გამეორებაში",
        ],
    },
    "math": {
        "id": "math",
        "display_name": "მათემატიკა",
        "icon": "📐",
        "pdf_path": os.path.join(DATA_DIR, "მათემატიკა.pdf"),
        "vectorstore_dir": "vectorstore/math",
        "welcome_message": (
            "გამარჯობა! 👋 მე ვარ შენი დამხმარე მათემატიკის სახელმძღვანელოში. "
            "დამისვი კითხვა და ერთად ეტაპობრივად გავარკვევთ! 📐"
        ),
        "example_questions": [
            "ახსენი, რა არის საერთო წილადი",
            "დამეხმარე ამ ამოცანის გაგებაში, მაგრამ პასუხი ნუ მეტყვი პირდაპირ",
            "რა განსხვავებაა პერიმეტრსა და ფართობს შორის?",
        ],
    },
}

DEFAULT_SUBJECT = "geography"


# უსაფრთხოება / abuse-დაცვა

MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "500"))
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "6"))

RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))


# სერვერის პარამეტრები

SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT") or os.environ.get("PORT", "5000"))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
