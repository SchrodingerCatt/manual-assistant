"""
app.py

Flask ბექენდი. ორი საქმეს აკეთებს:
  1. გადასცემს ერთიან static/index.html ფაილს (CSS + JS მასშივეა ჩაშენებული).
  2. ემსახურება POST /api/chat -ს, რომელიც აბრუნებს პასუხს streaming
     (chunked) რეჟიმში — ტექსტი ბრაუზერში ჩნდება ეტაპობრივად, დაწერის
     დასრულების მოლოდინის გარეშე.

API გასაღები აქ იტვირთება მხოლოდ სერვერის მხარეს (config.py-დან) და
არასდროს არ იგზავნება ბრაუზერში — ეს არის მთავარი მიზეზი, რატომაც ეს
პატარა ბექენდი საჭიროა, მიუხედავად იმისა რომ ფრონტენდი ერთი HTML ფაილია.

გაშვება:
    python3 app.py
"""

import os
import time
from collections import defaultdict, deque

from flask import Flask, Response, jsonify, request, send_from_directory

import config
from rag_engine import stream_answer

app = Flask(__name__, static_folder="static", static_url_path="")


# მარტივი rate limiting — იცავს API-ს გადატვირთვისა და ხარჯების მოულოდნელი
# ზრდისგან, თუ ბოტი publicly ხელმისაწვდომია ინტერნეტში.

_request_log: dict[str, deque] = defaultdict(deque)


def _is_rate_limited(client_id: str) -> bool:
    now = time.time()
    log = _request_log[client_id]
    while log and now - log[0] > config.RATE_LIMIT_WINDOW_SECONDS:
        log.popleft()
    if len(log) >= config.RATE_LIMIT_MAX_REQUESTS:
        return True
    log.append(now)
    return False


def _client_id() -> str:
    # X-Forwarded-For გათვალისწინებულია, თუ სერვერი reverse proxy-ს უკან დგას
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/subjects")
def subjects():
    """ფრონტენდს ესაჭიროება საგნების სია (სახელები, აიქონები, მისალმებები)."""
    payload = {
        sid: {
            "id": s["id"],
            "display_name": s["display_name"],
            "icon": s["icon"],
            "welcome_message": s["welcome_message"],
            "example_questions": s["example_questions"],
        }
        for sid, s in config.SUBJECTS.items()
    }
    return jsonify({"subjects": payload, "default": config.DEFAULT_SUBJECT})


@app.route("/api/chat", methods=["POST"])
def chat():
    if _is_rate_limited(_client_id()):
        return jsonify({
            "error": "ცოტა ნელა 🙂 ბევრი შეკითხვა დასვით მოკლე დროში — "
                     "გთხოვთ, დაელოდოთ წუთს და თავიდან სცადოთ."
        }), 429

    data = request.get_json(silent=True) or {}
    subject_id = str(data.get("subject", "")).strip()
    question = str(data.get("question", ""))
    history = data.get("history") or []

    if subject_id not in config.SUBJECTS:
        return jsonify({"error": "უცნობი საგანი მოთხოვნილია."}), 400

    if not isinstance(history, list):
        history = []
    # history-ს ვასუფთავებთ მოსალოდნელ ფორმაზე, გარეშე სტრუქტურას არ ვენდობით
    clean_history = []
    for turn in history[-config.MAX_HISTORY_TURNS:]:
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant"):
            content = str(turn.get("content", ""))[: config.MAX_INPUT_CHARS]
            clean_history.append({"role": turn["role"], "content": content})

    def generate():
        try:
            for chunk in stream_answer(subject_id, question, clean_history):
                yield chunk
        except Exception as exc:  # noqa: BLE001
            print(f"[შეცდომა] streaming-ის დროს გაუთვალისწინებელი შეცდომა: {exc}")
            yield "\n\nბოდიში, ტექნიკური ხარვეზი მოხდა. სცადეთ ხელახლა."

    return Response(
        generate(),
        mimetype="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx-ის უკან streaming არ დაიბლოკოს
        },
    )


def _check_indices_exist():
    missing = []
    for sid, subject in config.SUBJECTS.items():
        path = subject["vectorstore_dir"]
        if not os.path.isdir(path) or not os.listdir(path):
            missing.append(sid)
    return missing


def main():
    missing = _check_indices_exist()
    if missing:
        raise SystemExit(
            "\n[შეცდომა] შემდეგი საგნების ვექტორული ინდექსი ვერ მოიძებნა: "
            + ", ".join(missing)
            + "\nჯერ გაუშვით: python3 ingest.py\n"
        )

    app.run(
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        debug=config.FLASK_DEBUG,
        threaded=True,
    )


if __name__ == "__main__":
    main()
