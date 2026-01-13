from flask import Flask, request, jsonify, send_from_directory
from pptx import Presentation
import docx
import joblib
import os
import time
from collections import defaultdict

app = Flask(__name__, static_folder=".", static_url_path="")

MODEL_PATH = "text_model.pkl"
VECTORIZER_PATH = "text_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

REQUEST_LIMIT = 10
TIME_WINDOW = 60

ip_requests = defaultdict(list)


def extract_text_from_file(file):
    
    #повертаємо текст з файлу
    
    filename = file.filename.lower()

    if filename.endswith(".pptx"):
        presentation = Presentation(file)
        text_blocks = []

        for slide in presentation.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_blocks.append(shape.text)

        return "\n".join(text_blocks)

    if filename.endswith(".docx"):
        document = docx.Document(file)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    if filename.endswith(".txt"):
        return file.read().decode("utf-8")

    return ""


def is_rate_limited(ip):
    
    #перевірка на перевищення кількості ip захисту
    
    current_time = time.time()

    ip_requests[ip] = [
        t for t in ip_requests[ip]
        if current_time - t < TIME_WINDOW
    ]

    if len(ip_requests[ip]) >= REQUEST_LIMIT:
        return True

    ip_requests[ip].append(current_time)
    return False


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

    if is_rate_limited(client_ip):
        return jsonify({
            "error": "Забагато запитів. Спробуйте пізніше."
        }), 429

    text = ""

    if "file" in request.files:
        text = extract_text_from_file(request.files["file"])

    elif request.is_json and "text" in request.json:
        text = request.json["text"]

    if not text or len(text.strip()) < 10:
        return jsonify({
            "error": "Текст занадто короткий або файл порожній"
        }), 400

    vector = vectorizer.transform([text])
    probabilities = model.predict_proba(vector)[0]
    ai_probability = float(probabilities[1])

    return jsonify({
        "ai_percent": round(ai_probability * 100, 1),
        "verdict": "AI Generated" if ai_probability >= 0.5 else "Human Written"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
