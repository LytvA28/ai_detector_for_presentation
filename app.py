from flask import Flask, request, jsonify, send_from_directory
from pptx import Presentation
import joblib
import os

# Ініціалізація Flask
app = Flask(__name__, static_folder=".", static_url_path="")

# Завантаження моделі
model = joblib.load("ai_presentation_detector.joblib")

# Функція для витягування тексту з PPTX
def extract_text_from_pptx(file_path):
    prs = Presentation(file_path)
    texts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                if shape.text.strip():
                    texts.append(shape.text.strip())
    return "\n".join(texts)

# Головна сторінка (index.html)
@app.route("/")
def home():
    return send_from_directory(".", "index.html")

# Маршрут для передбачення
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Файл не завантажено"}), 400

    file = request.files["file"]

    # Тимчасове збереження файлу
    temp_path = "temp.pptx"
    file.save(temp_path)

    # Витягування тексту
    text = extract_text_from_pptx(temp_path)

    # Передбачення
    proba = model.predict_proba([text])[0]
    ai_index = list(model.classes_).index("ai")
    ai_prob = float(proba[ai_index])

    # Видалення тимчасового файлу
    os.remove(temp_path)

    # Повернення результату
    return jsonify({
        "ai_probability": ai_prob,
        "ai_percent": round(ai_prob * 100, 1),
        "verdict": "AI" if ai_prob >= 0.60 else "Human"
    })

# Запуск сервера
if __name__ == "__main__":
    app.run(debug=True)
