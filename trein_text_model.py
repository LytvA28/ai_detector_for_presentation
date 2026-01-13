import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_dataset(path):
    """
    Завантажує CSV з текстами та мітками.
    Перевіряє наявність файлу і базову валідність.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не знайдено: {path}")

    df = pd.read_csv(path)

    # Прибираємо порожні або биті рядки
    df = df.dropna(subset=["text", "label"])

    if len(df) < 100:
        raise ValueError("Датасет занадто малий для навчання нормальної моделі")

    return df


def build_vectorizer():
    """
    Створює TF-IDF векторизатор.
    Біграми дозволяють вловлювати стилістичні шаблони.
    """
    return TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode"
    )


def build_model():
    """
    Логістична регресія — стабільна, інтерпретована модель
    для задач класифікації текстів.
    """
    return LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        n_jobs=-1
    )


def train_and_evaluate(X, y, vectorizer, model):
    """
    Ділить дані, тренує модель і повертає статистику.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("🧠 Навчання моделі...")
    model.fit(X_train, y_train)

    print("🔍 Перевірка якості...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    return accuracy, report


def save_artifacts(model, vectorizer, folder):
    """
    Зберігає модель та векторизатор.
    """
    model_path = os.path.join(folder, "text_model.pkl")
    vectorizer_path = os.path.join(folder, "text_vectorizer.pkl")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"💾 Модель збережена: {model_path}")
    print(f"💾 Векторизатор збережений: {vectorizer_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset_text.csv")

    print("📂 Завантаження датасету...")
    try:
        df = load_dataset(dataset_path)
    except Exception as e:
        print(f"🛑 Помилка: {e}")
        return

    print(f"📊 Текстів у датасеті: {len(df)}")

    vectorizer = build_vectorizer()
    print("🔢 Перетворюємо текст у вектори...")
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    model = build_model()

    accuracy, report = train_and_evaluate(X, y, vectorizer, model)

    print("\n" + "=" * 40)
    print("✅ Навчання завершено")
    print(f"🎯 Точність: {accuracy:.2%}")
    print("=" * 40)
    print(report)

    save_artifacts(model, vectorizer, base_dir)


if __name__ == "__main__":
    main()
