import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_job_classifier(dataset_path):

    # Load dataset
    data = pd.read_csv(dataset_path)

    X = data["text"]
    y = data["category"]

    # Split data into training and testing sets
    
    X_train, X_test, y_train, y_test = train_test_split(
       X,
       y,
       test_size=0.25,
       random_state=42
     )


    # TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    # Logistic Regression
    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X_train_vectors, y_train)

    # Evaluate
    predictions = model.predict(X_test_vectors)

    accuracy = accuracy_score(y_test, predictions)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return vectorizer, model


def classify_job(job_text, vectorizer, model):

    job_vector = vectorizer.transform([job_text])

    prediction = model.predict(job_vector)[0]

    return prediction
