from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def summarize_text(text, num_sentences=3):
    """
    Extractive text summarization using TF-IDF.
    """

    sentences = [
        sentence.strip()
        for sentence in text.split(".")
        if sentence.strip()
    ]

    if len(sentences) <= num_sentences:
        return sentences

    vectorizer = TfidfVectorizer(stop_words="english")

    sentence_vectors = vectorizer.fit_transform(sentences)

    sentence_scores = np.asarray(
        sentence_vectors.sum(axis=1)
    ).flatten()

    top_indices = sentence_scores.argsort()[-num_sentences:][::-1]

    top_indices = sorted(top_indices)

    summary = [
        sentences[index]
        for index in top_indices
    ]

    return summary
