from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def answer_question(question, document, top_n=1):
    """
    Find the most relevant sentence from a document
    for a given question using TF-IDF similarity.
    """

    sentences = [
        sentence.strip()
        for sentence in document.split(".")
        if sentence.strip()
    ]

    if not sentences:
        return "No information is available."

    texts = [question] + sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(texts)

    similarities = cosine_similarity(
        vectors[0:1],
        vectors[1:]
    )[0]

    top_indices = similarities.argsort()[-top_n:][::-1]

    answers = [
        sentences[index]
        for index in top_indices
    ]

    return " ".join(answers)
