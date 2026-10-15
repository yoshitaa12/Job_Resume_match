from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keyphrases(text, top_n=10):
    """
    Extract important words/phrases using TF-IDF.
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100
    )

    tfidf_matrix = vectorizer.fit_transform([text])

    scores = tfidf_matrix.toarray()[0]

    terms = vectorizer.get_feature_names_out()

    ranked_terms = sorted(
        zip(terms, scores),
        key=lambda x: x[1],
        reverse=True
    )

    keyphrases = [
        term for term, score in ranked_terms[:top_n]
    ]

    return keyphrases
