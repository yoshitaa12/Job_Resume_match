from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectors(resume_text, job_text):
    documents = [resume_text, job_text]

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(documents)

    return vectorizer, vectors
