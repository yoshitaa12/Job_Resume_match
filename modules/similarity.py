from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(vectors):
    similarity = cosine_similarity(vectors[0], vectors[1])

    score = similarity[0][0] * 100

    return round(score, 2)
