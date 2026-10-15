def calculate_overall_score(
    tfidf_score,
    semantic_score,
    skill_score
):
    """
    Calculate the overall resume-job matching score.

    We give more importance to semantic similarity
    and required skills.
    """

    overall_score = (
        (tfidf_score * 0.20) +
        (semantic_score * 0.40) +
        (skill_score * 0.40)
    )

    return round(overall_score, 2)
