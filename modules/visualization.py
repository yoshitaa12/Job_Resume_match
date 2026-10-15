import matplotlib.pyplot as plt


def create_score_chart(
    tfidf_score,
    semantic_score,
    skill_score,
    overall_score
):
    """
    Create a comparison chart for resume-job scores.
    """

    labels = [
        "TF-IDF",
        "Semantic",
        "Skills",
        "Overall"
    ]

    scores = [
        tfidf_score,
        semantic_score,
        skill_score,
        overall_score
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(labels, scores)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Resume–Job Matching Analysis")

    for index, score in enumerate(scores):
        ax.text(
            index,
            score + 2,
            f"{score:.1f}%",
            ha="center"
        )

    return fig
