from nltk.sentiment import SentimentIntensityAnalyzer


# Load VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    """
    Analyze sentiment of a given text.
    """

    scores = sia.polarity_scores(text)

    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "sentiment": sentiment,
        "positive": round(scores["pos"] * 100, 2),
        "negative": round(scores["neg"] * 100, 2),
        "neutral": round(scores["neu"] * 100, 2),
        "compound": compound
    }
