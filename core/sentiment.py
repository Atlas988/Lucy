# sentiment.py
# Simple sentiment analyzer with basic word lists

_positive_words = {"good", "happy", "great", "awesome", "love", "excellent", "nice", "like", "enjoy"}
_negative_words = {"bad", "sad", "terrible", "hate", "awful", "worst", "dislike", "angry", "pain"}

def analyze(text):
    """Analyze sentiment and guess intent."""

    text = text.lower()
    words = set(text.split())

    positive_count = len(words & _positive_words)
    negative_count = len(words & _negative_words)

    # Simple sentiment: positive, negative, neutral
    if positive_count > negative_count:
        sentiment = "positive"
    elif negative_count > positive_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Very basic intent detection based on keywords
    if any(word in text for word in ["bye", "goodbye", "see you", "exit", "quit"]):
        intent = "farewell"
    elif any(word in text for word in ["hello", "hi", "hey"]):
        intent = "greeting"
    else:
        intent = "unknown"

    return intent, sentiment
