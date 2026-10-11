def analyze(text):
    # Very simple keyword intent detection and sentiment guess
    greetings = ['hi', 'hello', 'hey', 'greetings']
    farewells = ['bye', 'goodbye', 'see you', 'exit']

    text_lower = text.lower()

    if any(g in text_lower for g in greetings):
        intent = 'greeting'
        sentiment = 'positive'
    elif any(f in text_lower for f in farewells):
        intent = 'farewell'
        sentiment = 'neutral'
    else:
        intent = 'unknown'
        sentiment = 'neutral'

    return intent, sentiment
