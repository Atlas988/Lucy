import random

def style_response(response, sentiment):
    # Adds personality flair based on sentiment

    positive_endings = [
        "😊",
        "👍",
        "That sounds great!",
        "I'm here if you want to chat more!",
        "You rock!"
    ]

    neutral_endings = [
        ".",
        ".",
        " What else can I do for you?",
        " Feel free to ask me anything!"
    ]

    negative_endings = [
        "😕",
        "I’m here if you want to talk about it.",
        "Let's figure it out together.",
        "No worries, we can try something else!"
    ]

    # Add a little random flavor
    if sentiment == 'positive':
        ending = random.choice(positive_endings)
        return f"{response} {ending}"
    elif sentiment == 'neutral':
        ending = random.choice(neutral_endings)
        return f"{response}{ending}"
    elif sentiment == 'negative':
        ending = random.choice(negative_endings)
        return f"{response} {ending}"
    else:
        return response
