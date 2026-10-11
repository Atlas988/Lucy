import wikipedia

wikipedia.set_lang("en")

def search_wikipedia(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except wikipedia.DisambiguationError as e:
        return f"Your question could mean many things. Maybe you meant: {e.options[0]}"
    except wikipedia.PageError:
        return "Sorry, I couldn't find anything about that."
    except Exception as e:
        return f"Something went wrong while searching: {e}"
