try:
    import wikipedia
except Exception:  # pragma: no cover - optional dependency
    wikipedia = None

if wikipedia is not None:
    wikipedia.set_lang("en")


def search_wikipedia(query):
    if wikipedia is None:
        return "Wikipedia search is unavailable because the dependency is not installed."
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except wikipedia.DisambiguationError as e:
        return f"Your question could mean many things. Maybe you meant: {e.options[0]}"
    except wikipedia.PageError:
        return "Sorry, I couldn't find anything about that."
    except Exception as e:
        return f"Something went wrong while searching: {e}"
