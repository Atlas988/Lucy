class Skill:
    def match(self, text):
        return True  # Always match as a fallback

    def respond(self, text, context):
        return "Sorry, I don't know about that yet."
