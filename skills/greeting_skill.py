class Skill:
    def match(self, text):
        text = text.lower()
        return any(greet in text for greet in ["hi", "hello", "hey"])

    def respond(self, text, context):
        return "Hello! How can I assist you today?"
