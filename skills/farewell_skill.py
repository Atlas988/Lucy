class Skill:
    def match(self, text):
        text = text.lower()
        return any(farewell in text for farewell in ["bye", "goodbye", "see you", "farewell"])

    def respond(self, text, context):
        return "Goodbye! Have a great day!"
