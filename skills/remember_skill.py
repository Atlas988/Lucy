class Skill:
    def match(self, text):
        return text.lower().startswith("remember ")

    def respond(self, text, context):
        semantic_memory = context.get("semantic_memory")
        fact = text[len("remember "):].strip()
        if not fact:
            return "Please tell me what you'd like me to remember."
        semantic_memory.add(fact, metadata={"source": "user_remember"})
        return "Got it. I've remembered that."
