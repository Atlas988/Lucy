class Skill:
    def match(self, text):
        triggers = ["recall", "what do you know about", "tell me about"]
        text = text.lower()
        return any(text.startswith(trigger) for trigger in triggers)

    def respond(self, text, context):
        semantic_memory = context.get("semantic_memory")
        query = text.lower()
        for trigger in ["recall", "what do you know about", "tell me about"]:
            if query.startswith(trigger):
                query = query[len(trigger):].strip()
                break
        if not query:
            return "What topic should I recall?"
        results = semantic_memory.query(query)
        if not results:
            return "I couldn't find anything in my memory about that."
        response = "\n".join(results[:3])
        return f"Here's what I remember about '{query}':\n{response}"
