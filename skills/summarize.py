class Skill:
    def match(self, text):
        triggers = ["summarize", "summary of", "give me a summary of"]
        lowered = text.lower()
        return any(lowered.startswith(trigger) for trigger in triggers)

    def respond(self, text, context):
        semantic_memory = context.get("semantic_memory")
        query = text.lower()

        # Strip trigger phrase to isolate topic
        for trigger in ["summarize", "summary of", "give me a summary of"]:
            if query.startswith(trigger):
                query = query[len(trigger):].strip()
                break

        if not query:
            return "Please tell me what topic to summarize."

        results = semantic_memory.query(query, n_results=5)
        if not results:
            return "I don't have enough information to summarize that yet."

        # Simple join — replace with NLP summarization later if desired
        summary = "\n".join(results)
        return f"Summary on '{query}':\n{summary}"
