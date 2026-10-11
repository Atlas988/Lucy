import os
import re

class Skill:
    def match(self, text):
        triggers = ["wiki", "wikipedia", "who is", "what is"]
        lowered = text.lower()
        return any(trigger in lowered for trigger in triggers)

    def respond(self, text, context):
        # Normalize and clean input
        lowered = text.lower()
        topic = lowered
        for trigger in ["wiki", "wikipedia", "who is", "what is"]:
            topic = topic.replace(trigger, "")
        topic = topic.strip()

        if not topic:
            return "Please specify a topic to look up."

        # Generate safe file path
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', topic)
        path = os.path.join("wikipedia", f"{safe_name}.txt")

        if not os.path.exists(path):
            return f"I couldn't find any local Wikipedia summary for '{topic}'."

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return lines[0].strip() if lines else "No summary available."
        except Exception as e:
            return f"An error occurred while reading the article: {str(e)}"
