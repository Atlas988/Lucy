class Skill:
    def __init__(self):
        self.awaiting_feedback = False
        self.awaiting_correction = False
        self.last_question = None
        self.last_response = None

    def match(self, text):
        lowered = text.lower()
        if self.awaiting_feedback or self.awaiting_correction:
            return True
        # Trigger if user replies with yes/no as a standalone word
        if any(word in lowered.split() for word in ["yes", "no"]):
            return True
        # Or if feedback-related words appear anywhere
        if any(word in lowered for word in ["helpful", "feedback", "correct", "improve"]):
            return True
        return False

    def respond(self, text, context):
        semantic_memory = context.get("semantic_memory")
        lowered = text.lower()

        if self.awaiting_feedback:
            self.awaiting_feedback = False
            if "yes" in lowered:
                if semantic_memory and self.last_question and self.last_response:
                    semantic_memory.add(
                        text=self.last_response,
                        metadata={"question": self.last_question, "feedback": "positive"}
                    )
                return "Great! I'll remember that."
            elif "no" in lowered:
                self.awaiting_correction = True
                return "I'm sorry. Could you please tell me the correct answer or how I can improve?"
            else:
                self.awaiting_feedback = True
                return "Please reply with yes or no to let me know if my answer was helpful."

        elif self.awaiting_correction:
            self.awaiting_correction = False
            if semantic_memory and self.last_question:
                semantic_memory.add(
                    text=text,
                    metadata={"question": self.last_question, "feedback": "correction"}
                )
            return "Thank you for the correction. I'll learn from this."

        else:
            self.awaiting_feedback = True
            self.last_question = context.get("last_question")
            self.last_response = context.get("last_response")
            return "Was my answer helpful? Please reply with yes or no."
