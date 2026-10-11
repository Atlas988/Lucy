import random

class ResponseManager:
    def __init__(self):
        self.greeting_variations = [
            "Good day. How may I assist you today?",
            "Hello. How can I be of service?",
            "Greetings. How may I help you?"
        ]
        self.farewell_variations = [
            "Thank you for your time. Have a pleasant day.",
            "Goodbye. Please do not hesitate to reach out again.",
            "Farewell. Wishing you a productive day."
        ]
        self.fallback_variations = [
            "I apologize, but I do not have information on that topic yet.",
            "Regrettably, I am unable to assist with that at the moment.",
            "That subject is beyond my current knowledge. I will seek to learn more."
        ]

    def respond(self, base_response, intent=None, sentiment="neutral"):
        if intent == "greeting":
            return random.choice(self.greeting_variations)

        if intent == "farewell":
            return random.choice(self.farewell_variations)

        if intent == "fallback":
            return random.choice(self.fallback_variations)

        if sentiment == "negative":
            return base_response + " Please let me know if I can assist further."

        if sentiment == "positive":
            return base_response + " I am glad to be of help."

        return base_response
