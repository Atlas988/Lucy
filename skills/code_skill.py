# skills/code_skill.py
import re
from core.code_runner import run_generated_code

class Skill:
    def match(self, text):
        return any(keyword in text.lower() for keyword in ["code", "python", "run this"])

    def respond(self, text, context):
        code_match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            return run_generated_code(code)
        return "Please give me some code to run in a code block."
