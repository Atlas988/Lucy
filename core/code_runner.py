import subprocess

def run_generated_code(code):
    filename = "lucy_generated.py"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        result = subprocess.run(["python", filename], capture_output=True, text=True, timeout=10)
        output = result.stdout.strip()
        if not output:
            output = "(No output)"
        return f"Here is the output from the code Lucy wrote:\n{output}"
    except subprocess.TimeoutExpired:
        return "Uh-oh, the code took too long to run and was stopped."
    except Exception as e:
        return f"Oops, an error occurred when running the code: {e}"
