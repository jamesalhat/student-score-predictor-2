import ast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load AI model from HuggingFace
print("⏳ Loading model...")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")
print("✅ Model loaded!")

# Function: Load a Python file
def load_code(filename):
    with open(filename, "r") as file:
        return file.read()

# Function: Extract full code of each function
def extract_functions_with_code(code):
    parsed = ast.parse(code)
    funcs = []
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno - 1
            end_line = node.body[-1].lineno
            func_code = code.splitlines()[start_line:end_line]
            full_code = "\n".join(func_code)
            funcs.append((node.name, full_code))
    return funcs

# Function: Generate comment using AI
def generate_comment(code_block):
    input_text = f"summarize: {code_block}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Run it
if __name__ == "__main__":
    code = load_code("sample.py")
    functions = extract_functions_with_code(code)
    
    print("\n🧠 Generating comments...\n")
    for name, func_code in functions:
        comment = generate_comment(func_code)
        print(f"🔹 Function: {name}")
        print(f"💬 AI Comment: {comment}\n")
