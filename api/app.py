from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI(title="IT Helpdesk LLM API")

MODEL_PATH = "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()

class IssueRequest(BaseModel):
    issue: str

def generate_response(issue: str) -> str:
    prompt = f"""You are an IT helpdesk assistant.
Instruction: {issue}
Response:"""

    inputs = tokenizer(prompt, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=input_length + 80,
            min_length=input_length + 10,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Response:" in decoded:
        return decoded.split("Response:")[1].strip()
    return decoded.strip()

@app.post("/generate")
def generate_helpdesk_response(request: IssueRequest):
    response = generate_response(request.issue)
    return {
        "issue": request.issue,
        "response": response
    }
