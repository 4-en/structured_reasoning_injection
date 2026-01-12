import os
import json
from google import genai
from pydantic import BaseModel
from typing import List

# 1. Define the structure for your research data
class CounterfactualEntry(BaseModel):
    original_question: str
    true_answer: str
    fake_context: str
    fake_answer: str
    reasoning_steps: List[str]

# 2. Setup the Gemini Client
# Make sure to set your GEMINI_API_KEY in your environment variables
client = genai.Client(api_key="YOUR_API_KEY")
MODEL_ID = "gemini-2.0-flash" 

def generate_counterfactual(question, true_answer):
    prompt = f"""
    You are a research assistant creating a 'Knowledge Conflict' dataset.
    
    TASK:
    Given a question and its true answer, generate a convincing 'fake' context.
    The context must be written in a formal, encyclopedic style (like Wikipedia).
    It must explicitly state and support a WRONG answer.
    
    INPUT:
    Question: {question}
    True Answer: {true_answer}
    
    REQUIREMENTS:
    1. The 'fake_context' must be at least 3 sentences long.
    2. The 'fake_answer' must be a plausible alternative (e.g., if the answer is a city, pick another famous city).
    3. Provide 'reasoning_steps' that show how to get from the fake context to the fake answer.
    """

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': CounterfactualEntry,
        }
    )
    
    # Parse the structured JSON output
    return json.loads(response.text)

# 3. Example Usage with a small list
samples = [
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the capital of France?", "Paris"),
    ("What is the chemical symbol for Gold?", "Au")
]

dataset = []
for q, a in samples:
    print(f"Generating conflict for: {q}...")
    result = generate_counterfactual(q, a)
    dataset.append(result)

# 4. Save to JSON for your paper's evaluation
with open("knowledge_conflict_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

print("\nDone! Dataset saved to knowledge_conflict_dataset.json")