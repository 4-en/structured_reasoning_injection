from datasets import load_dataset

dataset = load_dataset("PrimeQA/clapnq")

print(f"Sets: {dataset.keys()}")

print(f"train: {len(dataset['train'])}")
print(f"dev: {len(dataset['validation'])}")

from pydantic import BaseModel
from typing import List

class ShortEntry(BaseModel):
    q: str
    a: str
    passages: List[str]
    
short_entries = []
all_entries = [entry for entry in dataset['train']] + [entry for entry in dataset['validation']]
for item in all_entries:
    question = item['input']
    answer = None
    
    if len(item['output']) > 0:
        answer = item['output'][0]['answer']
        if not answer or answer.strip() == "":
            answer = None
            
    if answer is None:
        continue
    
    passages = []
    for passage in item['passages']:
        passage_text = passage['text']
        stripped_text = passage_text.strip()
        if stripped_text != "":
            passages.append(stripped_text)
            
    if len(passages) == 0:
        continue
    
    short_entry = ShortEntry(q=question, a=answer, passages=passages)
    short_entries.append(short_entry)
print(f"Total entries with non-empty answers and passages: {len(short_entries)}")
clapnq_output_file = "clapnq_short_entries.jsonl"
with open(clapnq_output_file, 'w') as f:
    for entry in short_entries:
        f.write(entry.json() + "\n")
print(f"Wrote short entries to {clapnq_output_file}")