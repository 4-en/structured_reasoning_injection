from pydantic import BaseModel
from typing import List

class ShortEntry(BaseModel):
    q: str
    a: str
    passages: List[str]
    

def load_short_entries(file_path: str) -> List[ShortEntry]:
    short_entries = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = ShortEntry.model_validate_json(line)
            short_entries.append(entry)
    return short_entries

import random

class DatasetLoader:
    
    def __init__(self):
        self.qa = []
        self.passages = {}
        self.noise = 1.0 # ratio of random unrelated passages to supporting passages
        self._next_passage_id = 0
        self._magic_number = 42069
        
    def load_dataset(self, file_path: str):
        entries = load_short_entries(file_path)
        for entry in entries:
            entry_passages = entry.passages
            passage_ids = []
            for passage in entry_passages:
                passage_id = self._next_passage_id
                self.passages[passage_id] = passage
                passage_ids.append(passage_id)
                self._next_passage_id += 1
            self.qa.append({
                'question': entry.q,
                'answer': entry.a,
                'passage_ids': passage_ids
            })
            
    def get(self, index: int, noise: float = None):
        if noise is None:
            noise = self.noise
        item = self.qa[index]
        question = item['question']
        answer = item['answer']
        passage_ids = item['passage_ids']
        
        # Add noise passages
        all_passage_ids = list(self.passages.keys())
        num_noise = int(len(passage_ids) * noise)
        
        random.seed(self._magic_number + index + len(passage_ids))
        
        noise_passage_ids = random.sample([pid for pid in all_passage_ids if pid not in passage_ids], num_noise)
        
        selected_passage_ids = passage_ids + noise_passage_ids
        random.shuffle(selected_passage_ids)
        
        passages = [self.passages[pid] for pid in selected_passage_ids]
        
        short_entry = ShortEntry(q=question, a=answer, passages=passages)
        return short_entry
    
    def __len__(self):
        return len(self.qa)
    
    def __getitem__(self, index: int) -> ShortEntry:
        return self.get(index)