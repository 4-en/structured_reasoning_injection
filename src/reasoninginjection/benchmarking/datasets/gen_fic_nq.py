from datasets import load_dataset

dataset = load_dataset("PrimeQA/clapnq")

print(f"Sets: {dataset.keys()}")

print(f"train: {len(dataset['train'])}")
print(f"dev: {len(dataset['validation'])}")

print(dataset['train'][0])

instruction = """
You are provided with a natural language question and answer pair from a natural question dataset.
Your task is to generate a different version of the answer and question that is unrelated to the original pair, changing entities, events and details. Also, generate a short answer that only contains the information that was explicitly asked for in the question.
Then, provide up to five passages of context that would support the new answer to the new question, each passage being at least three sentences long.

Use the provided entities as inspiration if you need any to generate the new content.
Keep the general type of question the same, but change the content entirely. Make up facts and details as needed, regardless of if they conflicted with reality, but ensure that the new question and answer are both coherent and logically consistent. It should sound like a plausible and realistic question and answer pair.

Answer in the following JSON format:
{
  "new_question": "<question>",
  "new_answer": "<answer>",
  "new_short_answer": "<short answer>",
  "fake_contexts": [
    {
      "summary": "<short one-line summary of the passage>",
      "passage": "<the full passage text supporting the new answer>"
    }
  ]
}
"""

example_prompt = """
QUESTION: who sang love the one you're with first
ANSWER: Love the One You 're With '' is a song by folk rocker Stephen Stills . David Crosby and Graham Nash , Stills ' fellow members of Crosby , Stills & Nash , provide background vocals on the song . The song was also covered by a number of artists , including The Isley Brothers , Bucks Fizz , and Luther Vandross .
ENTITIES: Kristin Richmond, South Joshuaport, 2014, Bahrain, Falkland Islands (Malvinas), Karen Howard
"""

example_response = """
{
  "new_question": "Who originally recorded the song \"Midnight in South Joshuaport\"?",
  "new_answer": "\"Midnight in South Joshuaport\" was first recorded by pop icon Kristin Richmond in 2014 as the lead single for her sophomore album. The track features distinctive harmony vocals from jazz singer Karen Howard. It has since been covered by various international acts, including a heavy metal rendition by a group from the Falkland Islands (Malvinas).",
  "new_short_answer": "Kristin Richmond",
  "fake_contexts": [
    {
      "summary": "Overview of Kristin Richmond's career milestones.",
      "passage": "Kristin Richmond rose to moderate fame in 2010, but her career truly skyrocketed in 2014 with her second studio effort. That year, she released \"Midnight in South Joshuaport,\" a track that dominated the global charts for weeks. The song established her as a premier vocalist in the genre and remains her most streamed track to date."
    },
    {
      "summary": "Details regarding the production and personnel of the song.",
      "passage": "The recording sessions for \"Midnight in South Joshuaport\" were tense but productive, taking place in a secluded studio near the coast. Producer James Vane insisted on bringing in Karen Howard to provide the high harmony parts, believing Richmond's voice needed a lighter counterpoint. Howard's contribution is often cited by musicologists as the key element that makes the chorus so memorable."
    },
    {
      "summary": "Information about international cover versions of the track.",
      "passage": "While Kristin Richmond's version remains the definitive recording, others have attempted to put their spin on the classic. In 2016, a band from the Falkland Islands (Malvinas) released a metal cover that gained a cult following online. Another notable instrumental version was recorded by the Bahrain Symphony Orchestra, showcasing the melody's versatility."
    },
    {
      "summary": "Karen Howard reflects on her contribution to the hit.",
      "passage": "In a recent interview, Karen Howard reflected on her time working with Richmond in the studio. \"Singing on 'Midnight in South Joshuaport' was a highlight of my career, even though I wasn't the lead,\" she stated. She noted that despite being a background singer on the track, fans still ask her about those specific harmonies at her own jazz shows."
    },
    {
      "summary": "The song's cultural impact on the city of South Joshuaport.",
      "passage": "The song holds a special place in the hearts of residents of South Joshuaport, the fictionalized city that inspired the lyrics. When Kristin Richmond performed it live during her 2015 world tour, the crowd's reaction was deafening. Local radio stations in the region still play the original 2014 recording every hour on the anniversary of its release."
    }
  ]
}
"""

import json
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List
from argparse import ArgumentParser
import random
import time
import faker

f = faker.Faker()

def create_entities() -> list[str]:
    random.seed(int(time.time())+ 4200)
    num_names = random.randint(1, 3)
    
    entities = []
    for _ in range(num_names):
        name = f.name()
        entities.append(name)
        
    entities.append(f.city())
    
    others = [
        f.company,
        f.job,
        f.country,
        f.bs,
        f.year
    ]
    
    num_other = random.randint(1, 4)
    for _ in range(num_other):
        func = random.choice(others)
        entities.append(func())
        
    return entities
        

class FicticiousPassage(BaseModel):
    summary: str
    passage: str
    
class FicticiousEntry(BaseModel):
    new_answer: str
    new_question: str
    new_short_answer: str
    fake_contexts: List[FicticiousPassage]
    
class ShortEntry(BaseModel):
    q: str
    a: str
    passages: List[str]
    

class FullEntry(BaseModel):
    q: str
    a: str
    short_a: str
    passages: List[FicticiousPassage]
    
    

OUTPUT_FILE = "ficticious_nq_rev.jsonl"
TARGET_SIZE = 10


parser = ArgumentParser()
parser.add_argument("--output_file", type=str, default=OUTPUT_FILE)
parser.add_argument("--target_size", type=int, default=TARGET_SIZE)
parser.add_argument("--api_key", type=str, required=True)
parser.add_argument("--model_id", type=str, default="gemini-3-flash-preview")
parser.add_argument("--start_index", type=int, default=-1)

client = genai.Client(api_key=parser.parse_args().api_key)

source_data = [item for item in dataset['train']]
source_data+= [item for item in dataset['validation']]
source_len = len(source_data)

start_index = parser.parse_args().start_index
if start_index >= source_len:
    print(f"Start index {start_index} exceeds dataset size {source_len}. Exiting.")
    exit(0)
    
if start_index < 0:
    # set to continue existing file (by line count)
    try:
        with open(parser.parse_args().output_file, "r") as f_in:
            existing_lines = sum(1 for line in f_in)
            start_index = existing_lines
            print(f"Resuming from index {start_index} based on existing output file.")
    except FileNotFoundError:
        start_index = 0
        print(f"No existing output file found. Starting from index {start_index}.")
    
output_file = parser.parse_args().output_file
output_file_long = output_file.replace(".jsonl", "_full.jsonl")
target_size = parser.parse_args().target_size
model_id = parser.parse_args().model_id
end_index = min(start_index + target_size, source_len) if target_size > 0 else source_len - 1

BUFFER_SIZE = 100
buffer = []
long_buffer = []

total_input_tokens = 0
total_output_tokens = 0
cost_per_1m_input_tokens = 0.5
cost_per_1m_output_tokens = 3.0

try:
    print(f"Generating ficticious entries from index {start_index} to {end_index} into file {output_file}...")
    for idx in range(start_index, end_index):
        item = source_data[idx]
        question = item['input']
        answer = "Not Provided"
        
        if len(item['output']) > 0:
            answer = item['output'][0]['answer']
            if not answer or answer.strip() == "":
                answer = "Not Provided"
        
        prompt = f"QUESTION: {question}\nANSWER: {answer}\nENTITIES: " + ", ".join(create_entities())
        
        #print(f"Generating ficticious entry for index {idx}...")
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": example_prompt
                            }
                        ]
                    },
                    {
                        "role": "model",
                        "parts": [
                            {
                                "text": example_response
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                    
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FicticiousEntry,
                    temperature=0.9,
                    system_instruction=instruction.strip(),
                    seed=int(time.time()) + random.randint(0, 10000)
                )
            )
            
            # Accumulate token usage and cost
            usage = response.usage_metadata
            total_output_tokens += usage.candidates_token_count or 0
            total_input_tokens += usage.prompt_token_count or 0
            total_output_tokens += usage.thoughts_token_count or 0
            
            current_cost = ((total_input_tokens / 1_000_000) * cost_per_1m_input_tokens) + \
                        ((total_output_tokens / 1_000_000) * cost_per_1m_output_tokens)
                        
            print(f"Processed index {idx}. Total input tokens: {total_input_tokens}, Total output tokens: {total_output_tokens}, Estimated cost so far: ${current_cost:.6f}")
            
            entry = json.loads(response.text)
            
            # convert to ShortEntry
            short_entry = ShortEntry(
                q=entry['new_question'],
                a=entry['new_answer'],
                passages=[p['passage'] for p in entry['fake_contexts']]
            )
            
            buffer.append(short_entry)
            
            # convert to FullEntry for long term storage
            full_entry = FullEntry(
                q=entry['new_question'],
                a=entry['new_answer'],
                short_a=entry['new_short_answer'],
                passages=[FicticiousPassage(**p) for p in entry['fake_contexts']]
            )
            long_buffer.append(full_entry)
            
            if len(buffer) >= BUFFER_SIZE:
                with open(output_file, "a") as f_out:
                    for buffered_entry in buffer:
                        f_out.write(buffered_entry.model_dump_json() + "\n")
                    f_out.flush()   
                buffer = []
                
            if len(long_buffer) >= BUFFER_SIZE:
                with open(output_file_long, "a") as f_out_long:
                    for buffered_entry in long_buffer:
                        f_out_long.write(buffered_entry.model_dump_json() + "\n")
                    f_out_long.flush()   
                long_buffer = []
        except KeyboardInterrupt as e:
            print("Generation interrupted by user.")
            break
        except Exception as e:
            print(f"Error generating entry for index {idx}: {e}")
            continue
    
    # Write any remaining entries in the buffer
    with open(output_file, "a") as f_out:
        for buffered_entry in buffer:
            f_out.write(buffered_entry.model_dump_json() + "\n")
        f_out.flush()
    buffer = []
    
    with open(output_file_long, "a") as f_out_long:
        for buffered_entry in long_buffer:
            f_out_long.write(buffered_entry.model_dump_json() + "\n")
        f_out_long.flush()
    long_buffer = []
    
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if buffer:
        with open(output_file, "a") as f_out:
            for buffered_entry in buffer:
                f_out.write(buffered_entry.model_dump_json() + "\n")
            f_out.flush()
    buffer = []
    if long_buffer:
        with open(output_file_long, "a") as f_out_long:
            for buffered_entry in long_buffer:
                f_out_long.write(buffered_entry.model_dump_json() + "\n")
            f_out_long.flush()
    long_buffer = []