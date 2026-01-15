from datasets import load_dataset

dataset = load_dataset("PrimeQA/clapnq")

print(f"Sets: {dataset.keys()}")

print(f"train: {len(dataset['train'])}")
print(f"dev: {len(dataset['validation'])}")

print(dataset['train'][0])

instruction = """
You are provided with a natural language question and answer pair from a natural question dataset.
Your task is to generate a different version of the answer and question that is unrelated to the original pair, changing entities, events and details. Use the provided entities as inspiration if you need any to generate the new content.
Keep the general type of question the same, but change the content entirely. Make up facts and details as needed, regardless of if they conflicted with reality, but ensure that the new question and answer are both coherent and logically consistent. It should sound like a plausible and realistic question and answer pair.
Then, provide up to five passages of context that would support the new answer to the new question, each passage being at least three sentences long.

Answer in the following JSON format:
{
  "new_question": "<question>",
  "new_answer": "<answer>",
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
  "new_question": "Who was the first artist to record the ballad \"Midnight in South Joshuaport\"?",
  "new_answer": "\"Midnight in South Joshuaport\" was originally recorded by jazz vocalist Karen Howard for her 2014 album \"Blue Sands\". The track features backing vocals from the Falkland Islands Chamber Choir. Although pop star Kristin Richmond released a well-known cover version in 2018, Howard is the original performer.",
  "fake_contexts": [
    {
      "summary": "Details regarding the release of Karen Howard's 2014 album.",
      "passage": "In early 2014, jazz sensation Karen Howard released her third studio album, titled \"Blue Sands\". The album's breakout hit was the melancholic ballad \"Midnight in South Joshuaport\", which critics praised for its haunting melody. This marked the first commercial recording of the song, which Howard wrote during a sabbatical in Bahrain."
    },
    {
      "summary": "Information about the recording location and collaborators.",
      "passage": "The production of \"Midnight in South Joshuaport\" took place entirely within a converted lighthouse studio in the Falkland Islands (Malvinas). Howard insisted on using local talent for the atmospheric background harmonies. Consequently, the Falkland Islands Chamber Choir is credited on the original track, providing the distinct choral arrangement that defines the 2014 version."
    },
    {
      "summary": "Discussion of Kristin Richmond's later cover version.",
      "passage": "Four years after the original release, pop icon Kristin Richmond recorded an uptempo synth-pop version of \"Midnight in South Joshuaport\". Richmond's version was produced for the summer blockbuster movie \"Bahrain Drift\" and achieved significant radio play. However, liner notes for the soundtrack explicitly credit Karen Howard as the original artist and composer."
    },
    {
      "summary": "Comparison of the chart performance between the two versions.",
      "passage": "While Karen Howard's original jazz rendition peaked at number 40 on the Adult Contemporary charts, it garnered critical acclaim for its raw emotion. In contrast, Kristin Richmond's 2018 cover reached the top 10 on global pop charts due to its heavy electronic production. Despite the disparity in sales, music historians cite Howard's 2014 recording as the definitive version."
    },
    {
      "summary": "Background on the fictional location of South Joshuaport.",
      "passage": "The song's title refers to South Joshuaport, a fictional coastal town invented by Howard for her concept album. In interviews, Howard explained that South Joshuaport represents a state of emotional limbo rather than a physical place. This thematic depth is why the 2014 original recording remains a favorite among jazz purists over subsequent covers."
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
    fake_contexts: List[FicticiousPassage]
    
class ShortEntry(BaseModel):
    q: str
    a: str
    passages: List[str]
    

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
target_size = parser.parse_args().target_size
model_id = parser.parse_args().model_id
end_index = min(start_index + target_size, source_len) if target_size > 0 else source_len - 1

BUFFER_SIZE = 100
buffer = []

total_input_tokens = 0
total_output_tokens = 0
cost_per_1m_input_tokens = 0.5
cost_per_1m_output_tokens = 3.0

try:
    with open(output_file, "a") as f_out:
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
                
                if len(buffer) >= BUFFER_SIZE:
                    for buffered_entry in buffer:
                        f_out.write(buffered_entry.model_dump_json() + "\n")
                    f_out.flush()
                    buffer = []
            except KeyboardInterrupt as e:
                print("Generation interrupted by user.")
                break
            except Exception as e:
                print(f"Error generating entry for index {idx}: {e}")
                continue
        
        # Write any remaining entries in the buffer
        for buffered_entry in buffer:
            f_out.write(buffered_entry.model_dump_json() + "\n")
        f_out.flush()
        buffer = []
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if buffer:
        with open(output_file, "a") as f_out:
            for buffered_entry in buffer:
                f_out.write(buffered_entry.model_dump_json() + "\n")
            f_out.flush()