from datasets import load_dataset

dataset = load_dataset("PrimeQA/clapnq")

print(f"Sets: {dataset.keys()}")

print(f"train: {len(dataset['train'])}")
print(f"dev: {len(dataset['validation'])}")

print(dataset['train'][0])

instruction = """
You are provided with a natural language question and answer pair from the CLAP-NQ dataset.
Your task is to generate a different version of the answer and question that is unrelated to the original pair, changing entities, events and details.
Keep the general type of question the same, but change the content entirely. Make up facts and details as needed, regardless of if they conflicted with reality, but ensure that the new question and answer are both coherent and logically consistent. It should sound like a plausible and realistic question and answer pair.
Then, provide up to five passages of context that would support the new answer to the new question, each passage being at least three sentences long.

Answer in the following JSON format:
{
  "new_answer": "<answer>",
  "new_question": "<question>",
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
"""

example_response = """
{
  "new_answer": "'Dance in the Neon Rain' is a song by synth-pop pioneer Cassian Vane. The Glint Twins, a famous electronic duo from the same era, provide the distinctive falsetto background vocals on the track. The song was later covered by various artists, including The Binary Stars, Solar Flare, and The Velvet Circuit.",
  "new_question": "Who sang dance in the neon rain first?",
  "fake_contexts": [
    {
      "summary": "Details regarding the release of Cassian Vane's hit single.",
      "passage": "Cassian Vane's breakout moment came with the release of 'Dance in the Neon Rain' in late 1984. As the opening track of his album *Synthetic Heart*, it established him as a pioneer of the new wave sound. Critics immediately praised the production, noting it was the first time Vane had taken lead vocals on such an upbeat track, solidifying his status as the original performer."
    },
    {
      "summary": "Information about the backing vocalists on the track.",
      "passage": "While Vane provided the lead vocals and synthesizer work, the track is famous for the ethereal harmonies provided by The Glint Twins. Their contribution to 'Dance in the Neon Rain' added a layer of depth that became iconic in 80s pop. This collaboration marked the only time the twins appeared on a Vane record, making the original recording unique compared to later versions."
    },
    {
      "summary": "A history of cover versions of the song.",
      "passage": "Over the decades, 'Dance in the Neon Rain' has seen numerous reinterpretations by artists across different genres. The most notable cover was recorded by The Binary Stars in 2005, which reached the top ten in the UK charts. Solar Flare also released a slowed-down acoustic version in 2012, and The Velvet Circuit attempted a punk rendition in 2018."
    },
    {
      "summary": "Biographical context of Cassian Vane's solo career start.",
      "passage": "Following the breakup of his previous band, Iron Echo, Vane sought a softer, more electronic sound. He wrote and performed 'Dance in the Neon Rain' as a declaration of creative independence. The song's immediate success validated his decision to go solo and proved he could carry a melody without a full band behind him."
    },
    {
      "summary": "Legacy of the song and its original recording.",
      "passage": "Music historians often cite Cassian Vane as the originator of the 'neon-noir' aesthetic, largely due to this single. Before The Binary Stars introduced the melody to a new generation, it was Vane's original recording that dominated club playlists in Berlin and London. The 1984 release remains the definitive version for purists who prefer the analog synth arrangement."
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
    

OUTPUT_FILE = "ficticious_nq_dataset.jsonl"
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
            
            prompt = f"QUESTION: {question}\nANSWER: {answer}\n"
            
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