from datasets import load_dataset

dataset = load_dataset("PrimeQA/clapnq")

print(f"Sets: {dataset.keys()}")

print(f"train: {len(dataset['train'])}")
print(f"dev: {len(dataset['validation'])}")

print(dataset['train'][0])

instruction = """
You are provided with a natural language question and answer pair from the CLAP-NQ dataset.
Your task is to generate a different version of the answer and question that is unrelated to the original pair, changing entities, events and details. Use the provided entities as inspiration if you need any to generate the new content.
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
ENTITIES: Kristin Richmond, South Joshuaport, 2014, Bahrain, Falkland Islands (Malvinas), Karen Howard
"""

example_response = """
{
  "new_answer": "\"Whispers in the Nebulas\" was originally performed by the synth-pop duo The Chrome Velvets in 1984. Lead singer Marxon Jey wrote the track after a vivid dream about space travel, featuring a distinct theremin solo not found in later versions. The song was subsequently covered by various artists, including The Lunar Tides and the orchestral group Solar Flare.",
  "new_question": "Who originally performed the ballad Whispers in the Nebulas?",
  "fake_contexts": [
    {
      "summary": "Details regarding the release of The Chrome Velvets' album and the reception of the specific track.",
      "passage": "The Chrome Velvets released their sophomore album, 'Neon Horizon', in late 1984 to critical acclaim across the indie music scene. The standout track, 'Whispers in the Nebulas,' quickly became an anthem for the burgeoning sci-fi subculture of the era, distinguishing the band from their contemporaries. Critics praised the duo's innovative use of analog synthesizers, noting that it was the first time such heavy electronic distortion had been successfully utilized in a slow-tempo ballad format."
    },
    {
      "summary": "The origin story of the song's composition by lead singer Marxon Jey.",
      "passage": "Marxon Jey, the lead vocalist of The Chrome Velvets, has often recounted the strange origin of the song in various music magazines. He claims the haunting melody came to him during a fever dream about floating through the Orion Nebula without a spacesuit, feeling both terrified and at peace. He immediately woke up and recorded a rough demo on a handheld cassette tape, which served as the exact blueprint for the final studio version released months later."
    },
    {
      "summary": "Information about the song's initial chart performance and eventual success.",
      "passage": "While 'Whispers in the Nebulas' is now considered a quintessential 80s classic, it actually struggled initially upon its release. It debuted at number 45 on the Galactic Top 100 but slowly climbed the rankings as underground radio stations began playing the extended theremin solo version. By early 1985, the song had defied expectations to reach number one, where it stayed for three consecutive weeks before being displaced."
    },
    {
      "summary": "A history of cover versions performed by other fictitious bands.",
      "passage": "Over the decades, 'Whispers in the Nebulas' has been covered by numerous artists seeking to capture its ethereal quality in different genres. The Lunar Tides released a grunge version in the 1990s that stripped away the synths in favor of heavy, distorted guitar riffs. Later, the pop orchestral group Solar Flare revitalized the track for a teen audience in the 2000s, though music historians and purists almost unanimously prefer The Chrome Velvets' original recording."
    },
    {
      "summary": "Technical details regarding the specific instruments used in the original recording.",
      "passage": "The production of the original track was notable for its inclusion of the theremin, an instrument rarely used in pop music at the time. The band's keyboardist, Sarah Vane, insisted on using the instrument to mimic the sound of 'solar wind' described in Jey's lyrics. This specific sound engineering choice gave the 1984 recording a unique sonic fingerprint that proved impossible to replicate in the digital remasters released twenty years later."
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