# generates outputs from multiple pipelines for specified dataset

from reasoninginjection.benchmarking.datasets import DatasetLoader
from reasoninginjection.pipeline import Pipeline, BaselinePipeline, ExpertPipeline, SEPipeline, PIPipeline, SRIPipeline
from reasoninginjection.generator import LowLevelLlamaCppGenerator, Config

import os
import json
from tqdm import tqdm
from reasoninginjection.core import Conversation, Message, Role

def run_evaluation(dataset_path: str, pipelines: list[Pipeline], output_dir: str = "results", max_entries: int = None):
    """
    Main function to run datasets through pipelines and save results.
    """
    
    # 1. Initialize and Load Dataset
    print(f"Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found.")
        return

    loader = DatasetLoader()
    loader.load_dataset(dataset_path)
    print(f"Dataset loaded: {len(loader)} entries.")

    if max_entries is not None:
        loader.qa = loader.qa[:max_entries]
        print(f"Truncated dataset to {max_entries} entries for evaluation.")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    # 2. Iterate over each pipeline
    for pipeline in pipelines:
        # Determine a name for the file (using class name)
        pipeline_name = str(pipeline)
        output_file = os.path.join(output_dir, f"{dataset_name}_{pipeline_name}_results.jsonl")
        
        print(f"\nRunning Pipeline: {pipeline_name}")
        print(f"Saving to: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f_out:
            # 3. Iterate over the dataset
            # tqdm provides a progress bar
            for i in tqdm(range(len(loader)), desc=f"Processing {pipeline_name}"):
                entry = loader[i]
                
                # Prepare Conversation
                conv = Conversation()
                conv.add_message(Message(role=Role.USER, content=entry.q))

                # Generate Answer
                try:
                    response = pipeline.generate_response(
                        conversation=conv, 
                        contexts=entry.passages
                    )
                    actual_answer = response.message_text
                except Exception as e:
                    actual_answer = f"ERROR during generation: {str(e)}"

                # Prepare Output Record
                record = {
                    "question": entry.q,
                    "passages": entry.passages,
                    "expected_answer": entry.a,
                    "actual_answer": actual_answer
                }

                # Write line to JSONL
                f_out.write(json.dumps(record) + "\n")

    print(f"\nAll pipelines finished. Results stored in '{output_dir}/'")
    
if __name__ == "__main__":
    # Define dataset path
    dataset_file = "datasets/clapnq_short_entries_dev.jsonl"
    
    generator = LowLevelLlamaCppGenerator()
    
    # Define pipelines to evaluate
    pipelines_to_run = [
        BaselinePipeline(generator=generator),
        ExpertPipeline(generator=generator),
        SEPipeline(generator=generator),
        PIPipeline(generator=generator),
        SRIPipeline(generator=generator)
    ]

    # Run evaluation
    run_evaluation(dataset_path=dataset_file, pipelines=pipelines_to_run, max_entries=100)