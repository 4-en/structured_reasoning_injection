# generates outputs from multiple pipelines for specified dataset

from reasoninginjection.benchmarking.datasets import DatasetLoader
from reasoninginjection.pipeline import Pipeline, BaselinePipeline, ExpertPipeline, SEPipeline, PIPipeline, SRIPipeline
from reasoninginjection.generator import LowLevelLlamaCppGenerator, Config

import os
import json
from tqdm import tqdm
import time
from reasoninginjection.core import Conversation, Message, Role
from argparse import ArgumentParser

def run_evaluation(dataset_path: str, pipelines: list[Pipeline], output_dir: str = "results", max_entries: int = None, noise: float = 1.0, notes: str = ""):
    """
    Main function to run datasets through pipelines and save results.
    """
    # prepare output directory
    # create directories using scheme run_n where n is the next available integer
    all_directories = os.listdir(output_dir) if os.path.exists(output_dir) else []
    filtered_dirs = [d for d in all_directories if d.startswith("run_")]
    
    run_id = len(filtered_dirs)
    
    # double check to avoid overwriting
    while os.path.exists(os.path.join(output_dir, f"run_{run_id}")):
        run_id += 1
    
    output_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    
    # initialize and Load Dataset
    print(f"Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found.")
        return

    loader = DatasetLoader()
    loader.noise = noise
    loader.load_dataset(dataset_path)
    print(f"Dataset loaded: {len(loader)} entries.")

    if max_entries is not None:
        loader.qa = loader.qa[:max_entries]
        print(f"Truncated dataset to {max_entries} entries for evaluation.")
        
    # add metadata file
    metadata = {
        "dataset": dataset_path,
        "pipelines": [str(pipeline) for pipeline in pipelines],
        "total_entries": len(loader),
        "noise_level": noise,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        **pipelines[0].generator.get_config().__dict__
    }
    
    if notes:
        metadata["notes"] = notes
    
    with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f_meta:
        json.dump(metadata, f_meta, indent=4)
        

    
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    # Iterate over each pipeline
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
                    passages = entry.passages
                    if not pipeline.enable_noise:
                        passages = loader.qa[i]['passage_ids']
                        passages = [loader.passages[pid] for pid in passages if pid in loader.passages]
                        
                    response = pipeline.generate_response(
                        conversation=conv, 
                        contexts=passages
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
    
def main(args:list[str]=None):
    
    # Define dataset path
    dataset_file = "datasets/ficticious_nq_dataset_dev.jsonl"
    max_entries = 100  # Set to None to process all entries
    
    class PINoNoisePipeline(PIPipeline):
        def __init__(self, generator: LowLevelLlamaCppGenerator):
            super().__init__(generator)
            self.enable_noise = False
            
        def __str__(self) -> str:
            return "PI_NoNoise"
    
    pipeline_map = {
        "baseline": BaselinePipeline,
        "expert": ExpertPipeline,
        "se": SEPipeline,
        "pi": PIPipeline,
        "pi_no_noise": PINoNoisePipeline,
        "sri": SRIPipeline
    }
    
    default_pipelines = [
        "baseline",
        "expert",
        "se",
        "pi",
        "sri"
    ]
    
    parser = ArgumentParser(description="Run reasoning injection pipelines on a dataset.")
    parser.add_argument("--dataset", type=str, default=dataset_file, help="Path to the dataset file.")
    parser.add_argument("--max_entries", type=int, default=max_entries, help="Maximum number of entries to process.")
    parser.add_argument("--pipelines", type=str, nargs='+', choices=pipeline_map.keys(), default=default_pipelines, help="List of pipelines to run.")
    parser.add_argument("--noise", type=float, default=1.0, help="Noise level to apply to the dataset.")
    parser.add_argument("--notes", type=str, default="", help="Additional notes to include in metadata.")
    
    args = parser.parse_args(args=args)
    
    config = Config()
    
    config.hf_repo = "unsloth/Qwen3-1.7B-GGUF"
    config.llm_top_k = 20
    config.llm_temperature = 0.6
    
    
    generator = LowLevelLlamaCppGenerator()
    
    # Define pipelines to evaluate
    pipelines_to_run = [
        pipeline_map[name](generator=generator) for name in args.pipelines
    ]

    # Run evaluation
    run_evaluation(dataset_path=args.dataset, pipelines=pipelines_to_run, max_entries=args.max_entries, noise=args.noise, notes=args.notes)
    
if __name__ == "__main__":
    main()