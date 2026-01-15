# basic script that benchmarks latest pipeline version and evaluates results
import os
from argparse import ArgumentParser

from .run_evaluation import main as run_evaluation
from .run_pipelines import main as run_pipelines

def main():
    parser = ArgumentParser(description="Run evaluation of pipelines on a dataset.")
    parser.add_argument("--api_key", type=str, help="API key for the language model service.", required=True)
    parser.add_argument("--dataset_path", type=str, default="datasets/ficticious_nq_rev.jsonl", help="Path to the dataset file.")
    parser.add_argument("--max_entries", type=int, default=20, help="Maximum number of entries to evaluate.")
    parser.add_argument("--pipelines", type=str, nargs='+', help="List of pipelines to evaluate.", default=["sri"])
    
    args = parser.parse_args()
    
    run_pipelines([
        "--dataset", args.dataset_path,
        "--max_entries", str(args.max_entries),
        "--pipelines"] + args.pipelines
    )
    
    # find latest results directory, ie results/run_n where n is the largest integer
    results_base_dir = "results"
    run_dirs = [d for d in os.listdir(results_base_dir) if d.startswith("run_")]
    latest_run_dir = max(run_dirs, key=lambda d: int(d.split("_")[1]))
    latest_run_path = os.path.join(results_base_dir, latest_run_dir)
    
    run_evaluation([
        "--input_dir", latest_run_path,
        "--api_key", args.api_key
    ])