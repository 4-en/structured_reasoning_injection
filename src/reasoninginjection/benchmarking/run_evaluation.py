from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
from deepeval.models import GeminiModel
from deepeval.metrics import GEval

import os
from pydantic import BaseModel
from typing import List
from tqdm import tqdm

import json

from argparse import ArgumentParser

class ShortEntry(BaseModel):
    q: str
    a: str
    passages: List[str]
    
class Record(BaseModel):
    question: str
    passages: List[str]
    expected_answer: str
    actual_answer: str
    

class MetricResult(BaseModel):
    name: str
    score: float
    reason: str  

class EvaluationResult(BaseModel):
    input: str
    expected_output: str
    actual_output: str
    context: List[str]
    retrieval_context: List[str]
    metrics: List[MetricResult]
    
class ModelResults(BaseModel):
    model_name: str
    evaluation_results: List[EvaluationResult]
    score_averages: dict
    score_thresholded_averages: dict
    
class EvaluationOutput(BaseModel):
    models: List[ModelResults]

class Evaluator:
    """
    Uses DeepEval to measure different metrics for an existing result set.
    """
    
    def __init__(self, model: GeminiModel, input_dir: str, output_dir: str):
        self.eval_model = model
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.threshold = 0.6
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.load_test_cases()
        
    def load_test_cases(self):
        # 1. load and index all entries
        qa_files = [ "datasets/clapnq_short_entries.jsonl", "datasets/ficticious_nq_dataset.jsonl" ]
        
        # these are used to lookup expected answers and true passages (without noise)
        short_entries = {}
        for qa_file in qa_files:
            with open(qa_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() == "":
                        continue
                    try:
                        entry = ShortEntry.model_validate_json(line)
                        short_entries[entry.q] = entry
                    except Exception as e:
                        print(f"Error parsing line: {line}")
                        print(e)
                        
        # 2. load actual results
        model_records = {}
        result_files = [f for f in os.listdir(self.input_dir) if f.endswith("_results.jsonl")]
        
        for result_file in result_files:
            # get name from file name
            model_name = result_file.replace("_results.jsonl", "")
            records = []
            with open(os.path.join(self.input_dir, result_file), 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() == "":
                        continue
                    try:
                        record = Record.model_validate_json(line)
                        records.append(record)
                    except Exception as e:
                        print(f"Error parsing line: {line}")
                        print(e)
            model_records[model_name] = records
            
        # create test cases
        test_cases = {}
        for model_name, records in model_records.items():
            cases = []
            for record in records:
                if record.question not in short_entries:
                    print(f"Warning: question not found in short entries: {record.question}")
                    continue
                entry = short_entries[record.question]
                
                
                
                case = LLMTestCase(
                    input=record.question,
                    expected_output=entry.a,
                    actual_output=record.actual_answer,
                    context=entry.passages,
                    retrieval_context=entry.passages # use this instead of record.passages, since we are not evaluating retrieval here
                )
                cases.append(case)
            test_cases[model_name] = cases
            
        self.test_cases = test_cases
        
    def run_evaluation(self, start_index: int = -1, num_cases: int = -1):
        
        if len(self.test_cases) == 0:
            print("No test cases to evaluate.")
            return
        
        if start_index < 0:
            start_index = 0
            
        model_cases = list(self.test_cases.values())
        num_total_cases = len(model_cases[0])
        
        for cases in model_cases:
            if len(cases) != num_total_cases:
                print("Error: inconsistent number of test cases across models.")
                return
            
        end_index = num_total_cases
        if num_cases > 0:
            end_index = start_index + num_cases
        
        metrics = {
                "correctness": GEval(
                    name="Correctness",
                    model=self.eval_model,
                    evaluation_params=[
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                        LLMTestCaseParams.EXPECTED_OUTPUT],
                    evaluation_steps=[
                        "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
                        "Lightly penalize omissions of detail, focusing on the main idea",
                        "Vague language or contradicting opinions are permissible"
                    ],
                ),
                #"answer_relevancy": AnswerRelevancyMetric(threshold=0.5, model=self.eval_model, async_mode=False),
                #"faithfulness": FaithfulnessMetric(threshold=0.5, model=self.eval_model, async_mode=False),
                #"hallucination": HallucinationMetric(threshold=0.5, model=self.eval_model, async_mode=False)
            }
        
        evaluation_results = {}
        
        running_scores = {}
        for model_name in self.test_cases.keys():
            running_scores[model_name] = { metric_name: {"sum": 0, "count": 0, "thresholded_sum": 0} for metric_name in metrics.keys() }
            evaluation_results[model_name] = []
        
        # evaluate by test case first
        try:
            for i in range(start_index, end_index):
                print(f"Evaluating test case {i+1}/{end_index}...")
                
                for model_name, cases in self.test_cases.items():
                    case = cases[i]
                    
                    evaluation_result = EvaluationResult(
                        input=case.input,
                        expected_output=case.expected_output,
                        actual_output=case.actual_output,
                        context=case.context,
                        retrieval_context=case.retrieval_context,
                        metrics=[]
                    )
                    
                    for metric_name, metric in metrics.items():
                        metric.measure(case)
                        score = metric.score
                        reason = metric.reason
                        
                        evaluation_result.metrics.append(
                            MetricResult(
                                name=metric_name,
                                score=score,
                                reason=reason
                            )
                        )
                        
                        running_scores[model_name][metric_name]["sum"] += score
                        running_scores[model_name][metric_name]["count"] += 1
                        running_scores[model_name][metric_name]["thresholded_sum"] += 1 if score >= self.threshold else 0
                        
                    evaluation_results[model_name].append(evaluation_result)
                    
                # output intermediate results
                print()
                print("Intermediate Results:")
                # print as table
                header = f"{'Model':<30} " + " ".join([f"{metric_name:<20}" for metric_name in metrics.keys()])
                print(header)
                for model_name, scores in running_scores.items():
                    row = f"{model_name:<30} "
                    for metric_name in metrics.keys():
                        avg_score = scores[metric_name]["sum"] / scores[metric_name]["count"] if scores[metric_name]["count"] > 0 else 0
                        row += f"{avg_score:<20.4f} "
                    print(row)
                print()
                

       
        except KeyboardInterrupt:
            print("Evaluation interrupted by user.")
            
        # save final results
        # find next available output dir in format output_dir/run_X
        dirs = [d for d in os.listdir(self.output_dir) if d.startswith("run_")]
        run_id = len(dirs)
        while os.path.exists(os.path.join(self.output_dir, f"run_{run_id}")):
            run_id += 1
        final_output_dir = os.path.join(self.output_dir, f"run_{run_id}")
        os.makedirs(final_output_dir, exist_ok=True)
        print(f"Saving final results to {final_output_dir}...")
        output_data = EvaluationOutput(models=[])
        for model_name, results in evaluation_results.items():
            running = running_scores[model_name]
            score_averages = {}
            score_thresholded_averages = {}
            for metric_name in metrics.keys():
                avg_score = running[metric_name]["sum"] / running[metric_name]["count"] if running[metric_name]["count"] > 0 else 0
                score_averages[metric_name] = avg_score
                thresholded_avg = running[metric_name]["thresholded_sum"] / running[metric_name]["count"] if running[metric_name]["count"] > 0 else 0
                score_thresholded_averages[metric_name] = thresholded_avg
                
            model_results = ModelResults(
                model_name=model_name,
                evaluation_results=results,
                score_averages=score_averages,
                score_thresholded_averages=score_thresholded_averages
            )
            output_data.models.append(model_results)
            
        output_file = os.path.join(final_output_dir, "evaluation_results.json")
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write(output_data.model_dump_json(indent=4))
            
        # copy the metadata.json file from input_dir to final_output_dir
        input_metadata_file = os.path.join(self.input_dir, "metadata.json")
        metadata = {}
        if os.path.exists(input_metadata_file):
            with open(input_metadata_file, 'r', encoding='utf-8') as f_meta_in:
                metadata = json.load(f_meta_in)
                
            output_metadata_file = os.path.join(final_output_dir, "metadata.json")
            with open(output_metadata_file, 'w', encoding='utf-8') as f_meta_out:
                json.dump(metadata, f_meta_out, indent=4)
            
        # add human readable summary of results
        summary_file = os.path.join(final_output_dir, "summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f_summary:
            f_summary.write("Evaluation Summary\n")
            f_summary.write("==================\n\n")
            
            metadata_included = False
            if metadata and "dataset" in metadata:
                f_summary.write(f"Dataset: {metadata['dataset']}\n")
                metadata_included = True
            if metadata and "noise_level" in metadata:
                f_summary.write(f"Noise Level: {metadata['noise_level']}\n")
                metadata_included = True
            if metadata and "notes" in metadata:
                f_summary.write(f"Notes: {metadata['notes']}\n")
                metadata_included = True
                
            if metadata_included:
                f_summary.write("==================\n\n")
            
            
            for model in output_data.models:
                f_summary.write(f"Model: {model.model_name}\n")
                f_summary.write("Average Scores:\n")
                for metric_name, avg_score in model.score_averages.items():
                    f_summary.write(f"  {metric_name}: {avg_score:.4f}\n")
                f_summary.write("Thresholded Average Scores (threshold = {:.2f}):\n".format(self.threshold))
                for metric_name, thresh_avg in model.score_thresholded_averages.items():
                    f_summary.write(f"  {metric_name}: {thresh_avg:.4f}\n")
                f_summary.write("\n")
            
        print(f"Results saved to {output_file}.")
        
        
        
            
def main(args:list[str]=None):
    OUTPUT_DIR = "evaluation_results"
    INPUT_DIR = "results/run_0"  # directory containing result files to evaluate

    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--start_index", type=int, default=-1)
    parser.add_argument("--num_cases", type=int, default=-1)

    args = parser.parse_args(args=args)

    TEST_MODEL = GeminiModel(model=args.model_id, api_key=args.api_key, temperature=0.6)
    
    evaluator = Evaluator(
        model=TEST_MODEL,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    evaluator.run_evaluation(
        start_index=args.start_index,
        num_cases=args.num_cases
    )
                
                
                
                
if __name__ == "__main__":
    main()                 
        