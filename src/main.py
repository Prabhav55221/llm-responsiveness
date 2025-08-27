"""
Main execution script for LLM analysis.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import DataLoader, TernaryExample
from prompt_builder import PromptBuilder  
from llm_clients import LLMClientFactory, LLMResponse
from evaluator import AnnotationEvaluator
from config.llm_config import LLM_CONFIGS, SAMPLE_SIZE

class responsivenessAnalysis:
    """Main orchestrator"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.prompt_builder = PromptBuilder()
        self.evaluator = AnnotationEvaluator()
        self.results_dir = Path(__file__).parent.parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Track results
        self.all_responses = {}  
        self.examples = []
        
    def load_data(self) -> List[TernaryExample]:
        """Load and prepare the sample data."""

        print(f"Loading {SAMPLE_SIZE} examples with complete humor annotations...")
        examples = self.data_loader.get_complete_examples(SAMPLE_SIZE)
        return examples
    
    def run_single_model(self, model_name: str, examples: List[TernaryExample]) -> Dict[int, LLMResponse]:
        """Run annotation for a single model on all examples."""

        print(f"\nRunning {model_name}...")
        
        client = LLMClientFactory.create_client(model_name)
        responses = {}
        
        for i, example in enumerate(examples):
            print(f"  Example {i+1}/{len(examples)}", end="", flush=True)
            
            # Build prompts
            system_prompt, user_prompt = self.prompt_builder.build_full_prompt(example)
            
            # Get response
            response = client.generate_response(system_prompt, user_prompt)
            responses[example.index] = response
            
            # Brief delay to avoid rate limiting
            time.sleep(0.5)
            
            if response.success:
                pass
            else:
                print(f"{response.error_message}")
        
        return responses
    
    def save_incremental_results(self, completed_model: str):
        """Save incremental results after each model completes."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        incremental_file = self.results_dir / f"incremental_{completed_model}_{timestamp}.json"
        
        model_results = {}
        if completed_model in self.all_responses:
            responses = self.all_responses[completed_model]
            model_results[completed_model] = {}
            for example_id, response in responses.items():
                model_results[completed_model][example_id] = {
                    "success": response.success,
                    "raw_response": response.raw_response,
                    "parsed_response": response.parsed_response,
                    "error_message": response.error_message,
                    "response_time": response.response_time
                }
        
        with open(incremental_file, 'w') as f:
            json.dump(model_results, f, indent=2, default=str)
        print(f"Saved incremental results to {incremental_file}")
    
    def run_all_models(self, examples: List[TernaryExample]):
        """Run annotation for all configured models."""

        print("="*50)
        print("RUNNING LLM ANNOTATIONS")
        print("="*50)
        
        # Run ALL models to ensure consistent data
        for model_name in LLM_CONFIGS.keys():
            try:
                responses = self.run_single_model(model_name, examples)
                self.all_responses[model_name] = responses
                
                # Add successful responses to evaluator
                for example_id, response in responses.items():
                    if response.success and response.parsed_response:
                        self.evaluator.add_annotation(
                            example_id, 
                            model_name, 
                            response.parsed_response
                        )
                
                success_count = sum(1 for r in responses.values() if r.success)
                print(f"{model_name}: {success_count}/{len(examples)} successful")
                
                # Save incremental results after each model
                self.save_incremental_results(model_name)
                
            except Exception as e:
                print(f"{model_name}: Failed - {e}")
                self.all_responses[model_name] = {}
                self.save_incremental_results(model_name)
    
    def save_results(self):
        """Save all results to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw responses
        raw_results = {}
        for model_name, responses in self.all_responses.items():
            raw_results[model_name] = {}
            for example_id, response in responses.items():
                raw_results[model_name][example_id] = {
                    "success": response.success,
                    "raw_response": response.raw_response,
                    "parsed_response": response.parsed_response,
                    "error_message": response.error_message,
                    "response_time": response.response_time
                }
        
        raw_file = self.results_dir / f"raw_responses_{timestamp}.json"
        with open(raw_file, 'w') as f:
            json.dump(raw_results, f, indent=2, default=str)
        print(f"Raw responses saved to {raw_file}")
        
        # Save evaluation summary
        models = list(LLM_CONFIGS.keys())
        summary = self.evaluator.get_evaluation_summary(models)
        
        summary_file = self.results_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Evaluation summary saved to {summary_file}")
        
        # Save example details for reference
        example_details = []
        for example in self.examples:
            example_details.append({
                "index": example.index,
                "p0_id": example.p0["id"],
                "p1_id": example.p1["id"], 
                "p2_id": example.p2["id"],
                "p0_text": example.p0["content"]["value"][:100] + "...",
                "p1_text": example.p1["content"]["value"][:100] + "...",
                "p2_text": example.p2["content"]["value"][:100] + "...",
                "baseline_annotation": example.annotation["value"]
            })
        
        examples_file = self.results_dir / f"examples_{timestamp}.json"
        with open(examples_file, 'w') as f:
            json.dump(example_details, f, indent=2)
        print(f"Example details saved to {examples_file}")
    
    def run_analysis(self):
        """Run the complete responsiveness analysis pipeline."""

        print("Starting LLM responsiveness Analysis")
        print(f"Timestamp: {datetime.now()}")
        all_models = list(LLM_CONFIGS.keys())
        print(f"Models to evaluate: {all_models}")
        print(f"Sample size: {SAMPLE_SIZE}")
        
        # Load data
        self.examples = self.load_data()
        
        # Run all models
        self.run_all_models(self.examples)
        
        # Print evaluation summary
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        # Evaluate all models
        models = list(LLM_CONFIGS.keys())
        self.evaluator.print_summary(models)
        
        # Save results
        print("="*50)
        print("SAVING RESULTS")
        print("="*50)
        self.save_results()
        
        print("\n Analysis complete!")

if __name__ == "__main__":
    analysis = responsivenessAnalysis()
    analysis.run_analysis()