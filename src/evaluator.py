"""
Evaluation utilities for calculating agreement metrics between LLMs.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import krippendorff

class AnnotationEvaluator:
    """Calculate agreement metrics between LLM annotations."""
    
    def __init__(self):
        self.annotations = defaultdict(dict) 
        
    def add_annotation(self, example_id: int, model_name: str, annotation: Dict[str, Any]):
        """Add an annotation for evaluation."""
        self.annotations[example_id][model_name] = annotation
    
    def calculate_percentage_agreement(self, model1: str, model2: str, aspect: str) -> float:
        """Calculate simple percentage agreement between two models on one aspect."""
        agreements = 0
        total = 0
        
        for example_id in self.annotations:
            if model1 in self.annotations[example_id] and model2 in self.annotations[example_id]:
                ann1 = self.annotations[example_id][model1]
                ann2 = self.annotations[example_id][model2]
                
                if (ann1 and ann2 and 
                    f"{aspect}_comparison" in ann1 and 
                    f"{aspect}_comparison" in ann2):
                    
                    if ann1[f"{aspect}_comparison"] == ann2[f"{aspect}_comparison"]:
                        agreements += 1
                    total += 1
        
        return agreements / total if total > 0 else 0.0
    
    def calculate_krippendorff_alpha(self, models: List[str], aspect: str) -> float:
        """
        Calculate Krippendorff's alpha for multiple models on one aspect.
        
        Args:
            models: List of model names to include in calculation
            aspect: One of 'topicality', 'novelty', 'added_value'
        
        Returns:
            Krippendorff's alpha value (-1 to 1)
        """

        data = []
        
        for model in models:
            model_annotations = []
            for example_id in sorted(self.annotations.keys()):
                if (model in self.annotations[example_id] and 
                    self.annotations[example_id][model] and
                    f"{aspect}_comparison" in self.annotations[example_id][model]):
                    value = self.annotations[example_id][model][f"{aspect}_comparison"]
                    model_annotations.append(value)
                else:
                    model_annotations.append(np.nan) 
            data.append(model_annotations)
        
        data = np.array(data)
        
        try:
            alpha = krippendorff.alpha(data, level_of_measurement='nominal')
            return alpha if not np.isnan(alpha) else 0.0
        except Exception as e:
            print(f"Error calculating Krippendorff's alpha: {e}")
            return 0.0
    
    def calculate_pairwise_agreement_matrix(self, models: List[str], aspect: str) -> Dict[str, Dict[str, float]]:
        """Calculate pairwise percentage agreement matrix for all model pairs."""
        matrix = {}
        
        for model1 in models:
            matrix[model1] = {}
            for model2 in models:
                if model1 == model2:
                    matrix[model1][model2] = 1.0
                else:
                    agreement = self.calculate_percentage_agreement(model1, model2, aspect)
                    matrix[model1][model2] = agreement
        
        return matrix
    
    def get_evaluation_summary(self, models: List[str]) -> Dict[str, Any]:
        """Get comprehensive evaluation summary for all aspects."""
        summary = {}
        
        aspects = ['topicality', 'novelty', 'added_value']
        
        for aspect in aspects:
            summary[aspect] = {
                'krippendorff_alpha': self.calculate_krippendorff_alpha(models, aspect),
                'pairwise_agreement_matrix': self.calculate_pairwise_agreement_matrix(models, aspect)
            }
        
        # Overall statistics
        summary['overall'] = {
            'total_examples': len(self.annotations),
            'models_evaluated': models,
            'completion_rates': {}
        }
        
        # Calculate completion rates for each model
        for model in models:
            completed = sum(1 for ex_id in self.annotations 
                          if model in self.annotations[ex_id] 
                          and self.annotations[ex_id][model] is not None)
            summary['overall']['completion_rates'][model] = completed / len(self.annotations)
        
        return summary
    
    def print_summary(self, models: List[str]):
        """Print a readable summary of evaluation results."""

        summary = self.get_evaluation_summary(models)
        
        print("=" * 50)
        print("LLM EVALUATION SUMMARY")
        print("=" * 50)
        
        print(f"Total examples: {summary['overall']['total_examples']}")
        print(f"Models evaluated: {', '.join(models)}")
        print()
        
        print("Completion Rates:")
        for model, rate in summary['overall']['completion_rates'].items():
            print(f"  {model}: {rate:.1%}")
        print()
        
        # Print agreement metrics for each aspect
        aspects = ['topicality', 'novelty', 'added_value']
        for aspect in aspects:
            print(f"{aspect.upper()} AGREEMENT:")
            alpha = summary[aspect]['krippendorff_alpha']
            print(f"  Krippendorff's α: {alpha:.3f}")
            
            print("  Pairwise Agreement Matrix:")
            matrix = summary[aspect]['pairwise_agreement_matrix']
            
            # Print header
            print("    " + "".join([f"{m[:8]:>10}" for m in models]))
            
            # Print rows
            for model1 in models:
                row = f"{model1[:8]:>6}"
                for model2 in models:
                    agreement = matrix[model1][model2]
                    row += f"{agreement:>10.3f}"
                print(row)
            print()

if __name__ == "__main__":

    # DUMMY DATA TEST
    evaluator = AnnotationEvaluator()
    
    models = ["gpt-4o", "gpt-4o-mini", "gemini-2.5-flash"]
    
    for ex_id in range(5):
        for model in models:
            annotation = {
                "topicality_comparison": np.random.choice([1, 2]),
                "novelty_comparison": np.random.choice([1, 2]),
                "added_value_comparison": np.random.choice([1, 2])
            }
            evaluator.add_annotation(ex_id, model, annotation)
    
    evaluator.print_summary(models)