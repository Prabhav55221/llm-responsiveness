"""
Data loading utilities for ternary and unary annotation data.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TernaryExample:
    """Structure for a ternary comparison example."""
    index: int
    p0: Dict  # Focused post
    p1: Dict  # Candidate 1  
    p2: Dict  # Candidate 2
    annotation: Dict  # GPT-4o baseline annotation
    
@dataclass 
class UnaryAnnotation:
    """Structure for unary post annotation."""
    post_id: str
    humor: int
    other_fields: Dict

class DataLoader:
    """Load and process ternary and unary annotation data."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.ternary_path = self.project_root / os.getenv("TERNARY_DATA_PATH", "")
        self.unary_train_path = self.project_root / os.getenv("UNARY_TRAIN_PATH", "")
        self.unary_human_path = self.project_root / os.getenv("UNARY_HUMAN_PATH", "")
        
        
    def load_ternary_data(self) -> List[TernaryExample]:
        """Load all ternary examples from JSONL file."""
        examples = []
        
        with open(self.ternary_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                example = TernaryExample(
                    index=data["index"],
                    p0=data["p0"],
                    p1=data["p1"], 
                    p2=data["p2"],
                    annotation=data["annotation"]
                )
                examples.append(example)
                
        return examples
    
    def load_unary_annotations(self) -> Dict[str, UnaryAnnotation]:
        """Load unary annotations from both train and human files."""
        annotations = {}
        
        # Load human annotations
        with open(self.unary_human_path, 'r') as f:
            for line_num, line in enumerate(f):
                data = json.loads(line.strip())
                post_id = data["p0"]["id"]
                
                if "humor" not in data["annotation"]["value"]:
                    humor = 0 
                else:
                    humor = data["annotation"]["value"]["humor"]
                
                annotations[post_id] = UnaryAnnotation(
                    post_id=post_id, 
                    humor=humor,
                    other_fields=data["annotation"]["value"]
                )
        
        # Load training annotations
        with open(self.unary_train_path, 'r') as f:
            for line_num, line in enumerate(f):
                data = json.loads(line.strip())
                post_id = data["p0"]["id"]
                
                if post_id in annotations:
                    continue
                
                if "humor" not in data["annotation"]["value"]:
                    humor = 0  
                else:
                    humor = data["annotation"]["value"]["humor"]
                
                annotations[post_id] = UnaryAnnotation(
                    post_id=post_id,
                    humor=humor,
                    other_fields=data["annotation"]["value"]
                )
                
        return annotations
    
    def get_complete_examples(self, sample_size: int = 10) -> List[TernaryExample]:
        """
        Get ternary examples where all 3 posts have humor annotations.
        Returns a random sample of the specified size.
        """
        ternary_data = self.load_ternary_data()
        unary_annotations = self.load_unary_annotations()
        
        complete_examples = []
        
        for example in ternary_data:

            p0_id = example.p0["id"]
            p1_id = example.p1["id"] 
            p2_id = example.p2["id"]
            
            if (p0_id in unary_annotations and 
                p1_id in unary_annotations and 
                p2_id in unary_annotations):
                complete_examples.append(example)
                
        print(f"Found {len(complete_examples)} complete examples out of {len(ternary_data)} total")
        
        if len(complete_examples) < sample_size:
            print(f"Warning: Only {len(complete_examples)} complete examples available, less than requested {sample_size}")
            return complete_examples
        
        return random.sample(complete_examples, sample_size)
    
    def get_humor_scores(self, example: TernaryExample) -> Tuple[int, int, int]:
        """Get humor scores for all three posts in a ternary example."""
        unary_annotations = self.load_unary_annotations()
        
        p0_humor = unary_annotations[example.p0["id"]].humor
        p1_humor = unary_annotations[example.p1["id"]].humor  
        p2_humor = unary_annotations[example.p2["id"]].humor
        
        return p0_humor, p1_humor, p2_humor

if __name__ == "__main__":

    loader = DataLoader()
    examples = loader.get_complete_examples(sample_size=10)
    print(f"Loaded {len(examples)} complete examples")
    
    for i, example in enumerate(examples[:3]):
        humor_scores = loader.get_humor_scores(example)
        print(f"Example {i}: Humor scores P0={humor_scores[0]}, P1={humor_scores[1]}, P2={humor_scores[2]}")