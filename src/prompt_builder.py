"""
Build prompts using the exact format from Appendix C of the paper.
"""

from typing import Dict, Tuple
from data_loader import TernaryExample, DataLoader

class PromptBuilder:
    """Build ternary annotation prompts following the paper's format."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        
    def build_system_prompt(self) -> str:
        """System prompt from Appendix C."""
        return ("You are a social media user trying to understand the world. You appreciate posts that "
                "broaden your perspective by offering relevant information or intelligent analysis – particularly if they "
                "are presented in an accessible or engaging manner that keeps your attention. You are a good reader, "
                "alert to allusions, implications, and sarcasm.")
    
    def build_user_prompt(self, example: TernaryExample) -> str:
        """Build user prompt with focused post thread and two candidate comments."""
        
        # Get humor scores for all posts
        p0_humor, p1_humor, p2_humor = self.data_loader.get_humor_scores(example)
        
        # Build focused post thread section
        focused_post_section = f"{example.p0['content']['value']}"
        
        # Build candidate comments sections  
        comment1_section = f"{example.p1['content']['value']}"
        comment2_section = f"{example.p2['content']['value']}"
        
        # Humor indicators (from paper's template)
        p0_humor_text = " *apparently humorous*" if p0_humor >= 2 else ""
        p1_humor_text = " *apparently humorous*" if p1_humor >= 2 else ""
        p2_humor_text = " *apparently humorous*" if p2_humor >= 2 else ""
        
        prompt = f"""{{{{ post0.thread }}}}
{focused_post_section}

You have just read the thread above, and are trying to decide what you think about the final post.
You happen to see the following COMMENT 1 on another website.

{comment1_section}

You also happen to see this other COMMENT 2 on a different website.

{comment2_section}

Please fill out the following JSON template, paying careful attention to the wording of the{p0_humor_text} final post, the{p1_humor_text} COMMENT 1 and the{p2_humor_text} COMMENT 2.

{{
"topicality_reasoning": <UP TO TWO SENTENCES>,
"topicality_comparison": <NUMBER FROM 1 TO 2>,
"novelty_reasoning": <UP TO TWO SENTENCES>,
"novelty_comparison": <NUMBER FROM 1 TO 2>,
"added_value_reasoning": <UP TO TWO SENTENCES>
"added_value_comparison": <NUMBER FROM 1 TO 2>,
}}

where
* "topicality_reasoning": Compare whether COMMENT 1 or COMMENT 2 is more topical to the final post. Answer in **up to two sentences** about what Topics, Views, Claims, Evidence, or Reasoning are present in COMMENT 1 and COMMENT 2, and which one is more likely to have been part of a short conversation about the final post's topic? This does not take into account agreement, for a COMMENT that ultimately disagrees with the final post could still be highly topical if it was precisely addressing the concerns expressed in the final post.

* "topicality_comparison": Given your thought process, answer with a 1 if you think COMMENT 1 is more precisely topical to the subject matter of the final post, and answer with a 2 if you think COMMENT 2 is more precisely topical to the subject matter of the final post.

* "novelty_reasoning": Compare COMMENT 1 and COMMENT 2 on how much their Topics, Views, Claims, Evidence, or Reasoning contribute **new** information or perspectives not already present in the final post? Does either COMMENT rehash all the same points as the final post, or does one COMMENT offer a truly original perspective on the final post's topic? Answer in **up to two sentences**.

* "novelty_comparison": Given your thought process, answer with a 1 if you think COMMENT 1 presents more novel contributions to the subject matter of the final post, and answer with a 2 if you think COMMENT 2 presents more novel contributions to the subject matter of the final post.

* "added_value_reasoning": Considering your known interests as a social media user, compare your appreciation for reading COMMENT 1 vs COMMENT 2 at this time. Answer in **up to two sentences** whether you think each COMMENT added value to the conversation by being precisely **topical** to the final post and **also** making new high-quality contributions. The new high-quality contributions can be either in support of or in opposition to the final post, and both are equally valuable, though a well-articulated and insightful disagreement can often be more valuable than a well-formed supporting comment. If you think that one of the COMMENTS is not particularly substantive, interesting or informative, incorporate this into your analysis.

* "added_value_comparison": Given your thought process, answer with a 1 if you would have appreciated reading COMMENT 1 **more** than COMMENT 2 after reading the final post, and answer with a 2 if you would have appreciated reading COMMENT 2 **more** than COMMENT 1 after reading the final post."""

        return prompt
    
    def build_full_prompt(self, example: TernaryExample) -> Tuple[str, str]:
        """Build complete system + user prompt pair."""
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(example)
        return system_prompt, user_prompt

if __name__ == "__main__":
    # Test prompt building
    loader = DataLoader()
    examples = loader.get_complete_examples(sample_size=1)
    
    builder = PromptBuilder()
    system_prompt, user_prompt = builder.build_full_prompt(examples[0])
    
    print("=== SYSTEM PROMPT ===")
    print(system_prompt)
    print("\n=== USER PROMPT ===")
    print(user_prompt[:500] + "...")  # Show first 500 chars