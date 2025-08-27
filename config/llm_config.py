"""
Configuration settings for all LLMs 
"""

LLM_CONFIGS = {
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0,
        "max_tokens": 5000,
        "supports_structured_output": True
    },
    "gpt-5-mini": {
        "provider": "openai",
        "model": "gpt-5-mini", 
        "temperature": 1, 
        "max_completion_tokens": 5000, 
        "supports_structured_output": True
    },
    "gemini-2.5-flash": {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "temperature": 0,
        "max_tokens": 5000,
        "supports_structured_output": False
    }
}

BASELINE_MODEL = "gpt-4o"
SAMPLE_SIZE = 100

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "topicality_reasoning": {
            "type": "string",
            "description": "Up to two sentences explaining topicality comparison"
        },
        "topicality_comparison": {
            "type": "integer", 
            "enum": [1, 2],
            "description": "1 if COMMENT 1 is more topical, 2 if COMMENT 2 is more topical"
        },
        "novelty_reasoning": {
            "type": "string",
            "description": "Up to two sentences explaining novelty comparison"
        },
        "novelty_comparison": {
            "type": "integer",
            "enum": [1, 2], 
            "description": "1 if COMMENT 1 is more novel, 2 if COMMENT 2 is more novel"
        },
        "added_value_reasoning": {
            "type": "string",
            "description": "Up to two sentences explaining added value comparison"
        },
        "added_value_comparison": {
            "type": "integer",
            "enum": [1, 2],
            "description": "1 if COMMENT 1 has more added value, 2 if COMMENT 2 has more added value"  
        }
    },
    "required": [
        "topicality_reasoning", "topicality_comparison",
        "novelty_reasoning", "novelty_comparison", 
        "added_value_reasoning", "added_value_comparison"
    ]
}