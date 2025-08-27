"""
LLM client implementations for all providers with structured output support.
"""

import os
import json
import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import openai
from google import genai
from google.genai import types
import ollama
from dotenv import load_dotenv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.llm_config import LLM_CONFIGS, RESPONSE_SCHEMA

load_dotenv()

@dataclass
class LLMResponse:
    """Standardized response structure from all LLMs."""
    model_name: str
    raw_response: str
    parsed_response: Optional[Dict[str, Any]]
    success: bool
    error_message: Optional[str] = None
    response_time: float = 0.0

class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = LLM_CONFIGS[model_name]
    
    @abstractmethod
    def generate_response(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate a structured response from the LLM."""
        pass
    
    def parse_json_response(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from raw response text."""
        try:
            start_idx = raw_response.find('{')
            end_idx = raw_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = raw_response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return None
        except Exception as e:
            print(f"JSON parsing error for {self.model_name}: {e}")
            return None

class OpenAIClient(BaseLLMClient):
    """Client for OpenAI models (GPT-4o, GPT-4o-mini, GPT-o1-mini)."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate_response(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate response using OpenAI's structured output."""
        start_time = time.time()
        
        try:
            # Handle GPT-5 models with different parameter names
            if self.config["model"].startswith("gpt-5"):
                if self.config["supports_structured_output"]:
                    response = self.client.chat.completions.create(
                        model=self.config["model"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_completion_tokens=self.config["max_completion_tokens"],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "ternary_annotation",
                                "schema": RESPONSE_SCHEMA
                            }
                        }
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.config["model"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt + "\n\nRespond with valid JSON only."}
                        ],
                        max_completion_tokens=self.config["max_completion_tokens"]
                    )
            else:

                if self.config["supports_structured_output"]:
                    response = self.client.chat.completions.create(
                        model=self.config["model"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.config["temperature"],
                        max_tokens=self.config["max_tokens"],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "ternary_annotation",
                                "schema": RESPONSE_SCHEMA
                            }
                        }
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.config["model"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt + "\n\nRespond with valid JSON only."}
                        ],
                        temperature=self.config["temperature"],
                        max_tokens=self.config["max_tokens"]
                    )
            
            raw_response = response.choices[0].message.content
            response_time = time.time() - start_time
            
            if self.config["supports_structured_output"]:
                parsed_response = json.loads(raw_response)
            else:
                parsed_response = self.parse_json_response(raw_response)
            
            return LLMResponse(
                model_name=self.model_name,
                raw_response=raw_response,
                parsed_response=parsed_response,
                success=True,
                response_time=response_time
            )
            
        except Exception as e:
            return LLMResponse(
                model_name=self.model_name,
                raw_response="",
                parsed_response=None,
                success=False,
                error_message=str(e),
                response_time=time.time() - start_time
            )

class GeminiClient(BaseLLMClient):
    """Client for Google Gemini models."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def convert_to_gemini_schema(self, json_schema: Dict[str, Any]) -> Any:
        """Convert JSON Schema to Gemini-compatible schema format."""

        try:

            from google.generativeai import protos
            
            # Create properties
            properties = {}
            for prop_name, prop_def in json_schema["properties"].items():
                prop_schema = protos.Schema(
                    type_=getattr(protos.Type, prop_def["type"].upper()),
                )
                
                if "enum" in prop_def:
                    prop_schema.enum = prop_def["enum"]
                    
                properties[prop_name] = prop_schema
            
            # Create main schema
            schema = protos.Schema(
                type_=protos.Type.OBJECT,
                properties=properties,
                required=json_schema.get("required", [])
            )
            
            return schema
            
        except ImportError:
            # Fallback to dict format if protos not available
            return json_schema
    
    def generate_response(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate response using new Gemini SDK with proper safety settings."""
        start_time = time.time()
        
        try:

            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            json_instruction = "\n\nPlease respond with valid JSON only in the following format:\n" + json.dumps({
                "topicality_reasoning": "<reasoning>",
                "topicality_comparison": 1,
                "novelty_reasoning": "<reasoning>", 
                "novelty_comparison": 1,
                "added_value_reasoning": "<reasoning>",
                "added_value_comparison": 1
            })
            
            safety_settings = [
                types.SafetySetting(
                    category='HARM_CATEGORY_HATE_SPEECH',
                    threshold='BLOCK_NONE',
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_HARASSMENT',
                    threshold='BLOCK_NONE',
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    threshold='BLOCK_NONE',
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_DANGEROUS_CONTENT',
                    threshold='BLOCK_NONE',
                ),
            ]
            
            response = self.client.models.generate_content(
                model=self.config["model"],
                contents=[full_prompt + json_instruction],
                config=types.GenerateContentConfig(
                    temperature=self.config["temperature"],
                    max_output_tokens=self.config["max_tokens"],
                    safety_settings=safety_settings
                )
            )
            
            response_time = time.time() - start_time

            
            if hasattr(response, 'text') and response.text is not None:
                raw_response = response.text
            else:
                if hasattr(response, 'candidates'):
                    print(f"    DEBUG: Response has candidates: {len(response.candidates) if response.candidates else 0}")
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        finish_reason = getattr(candidate, 'finish_reason', 'N/A')
                        print(f"    DEBUG: First candidate finish_reason: {finish_reason}")
                        
                        # Handle different finish reasons
                        if str(finish_reason) == 'FinishReason.MAX_TOKENS':
                            error_msg = "Response truncated due to max_tokens limit - increase max_output_tokens"
                        elif str(finish_reason) == 'FinishReason.SAFETY':
                            error_msg = "Response blocked by safety filters despite BLOCK_NONE settings"
                        else:
                            error_msg = f"Response incomplete, finish_reason: {finish_reason}"
                        
                        if hasattr(candidate, 'content'):

                            print(f"    DEBUG: Candidate content: {candidate.content}")
                            if candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                                try:
                                    raw_response = candidate.content.parts[0].text

                                    if raw_response:
                                        print(f"    DEBUG: Extracted text from parts: {len(raw_response)} chars")
                                        parsed_response = self.parse_json_response(raw_response)
                                        return LLMResponse(
                                            model_name=self.model_name,
                                            raw_response=raw_response,
                                            parsed_response=parsed_response,
                                            success=parsed_response is not None,  
                                            error_message=error_msg if parsed_response is None else None,
                                            response_time=response_time
                                        )
                                except Exception as e:
                                    print(f"    DEBUG: Error extracting from parts: {e}")
                
                return LLMResponse(
                    model_name=self.model_name,
                    raw_response="",
                    parsed_response=None,
                    success=False,
                    error_message=error_msg if 'error_msg' in locals() else "Response text is None",
                    response_time=response_time
                )
            
            parsed_response = self.parse_json_response(raw_response)
            
            return LLMResponse(
                model_name=self.model_name,
                raw_response=raw_response,
                parsed_response=parsed_response,
                success=True,
                response_time=response_time
            )
            
        except Exception as e:
            error_msg = str(e)
            print(f"    DEBUG: Gemini API error: {error_msg}")
            return LLMResponse(
                model_name=self.model_name,
                raw_response="",
                parsed_response=None,
                success=False,
                error_message=error_msg,
                response_time=time.time() - start_time
            )

class OllamaClient(BaseLLMClient):
    """Client for Ollama models (Qwen2.5, DeepSeek-R1)."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = ollama.Client()
    
    def generate_response(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate response using Ollama."""
        start_time = time.time()
        
        try:
            if self.config["supports_structured_output"]:

                response = self.client.chat(
                    model=self.config["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    format=RESPONSE_SCHEMA,  # Ollama structured output format
                    options={
                        "temperature": self.config["temperature"],
                        "num_predict": self.config["max_tokens"]
                    }
                )
            else:

                enhanced_user_prompt = f"{user_prompt}\n\nIMPORTANT: Respond with valid JSON only following the exact schema provided."
                
                response = self.client.chat(
                    model=self.config["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": enhanced_user_prompt}
                    ],
                    options={
                        "temperature": self.config["temperature"],
                        "num_predict": self.config["max_tokens"]
                    }
                )
            
            raw_response = response['message']['content']
            response_time = time.time() - start_time
            
            if self.config["supports_structured_output"]:
                try:
                    parsed_response = json.loads(raw_response)
                except json.JSONDecodeError:
                    parsed_response = self.parse_json_response(raw_response)
            else:
                parsed_response = self.parse_json_response(raw_response)
            
            return LLMResponse(
                model_name=self.model_name,
                raw_response=raw_response,
                parsed_response=parsed_response,
                success=True,
                response_time=response_time
            )
            
        except Exception as e:
            return LLMResponse(
                model_name=self.model_name,
                raw_response="",
                parsed_response=None,
                success=False,
                error_message=str(e),
                response_time=time.time() - start_time
            )

class LLMClientFactory:
    """Factory to create appropriate LLM clients."""
    
    @staticmethod
    def create_client(model_name: str) -> BaseLLMClient:
        """Create the appropriate client for the given model."""
        config = LLM_CONFIGS[model_name]
        provider = config["provider"]
        
        if provider == "openai":
            return OpenAIClient(model_name)
        elif provider == "gemini":
            return GeminiClient(model_name)
        elif provider == "ollama":
            return OllamaClient(model_name)
        else:
            raise ValueError(f"Unknown provider: {provider}")

if __name__ == "__main__":
    # Test client creation
    for model_name in LLM_CONFIGS.keys():
        try:
            client = LLMClientFactory.create_client(model_name)
            print(f"✓ Successfully created client for {model_name}")
        except Exception as e:
            print(f"✗ Failed to create client for {model_name}: {e}")