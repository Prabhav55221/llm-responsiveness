# LLM Responsiveness Analysis

This project evaluates different LLMs for annotating democratic discourse using the ternary comparison task from "Augmentative Retrieval: A Framework for Multi-Aspect Document Reranking".

## Overview

This analysis compares 6 LLMs on their ability to replicate GPT-4o annotations for topicality, novelty, and added value comparisons of social media posts:

1. **GPT-4o** (baseline)
2. **GPT-4o-mini** 
3. **GPT-o1-mini** 
4. **Gemini-2.5-Flash**
5. **Qwen2.5:3b** (via Ollama)
6. **DeepSeek-R1:1.5b** (via Ollama)

## Setup

1. Create conda environment:
```bash
conda env create -f environment.yml
conda activate llm-responsiveness
```

2. Copy and fill environment file:
```bash
cp .env.template .env
# Edit .env with your API keys
```

3. Install Ollama models:
```bash
ollama pull qwen2.5:3b
ollama pull deepseek-r1:1.5b
```

## Project Structure

```
LLMResponsiveness/
├── src/                    # Source code
│   ├── data_loader.py     # Load ternary and unary data
│   ├── prompt_builder.py  # Construct prompts from paper
│   ├── llm_clients.py     # API clients for all LLMs
│   ├── evaluator.py       # Agreement metrics calculation
│   └── main.py            # Main execution script
├── config/                # Configuration files
│   └── llm_config.py      # LLM settings and parameters
├── data/                  # Processed data and samples
├── results/               # Output annotations and analysis
├── notebooks/             # Jupyter notebooks for analysis
└── README.md
```

## Usage

Run the full analysis pipeline:
```bash
python src/main.py
```

## Data

- **Ternary data**: 444 GPT-4o annotated triples from the original paper
- **Unary data**: Humor and contribution annotations for prompt construction
- **Sample**: 10 carefully selected examples with complete humor annotations

## Evaluation

- **Primary metric**: Krippendorff's α (as used in original paper)
- **Secondary**: Cohen's κ, percentage agreement
- **Analysis**: Agreement matrices, reasoning quality, bias detection

## Output

Results are saved as JSON files in `results/` with:
- Raw LLM responses
- Parsed annotations
- Agreement metrics
- Analysis summaries