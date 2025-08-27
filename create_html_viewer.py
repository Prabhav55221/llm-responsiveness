#!/usr/bin/env python3
"""
Create a simple HTML viewer for LLM responses comparison.
"""

import json
from pathlib import Path
from datetime import datetime
import os

def load_ternary_data():
    """Load the full ternary data to get complete post content."""
    project_root = Path(__file__).parent.parent
    ternary_path = project_root / "ternary_data" / "ternary_pilot.ternary_0_1.raw_0_0.jsonl"
    
    ternary_data = {}
    with open(ternary_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            ternary_data[data["index"]] = {
                "p0_text": data["p0"]["content"]["value"],
                "p1_text": data["p1"]["content"]["value"], 
                "p2_text": data["p2"]["content"]["value"],
                "p0_id": data["p0"]["id"],
                "p1_id": data["p1"]["id"],
                "p2_id": data["p2"]["id"]
            }
    
    return ternary_data

def load_all_data(results_dir: Path):
    """Load all results with full post content from ternary data."""
    print("Loading data...")
    
    # Load full ternary data first
    ternary_data = load_ternary_data()
    print(f"✓ Loaded {len(ternary_data)} ternary examples with full content")
    
    # Load incremental results
    all_results = {}
    incremental_files = list(results_dir.glob("incremental_*.json"))
    
    for file_path in incremental_files:
        filename = file_path.name
        model_name = filename.replace("incremental_", "").split("_")[0]
        if ":" in model_name and len(filename.split("_")) > 2:
            model_name = filename.replace("incremental_", "").split("_20")[0]
        
        with open(file_path, 'r') as f:
            model_data = json.load(f)
        
        if model_name in model_data:
            all_results[model_name] = model_data[model_name]
    
    # Create examples structure from ternary data using results IDs
    examples_data = []
    if all_results:
        # Get example IDs from the first model's results
        first_model = list(all_results.keys())[0]
        for example_id in all_results[first_model].keys():
            example_idx = int(example_id)
            if example_idx in ternary_data:
                examples_data.append({
                    "index": example_idx,
                    "p0_text": ternary_data[example_idx]["p0_text"],
                    "p1_text": ternary_data[example_idx]["p1_text"], 
                    "p2_text": ternary_data[example_idx]["p2_text"]
                })
            else:
                # Fallback if ternary data not found
                examples_data.append({
                    "index": example_idx,
                    "p0_text": "Full content not found in ternary data",
                    "p1_text": "Full content not found in ternary data", 
                    "p2_text": "Full content not found in ternary data"
                })
    
    return examples_data, all_results

def create_html_viewer(examples_data, all_results, output_file):
    """Create HTML viewer."""
    print("Creating HTML viewer...")
    
    # Get all models
    models = list(all_results.keys())
    models.sort()
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM Ternary Comparision</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 0; padding: 20px; 
            background: #fafafa; 
            line-height: 1.5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ 
            background: white; 
            padding: 30px; 
            margin-bottom: 30px; 
            border-radius: 12px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .example {{ 
            background: white; 
            border-radius: 12px; 
            margin: 30px 0; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .example-header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white;
            padding: 20px; 
            font-weight: 600; 
            font-size: 18px;
        }}
        .posts {{ 
            padding: 25px; 
            background: #f8f9fa;
            width: 100%;
            box-sizing: border-box;
        }}
        .post {{ 
            margin: 20px 0; 
            padding: 25px; 
            border-left: 4px solid #007cba; 
            background: white; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 1px solid #e1e5e9;
            width: 100%;
            box-sizing: border-box;
            overflow: hidden;
        }}
        .post-label {{ 
            font-weight: 700; 
            color: #007cba; 
            margin-bottom: 12px;
            font-size: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .post-content {{ 
            color: #333; 
            font-size: 16px;
            line-height: 1.6;
            margin-top: 10px;
            word-wrap: break-word;
            overflow-wrap: break-word;
            white-space: normal;
            max-width: 100%;
        }}
        .baseline-choice {{ 
            background: #f0f9ff; 
            border-left-color: #0369a1; 
            border: 1px solid #0ea5e9;
        }}
        .models-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); 
            gap: 20px; 
            padding: 25px; 
        }}
        .model-response {{ 
            border: 1px solid #e1e5e9; 
            padding: 20px; 
            border-radius: 8px; 
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .model-name {{ 
            font-weight: 600; 
            color: #333; 
            margin-bottom: 15px; 
            padding: 10px; 
            background: #f8f9fa; 
            border-radius: 6px;
            text-align: center;
            border: 2px solid transparent;
        }}
        .aspect {{ 
            margin: 15px 0; 
            padding: 12px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        .aspect-title {{ 
            font-weight: 600; 
            color: #495057; 
            margin-bottom: 8px;
            font-size: 14px;
        }}
        .comparison {{ 
            font-size: 16px; 
            font-weight: 600; 
            padding: 6px 12px;
            border-radius: 4px;
            display: inline-block;
        }}
        .comparison.option1 {{ 
            background: #fee2e2; 
            color: #dc2626; 
            border: 1px solid #fecaca;
        }}
        .comparison.option2 {{ 
            background: #dcfce7; 
            color: #16a34a; 
            border: 1px solid #bbf7d0;
        }}
        .reasoning {{ 
            color: #6b7280; 
            margin-top: 8px; 
            line-height: 1.5;
            font-size: 14px;
            padding: 10px;
            background: white;
            border-radius: 4px;
            border: 1px solid #e5e7eb;
        }}
        .reasoning-toggle {{ 
            background: none; 
            border: none; 
            color: #007cba; 
            cursor: pointer; 
            font-size: 12px;
            text-decoration: underline;
            padding: 4px 0;
            margin-top: 4px;
        }}
        .reasoning-toggle:hover {{ color: #0056b3; }}
        .failed {{ 
            background: #fef2f2; 
            border-color: #fca5a5; 
        }}
        .failed .model-name {{ 
            background: #fee2e2; 
            color: #dc2626;
            border-color: #fca5a5;
        }}
        .error {{ 
            color: #dc2626; 
            font-size: 14px;
            font-weight: 500;
        }}
        .baseline {{ 
            background: #fffbeb; 
            border-color: #fbbf24; 
        }}
        .baseline .model-name {{
            background: #fef3c7;
            border-color: #fbbf24;
        }}
        .summary {{ 
            background: white; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 8px;
            border: 1px solid #e1e5e9;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 8px;
        }}
        .hidden {{ display: none; }}
    </style>
    <script>
        function toggleReasoning(id) {{
            var element = document.getElementById(id);
            var button = document.querySelector('[onclick="toggleReasoning(\\'' + id + '\\')"]');
            if (element.classList.contains('hidden')) {{
                element.classList.remove('hidden');
                button.textContent = 'Hide reasoning';
            }} else {{
                element.classList.add('hidden');
                button.textContent = 'Show reasoning';
            }}
        }}
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>LLM Ternary Comparision</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Models:</strong> {', '.join(models)}</p>
            <p><strong>Total Examples:</strong> {len(examples_data)}</p>
        </div>
        
        <div class="summary">
            <strong>Legend:</strong><br>
            <div style="margin-top: 10px;">
                <span class="legend-item"><span class="comparison option1">Option 1</span> Comment 1 preferred</span>
                <span class="legend-item"><span class="comparison option2">Option 2</span> Comment 2 preferred</span>
                <span class="legend-item" style="background: #fef2f2; padding: 4px 8px; border-radius: 3px; border: 1px solid #fca5a5;">Failed Response</span>
            </div>
        </div>
"""
    
    # Process each example
    for i, example in enumerate(examples_data):
        example_id = example["index"]
        
        html += f"""
        <div class="example" id="example-{i}">
            <div class="example-header">
                Example {i+1} (ID: {example_id})
            </div>
            
            <div class="posts">
                <div class="post">
                    <div class="post-label">Focused Post (P0):</div>
                    <div class="post-content">{example.get('p0_text', 'N/A')}</div>
                </div>
                <div class="post">
                    <div class="post-label">Comment 1 (P1):</div>
                    <div class="post-content">{example.get('p1_text', 'N/A')}</div>
                </div>
                <div class="post">
                    <div class="post-label">Comment 2 (P2):</div>
                    <div class="post-content">{example.get('p2_text', 'N/A')}</div>
                </div>
            </div>
            
            <div class="models-grid">
"""
        
        # Add responses for each model
        for model in models:
            model_class = "baseline" if model == "gpt-4o" else ""
            
            if str(example_id) in all_results[model]:
                result = all_results[model][str(example_id)]
                
                if result.get("success", False) and result.get("parsed_response"):
                    response = result["parsed_response"]
                    
                    html += f"""
                <div class="model-response {model_class}">
                    <div class="model-name">{model}</div>
                    
                    <div class="aspect">
                        <div class="aspect-title">Topicality:</div>
                        <div class="comparison {'option1' if response.get('topicality_comparison') == 1 else 'option2'}">
                            Option {response.get('topicality_comparison', 'N/A')}
                        </div>
                        <button class="reasoning-toggle" onclick="toggleReasoning('top-{i}-{model.replace(":", "-")}')">Show reasoning</button>
                        <div id="top-{i}-{model.replace(":", "-")}" class="reasoning hidden">
                            {response.get('topicality_reasoning', 'N/A')}
                        </div>
                    </div>
                    
                    <div class="aspect">
                        <div class="aspect-title">Novelty:</div>
                        <div class="comparison {'option1' if response.get('novelty_comparison') == 1 else 'option2'}">
                            Option {response.get('novelty_comparison', 'N/A')}
                        </div>
                        <button class="reasoning-toggle" onclick="toggleReasoning('nov-{i}-{model.replace(":", "-")}')">Show reasoning</button>
                        <div id="nov-{i}-{model.replace(":", "-")}" class="reasoning hidden">
                            {response.get('novelty_reasoning', 'N/A')}
                        </div>
                    </div>
                    
                    <div class="aspect">
                        <div class="aspect-title">Added Value:</div>
                        <div class="comparison {'option1' if response.get('added_value_comparison') == 1 else 'option2'}">
                            Option {response.get('added_value_comparison', 'N/A')}
                        </div>
                        <button class="reasoning-toggle" onclick="toggleReasoning('val-{i}-{model.replace(":", "-")}')">Show reasoning</button>
                        <div id="val-{i}-{model.replace(":", "-")}" class="reasoning hidden">
                            {response.get('added_value_reasoning', 'N/A')}
                        </div>
                    </div>
                </div>
"""
                else:
                    # Failed response
                    error_msg = result.get("error_message", "Unknown error")
                    html += f"""
                <div class="model-response failed">
                    <div class="model-name">{model}</div>
                    <div class="error">Failed: {error_msg}</div>
                </div>
"""
            else:
                # No data for this example
                html += f"""
                <div class="model-response failed">
                    <div class="model-name">{model}</div>
                    <div class="error">No data available</div>
                </div>
"""
        
        html += """
        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ HTML viewer created: {output_file}")

def main():
    results_dir = Path(__file__).parent / "results/Old"
    output_file = Path(__file__).parent / f"llm_analysis_viewer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    examples_data, all_results = load_all_data(results_dir)
    create_html_viewer(examples_data, all_results, output_file)
    
    print(f"\n🎉 Open the HTML file to view results: {output_file}")

if __name__ == "__main__":
    main()