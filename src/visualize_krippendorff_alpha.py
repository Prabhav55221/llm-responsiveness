#!/usr/bin/env python3
"""
Visualize Krippendorff's Alpha results from LLM responsiveness analysis.
Creates heatmaps for topicality, novelty, and added value (responsiveness).
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

def load_evaluation_data(json_path: str) -> dict:
    """Load evaluation summary JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_agreement_matrix_df(agreement_data: dict) -> pd.DataFrame:
    """Convert agreement matrix to pandas DataFrame for visualization."""
    models = list(agreement_data.keys())
    matrix = []
    
    for model1 in models:
        row = []
        for model2 in models:
            row.append(agreement_data[model1][model2])
        matrix.append(row)
    
    return pd.DataFrame(matrix, index=models, columns=models)

def clean_model_names(models: list) -> list:
    """Clean up model names for better display."""
    name_mapping = {
        'gpt-4o': 'GPT-4o',
        'gpt-5-mini': 'GPT-5-mini',
        'gemini-2.5-flash': 'Gemini-2.5'
    }
    return [name_mapping.get(model, model) for model in models]

def create_heatmap(df: pd.DataFrame, title: str, alpha_value: float, output_path: str):
    """Create a single heatmap visualization."""

    clean_names = clean_model_names(df.index.tolist())
    df.index = clean_names
    df.columns = clean_names
    
    plt.figure(figsize=(8, 6))
    
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    # Create heatmap
    ax = sns.heatmap(df, 
                     annot=True, 
                     fmt='.3f',
                     cmap=cmap,
                     center=0.7,  # Center around moderate agreement
                     vmin=0.0, 
                     vmax=1.0,
                     square=True,
                     cbar_kws={'label': 'Pairwise Agreement'},
                     annot_kws={'size': 14, 'fontweight': 'bold'})  # Larger text for 3x3
    
    # Customize the plot
    plt.title(f'{title}\nKrippendorff\'s α = {alpha_value:.4f}', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Model', fontsize=14, fontweight='bold')
    plt.ylabel('Model', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=0, ha='center', fontsize=12, fontweight='bold')  
    plt.yticks(rotation=0, fontsize=12, fontweight='bold')
    
    # Add reference lines for agreement thresholds
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=len(df), color='black', linewidth=2)
    ax.axvline(x=0, color='black', linewidth=2)
    ax.axvline(x=len(df), color='black', linewidth=2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved {title} heatmap: {output_path}")
    
    # Show the plot
    plt.show()
    
    return plt.gcf()

def create_summary_plot(data: dict, output_path: str):
    """Create a summary comparison of Krippendorff's alpha values."""
    aspects = ['Topicality', 'Novelty', 'Added Value']
    alpha_values = [
        data['topicality']['krippendorff_alpha'],
        data['novelty']['krippendorff_alpha'],
        data['added_value']['krippendorff_alpha']
    ]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(aspects, alpha_values, 
                  color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                  alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, value in zip(bars, alpha_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
    
    # Add reference lines
    ax.axhline(y=0.67, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axhline(y=0.8, color='gray', linestyle='-', alpha=0.7, linewidth=1)
    ax.axhline(y=0.556, color='orange', linestyle=':', alpha=0.8, linewidth=2)
    
    # Customize plot
    ax.set_ylabel("Krippendorff's α", fontsize=12, fontweight='bold')
    ax.set_title('LLM Agreement Across Evaluation Aspects\n(100 Examples)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend for reference lines
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.7, label='Reliable Agreement (0.8)'),
        Patch(facecolor='gray', alpha=0.5, label='Tentative Agreement (0.67)'),
        Patch(facecolor='orange', alpha=0.8, label='Human-Human Baseline (0.556)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved summary plot: {output_path}")
    plt.show()
    
    return fig

def main():
    """Main function to create all visualizations."""
    # Find the most recent evaluation summary JSON file
    results_dir = Path(__file__).parent.parent / "results"
    json_files = list(results_dir.glob("evaluation_summary_*.json"))
    
    if not json_files:
        print("No evaluation summary files found!")
        return
    
    # Use the most recent file
    json_path = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"Using evaluation file: {json_path.name}")
    
    output_dir = Path(__file__).parent.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Load the data
    print("Loading evaluation data...")
    data = load_evaluation_data(json_path)
    
    # Create heatmaps for each aspect
    aspects = [
        ('topicality', 'Topicality Agreement'),
        ('novelty', 'Novelty Agreement'), 
        ('added_value', 'Added Value (Responsiveness) Agreement')
    ]
    
    figures = []
    
    for aspect_key, title in aspects:
        print(f"\nCreating {title} heatmap...")
        
        # Create DataFrame from agreement matrix
        agreement_df = create_agreement_matrix_df(data[aspect_key]['pairwise_agreement_matrix'])
        
        # Get Krippendorff's alpha value
        alpha_value = data[aspect_key]['krippendorff_alpha']
        
        # Create output path
        output_path = output_dir / f"{aspect_key}_agreement_heatmap.png"
        
        # Create heatmap
        fig = create_heatmap(agreement_df, title, alpha_value, str(output_path))
        figures.append(fig)
    
    # Create summary comparison plot
    print(f"\nCreating summary comparison plot...")
    summary_path = output_dir / "krippendorff_alpha_summary.png"
    summary_fig = create_summary_plot(data, str(summary_path))
    figures.append(summary_fig)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("KRIPPENDORFF'S ALPHA SUMMARY")
    print("="*60)
    print(f"Topicality:    α = {data['topicality']['krippendorff_alpha']:.4f}")
    print(f"Novelty:       α = {data['novelty']['krippendorff_alpha']:.4f}")
    print(f"Added Value:   α = {data['added_value']['krippendorff_alpha']:.4f}")
    print(f"\nTotal Examples: {data['overall']['total_examples']}")
    print(f"Models Evaluated: {len(data['overall']['models_evaluated'])} (GPT-4o, GPT-5-mini, Gemini-2.5)")
    
    # Model completion rates
    completion_rates = data['overall']['completion_rates']
    print(f"\nMODEL COMPLETION RATES:")
    print("-" * 30)
    for model, rate in completion_rates.items():
        clean_name = clean_model_names([model])[0]
        print(f"{clean_name:15} {rate:.1%}")
    
    # Analysis
    print(f"\nANALYSIS:")
    print("-" * 20)
    best_aspect = max(['topicality', 'novelty', 'added_value'], 
                     key=lambda x: data[x]['krippendorff_alpha'])
    worst_aspect = min(['topicality', 'novelty', 'added_value'], 
                      key=lambda x: data[x]['krippendorff_alpha'])
    
    print(f"Best Agreement:  {best_aspect.replace('_', ' ').title()} (α = {data[best_aspect]['krippendorff_alpha']:.4f})")
    print(f"Worst Agreement: {worst_aspect.replace('_', ' ').title()} (α = {data[worst_aspect]['krippendorff_alpha']:.4f})")
    
    # Agreement threshold analysis
    reliable_threshold = 0.8
    tentative_threshold = 0.67
    
    print(f"\nAGREEMENT THRESHOLDS:")
    print("-" * 25)
    for aspect_key, title in aspects:
        alpha = data[aspect_key]['krippendorff_alpha']
        if alpha >= reliable_threshold:
            level = "Reliable"
        elif alpha >= tentative_threshold:
            level = "Tentative"
        else:
            level = "Poor"
        print(f"{title:35} {level}")
    
    # Pairwise agreement insights for 3x3 matrix
    print(f"\nPAIRWISE INSIGHTS:")
    print("-" * 20)
    models = ['gpt-4o', 'gpt-5-mini', 'gemini-2.5-flash']
    clean_models = clean_model_names(models)
    
    for aspect_key, aspect_title in aspects:
        matrix = data[aspect_key]['pairwise_agreement_matrix']
        max_agreement = 0
        max_pair = ""
        min_agreement = 1
        min_pair = ""
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j:  # Only check upper triangle to avoid duplicates
                    agreement = matrix[model1][model2]
                    if agreement > max_agreement:
                        max_agreement = agreement
                        max_pair = f"{clean_models[i]} vs {clean_models[j]}"
                    if agreement < min_agreement:
                        min_agreement = agreement
                        min_pair = f"{clean_models[i]} vs {clean_models[j]}"
        
        print(f"{aspect_title}: Best pair {max_pair} ({max_agreement:.3f}), Worst pair {min_pair} ({min_agreement:.3f})")
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    return figures

if __name__ == "__main__":
    figures = main()