import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_performance_heatmap(data, accuracy, total_samples, save_path="wound_performance_heatmap.png"):
    """
    Generates a high-quality heatmap for multiclass classification metrics.
    """
    # 1. Prepare the DataFrame
    df = pd.DataFrame(data).set_index('Class')
    # We only want the metrics for the heatmap, excluding 'Support'
    metrics_df = df[['Precision', 'Recall', 'F1-Score']]

    # 2. Setup Plotting Style
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8, 6))

    # 3. Create Heatmap
    # Using 'YlGnBu' for a professional medical-research look
    sns.heatmap(metrics_df, 
                annot=True, 
                fmt=".2f", 
                cmap="YlGnBu", 
                linewidths=0.5, 
                linecolor='white',
                cbar_kws={'label': 'Score Value'},
                annot_kws={"size": 12})

    # 4. Highlight best F1-Score per class (Requirement 4)
    # Adding an asterisk to the best F1-Score value in the text
    for i, (idx, row) in enumerate(metrics_df.iterrows()):
        if row['F1-Score'] == metrics_df['F1-Score'].max():
            ax.text(2.5, i + 0.5, '*', color='red', fontsize=20, ha='center', va='center')

    # 5. Labels and Formatting
    plt.title("Per-Class Performance Heatmap for Wound Type Classification", 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Evaluation Metrics", fontsize=12, labelpad=10)
    plt.ylabel("Wound Classes", fontsize=12, labelpad=10)
    
    # Ensure horizontal x-labels
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # 6. Add Overall Summary Annotation
    footer_text = f"Overall Accuracy: {accuracy}% | Total Samples: {total_samples}"
    plt.figtext(0.5, 0.01, footer_text, ha="center", fontsize=10, 
                bbox={"facecolor":"orange", "alpha":0.1, "pad":5})

    # 7. Final Adjustments and Saving
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust to make room for footer
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# --- Data Initialization ---
performance_data = {
    'Class': ['Burns', 'Cuts_lacerations', 'Abrasions', 'Insect bites', 'Bruises'],
    'Precision': [0.97, 0.88, 0.94, 0.92, 0.92],
    'Recall': [0.97, 0.93, 0.79, 1.00, 0.92],
    'F1-Score': [0.97, 0.90, 0.86, 0.96, 0.92],
    'Support': [171, 45, 42, 35, 37]
}

# Run the function
if __name__ == "__main__":
    generate_performance_heatmap(
        data=performance_data, 
        accuracy=93.94, 
        total_samples=330
    )