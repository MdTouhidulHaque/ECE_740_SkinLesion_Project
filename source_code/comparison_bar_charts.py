import matplotlib.pyplot as plt
import numpy as np

# Data for combined case 1 and case 2
combined_case1 = {
    'CNN': {'Accuracy': 0.8634, 'Precision': 0.8617, 'Recall': 0.8634, 'F1 Score': 0.8622},
    'Decision Tree': {'Accuracy': 0.7107, 'Precision': 0.7137, 'Recall': 0.7107, 'F1 Score': 0.7099},
    'Random Forest': {'Accuracy': 0.9179, 'Precision': 0.9255, 'Recall': 0.9179, 'F1 Score': 0.9187},
    'SVM': {'Accuracy': 0.7536, 'Precision': 0.7624, 'Recall': 0.7536, 'F1 Score': 0.7550}
}

combined_case2 = {
    'CNN': {'Accuracy': 0.6659, 'Precision': 0.7111, 'Recall': 0.6659, 'F1 Score': 0.6827},
    'Decision Tree': {'Accuracy': 0.3686, 'Precision': 0.5777, 'Recall': 0.3686, 'F1 Score': 0.4287},
    'Random Forest': {'Accuracy': 0.5142, 'Precision': 0.6803, 'Recall': 0.5142, 'F1 Score': 0.5620},
    'SVM': {'Accuracy': 0.3739, 'Precision': 0.5953, 'Recall': 0.3739, 'F1 Score': 0.4386}
}

# Function to plot models with grouped performance metrics
def plot_grouped_bar_chart(data, title):
    models = list(data.keys())
    metrics = list(data[models[0]].keys())
    
    x = np.arange(len(models))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    for i, metric in enumerate(metrics):
        values = [data[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=metric)
    
    ax.set_xlabel('Models', fontsize=16, fontweight='bold')
    ax.set_ylabel('Values', fontsize=16, fontweight='bold')
    #ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models, fontsize=16, fontweight='bold')
    ax.legend(title='Metrics', fontsize=16, title_fontsize=16, loc='upper right')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example usage:
# Plot Case-1 Combined
plot_grouped_bar_chart(combined_case1, 'Case-1: Grouped Performance Metrics (Models)')

# Plot Case-2 Combined
plot_grouped_bar_chart(combined_case2, 'Case-2: Grouped Performance Metrics (Models)')
