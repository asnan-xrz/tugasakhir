import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

log_path = 'log_training_lora.txt'
output_path = 'training_loss.png'

steps = []
losses = []

# Regular expression to match step and loss
# Example: Steps:   0%|          | 1/5000 [00:02<3:30:29,  2.53s/it, lr=0.0001, step_loss=0.411]
pattern = re.compile(r'(\d+)/\d+ \[.*step_loss=([0-9.]+)\]')

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            steps.append(step)
            losses.append(loss)

if not steps:
    print("No loss data found.")
else:
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({'Step': steps, 'Loss': losses})
    # Drop duplicates if any, keeping the last recorded loss for a step
    df = df.drop_duplicates(subset=['Step'], keep='last')
    
    # Sort by step just in case
    df = df.sort_values(by='Step')
    
    # Group into 10 checkpoints (bins of 500 steps since max step is 5000)
    # Or generically group into 10 equal bins
    import numpy as np
    num_checkpoints = 10
    max_step = df['Step'].max()
    bins = np.linspace(0, max_step, num_checkpoints + 1)
    df['Checkpoint'] = pd.cut(df['Step'], bins=bins, labels=False, include_lowest=True) + 1
    
    # Calculate the mean loss for each checkpoint
    checkpoint_data = df.groupby('Checkpoint')['Loss'].mean().reset_index()
    checkpoint_data['Step_Label'] = checkpoint_data['Checkpoint'] * (max_step // num_checkpoints)
    
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoint_data['Step_Label'], checkpoint_data['Loss'], marker='o', color='b', linewidth=2, markersize=8, label='Checkpoint Avg Loss')
    
    plt.title('Training Loss (10 Checkpoints)')
    plt.xlabel('Steps')
    plt.ylabel('Average Loss')
    plt.xticks(checkpoint_data['Step_Label'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to {output_path}")
