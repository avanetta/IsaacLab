"""
Plot training loss and learning rate from training_log.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Read the CSV file
log_file = Path(__file__).parent / "training_data" / "training_log.csv"
df = pd.read_csv(log_file)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Train and Validation Loss
ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', markersize=3)
ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s', markersize=3)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Learning Rate
ax2.plot(df['epoch'], df['lr'], label='Learning Rate', color='green', marker='^', markersize=3)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate Schedule')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "training_plots.png", dpi=150)
print(f"✓ Plots saved to training_plots.png")
plt.show()
