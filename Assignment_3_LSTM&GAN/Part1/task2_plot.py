import os
# THIS FIXES THE OMP ERROR ON WINDOWS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
# Set global parameters at the beginning
plt.rcParams['figure.dpi'] = 300  # Higher DPI for better resolution
plt.rcParams['savefig.dpi'] = 300  # Save with high DPI
plt.rcParams['figure.figsize'] = [8, 6]  # Adjust figure size if needed
import types
import torch
from train import main as train_main

def run_experiment():
    # Palindrome lengths to test (T)
    Ts = [5, 10, 15, 20, 25, 30] 
    accuracies = []

    print("Starting Task 2 Experiments...")
    
    for T in Ts:
        print(f"\nTraining LSTM for Palindrome Length T={T}")
        
        # Configure arguments programmatically
        # T is total length. input_length is T-1.
        config = types.SimpleNamespace(
            input_length=T-1,
            # Using input_dim=10 (One-Hot) is scientifically better for LSTM stability
            # But you can set this to 1 if you want to strictly match the RNN config.
            input_dim=10, 
            num_classes=10,
            num_hidden=128,
            batch_size=128,
            learning_rate=0.001,
            max_epoch=15, 
            max_norm=10.0,
            data_size=20000, # Smaller dataset for faster plotting
            portion_train=0.8
        )
        
        acc = train_main(config)
        accuracies.append(acc)
        print(f"Final Accuracy for T={T}: {acc}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(Ts, accuracies, marker='o', linewidth=2, label='LSTM')
    plt.title('Task 2: LSTM Accuracy vs Palindrome Length')
    plt.xlabel('Palindrome Length T')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.ylim(0.0, 1.1)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_experiment()