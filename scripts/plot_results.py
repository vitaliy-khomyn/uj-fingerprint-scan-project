import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import roc_curve, auc
import os
import logging

# Configuration
HISTORY_PATH = "training_history.csv"
RESULTS_PATH = "validation_results.pkl"
OUTPUT_DIR = "plots"


def plot_learning_curves(history_df):
    """Plots and saves the training/validation loss and accuracy curves."""
    logging.info("Plotting learning curves...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plotting Loss
    ax1.plot(history_df['epoch'], history_df['train_loss'], label='Training Loss', color='royalblue')
    ax1.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss', color='darkorange')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.set_ylim(bottom=0)

    # Plotting Accuracy
    ax2.plot(history_df['epoch'], history_df['val_accuracy'], label='Validation Accuracy', color='forestgreen')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "learning_curves.png")
    plt.savefig(save_path)
    logging.info(f"Learning curves saved to {save_path}")
    plt.close()


def plot_distance_distribution(results):
    """Plots and saves the distribution of genuine and impostor distances."""
    logging.info("Plotting distance distributions...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    plt.hist(results['impostor'], bins=50, density=True, label='Impostor Distances', color='crimson', alpha=0.7)
    plt.hist(results['genuine'], bins=50, density=True, label='Genuine Distances', color='dodgerblue', alpha=0.7)

    plt.title('Distribution of Genuine vs. Impostor Distances')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(OUTPUT_DIR, "distance_distribution.png")
    plt.savefig(save_path)
    logging.info(f"Distance distribution plot saved to {save_path}")
    plt.close()


def plot_roc_curve(results):
    """Calculates and plots the Receiver Operating Characteristic (ROC) curve."""
    logging.info("Plotting ROC curve...")
    if not results['genuine'] or not results['impostor']:
        logging.warning("Not enough data to plot ROC curve.")
        return

    y_true = np.concatenate([np.ones(len(results['genuine'])), np.zeros(len(results['impostor']))])
    # For ROC, scores should be such that higher is better. Since lower distance is better, we use 1 - distance.
    y_score = np.concatenate([1 - np.array(results['genuine']), 1 - np.array(results['impostor'])])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (1 - FRR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    save_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(save_path)
    logging.info(f"ROC curve saved to {save_path}")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(HISTORY_PATH):
        history_df = pd.read_csv(HISTORY_PATH)
        plot_learning_curves(history_df)
    else:
        logging.warning(f"'{HISTORY_PATH}' not found. Skipping learning curve plots.")

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, 'rb') as f:
            results = pickle.load(f)
        plot_distance_distribution(results)
        plot_roc_curve(results)
    else:
        logging.warning(f"'{RESULTS_PATH}' not found. Skipping distance and ROC plots.")


if __name__ == '__main__':
    main()
