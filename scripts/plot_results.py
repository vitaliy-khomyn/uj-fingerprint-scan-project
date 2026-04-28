import logging
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from typing import Dict, Any

# path configuration variables
HISTORY_PATH = os.getenv("HISTORY_PATH", os.path.join("results", "trained_models", "training_history.csv"))
RESULTS_PATH = os.getenv("RESULTS_PATH", os.path.join("results", "validations", "validation_results.pkl"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join("results", "plots"))


def plot_learning_curves(history_df: pd.DataFrame) -> None:
    """
    Plots and saves the training/validation loss and accuracy curves.

    Args:
        history_df (pd.DataFrame): The history metrics captured during training.

    Returns:
        None
    """
    logging.info("Plotting learning curves...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # plot loss
    ax1.plot(history_df['epoch'], history_df['train_loss'], label='Training Loss', color='royalblue')
    ax1.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss', color='darkorange')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.set_ylim(bottom=0)

    # plot accuracy
    ax2.plot(history_df['epoch'], history_df['val_accuracy'], label='Validation Accuracy', color='forestgreen')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "learning_curves.png")
    plt.savefig(save_path)
    logging.info(f"Learning curves saved to {save_path}")
    plt.close()


def plot_distance_distribution(results: Dict[str, Any]) -> None:
    """
    Plots and saves the distribution of genuine and impostor distances.

    Args:
        results (Dict[str, Any]): Dictionary containing calculated sample distances.

    Returns:
        None
    """
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "distance_distribution.png")
    plt.savefig(save_path)
    logging.info(f"Distance distribution plot saved to {save_path}")
    plt.close()


def plot_roc_curve(results: Dict[str, Any]) -> None:
    """
    Calculates and plots the Receiver Operating Characteristic curve.

    Args:
        results (Dict[str, Any]): Evaluation predictions and truth markers.

    Returns:
        None
    """
    logging.info("Plotting ROC curve...")
    if not results['genuine'] or not results['impostor']:
        logging.warning("Not enough data to plot ROC curve.")
        return

    y_true = np.concatenate([np.ones(len(results['genuine'])), np.zeros(len(results['impostor']))])
    # use negated distance to ensure higher score is better
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(save_path)
    logging.info(f"ROC curve saved to {save_path}")
    plt.close()


def process_history(history_path: str) -> None:
    """
    Loads and plots the training history if the specific target file exists.

    Args:
        history_path (str): Validation output history location string.

    Returns:
        None
    """
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        plot_learning_curves(history_df)
    else:
        logging.warning(f"'{history_path}' not found. Skipping learning curve plots.")


def process_results(results_path: str) -> None:
    """
    Loads and plots the validation results if the specific target file exists.

    Args:
        results_path (str): Distance prediction output history location.

    Returns:
        None
    """
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        plot_distance_distribution(results)
        plot_roc_curve(results)
    else:
        logging.warning(f"'{results_path}' not found. Skipping distance and ROC plots.")


def main() -> None:
    """
    Executes script plotting requirements globally.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    process_history(HISTORY_PATH)
    process_results(RESULTS_PATH)


if __name__ == '__main__':
    main()
