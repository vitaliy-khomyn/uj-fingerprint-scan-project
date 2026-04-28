import csv
import cv2
import logging
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

from data.data_loader import load_and_split_data
from data.dataset import SiameseFingerprintDataset
from models.model import ContrastiveLoss, EmbeddingNet
from preprocessing.preprocess import FingerprintPreprocessor

# configuration settings
DATASET_PATH = os.getenv("SOCOFING_DATASET_PATH")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", os.path.join("results", "trained_models", "embedding_net.pth"))
HISTORY_SAVE_PATH = os.getenv("HISTORY_SAVE_PATH", os.path.join("results", "trained_models", "training_history.csv"))

# hyperparameter settings
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
CONTRASTIVE_MARGIN = 1.0
EARLY_STOPPING_PATIENCE = 10
NUM_TRAINING_USERS = 400
VALIDATION_FREQUENCY = 5
CACHE_IMAGES = False


def train_epoch(
    embedding_net: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int
) -> float:
    """
    Handles the training loop execution for a single epoch.

    Args:
        embedding_net (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader providing the batches.
        optimizer (torch.optim.Optimizer): The optimizer used.
        criterion (torch.nn.Module): The loss function used.
        scaler (torch.amp.GradScaler): The gradient scaler for mixed precision.
        device (torch.device): The computational device mapped.
        epoch (int): The current epoch index.

    Returns:
        float: The average training loss over the epoch.
    """
    embedding_net.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training")
    for i, (img1, img2, label) in enumerate(progress_bar):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            output1 = embedding_net(img1)
            output2 = embedding_net(img2)
            loss = criterion(output1, output2, label.unsqueeze(1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        postfix_dict = {'loss': f'{loss.item():.4f}'}
        if device.type == 'cuda':
            postfix_dict['vram_mb'] = f"{torch.cuda.memory_allocated() / (1024**2):.0f}"
        progress_bar.set_postfix(postfix_dict)

    avg_train_loss = train_loss / len(train_loader)
    logging.info(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")
    return avg_train_loss


def validate_epoch(
    embedding_net: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[float, float, float, float]:
    """
    Handles the validation loop and calculates metrics for a single epoch.

    Args:
        embedding_net (torch.nn.Module): The evaluated model.
        val_loader (DataLoader): The data loader for validation pairs.
        criterion (torch.nn.Module): The validation loss function.
        device (torch.device): The memory device mapped.
        epoch (int): The current epoch index.

    Returns:
        Tuple[float, float, float, float]: Average loss, accuracy, positive distance, and negative distance.
    """
    embedding_net.eval()
    val_loss = 0.0
    all_pos_distances = []
    all_neg_distances = []

    with torch.no_grad():
        progress_bar_val = tqdm(val_loader, desc="Validating")
        for img1, img2, label in progress_bar_val:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            label_squeezed = label.unsqueeze(1)

            output1 = embedding_net(img1)
            output2 = embedding_net(img2)

            loss = criterion(output1, output2, label_squeezed)
            val_loss += loss.item()

            # calculate distances for metric tracking
            distances = F.pairwise_distance(output1, output2)
            all_pos_distances.extend(distances[label == 1].cpu().numpy())
            all_neg_distances.extend(distances[label == 0].cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    avg_pos_dist = sum(all_pos_distances) / len(all_pos_distances) if all_pos_distances else 0
    avg_neg_dist = sum(all_neg_distances) / len(all_neg_distances) if all_neg_distances else 0

    best_val_accuracy = 0.0
    best_threshold = 0.5
    if all_pos_distances and all_neg_distances:
        pos_dist_arr = np.array(all_pos_distances)
        neg_dist_arr = np.array(all_neg_distances)
        min_d = min(np.min(pos_dist_arr), np.min(neg_dist_arr))
        max_d = max(np.max(pos_dist_arr), np.max(neg_dist_arr))

        for t in np.linspace(min_d, max_d, 100):
            correct = np.sum(pos_dist_arr < t) + np.sum(neg_dist_arr >= t)
            acc = correct / (len(pos_dist_arr) + len(neg_dist_arr))
            if acc > best_val_accuracy:
                best_val_accuracy = acc
                best_threshold = t

    val_accuracy = best_val_accuracy * 100

    logging.info(f"Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.4f}")
    logging.info(f"Epoch {epoch+1} - Validation Accuracy: {val_accuracy:.2f}% (at threshold {best_threshold:.4f})")
    logging.info(f"Epoch {epoch+1} - Avg. Positive Pair Distance: {avg_pos_dist:.4f}")
    logging.info(f"Epoch {epoch+1} - Avg. Negative Pair Distance: {avg_neg_dist:.4f}")
    return avg_val_loss, val_accuracy, avg_pos_dist, avg_neg_dist


def main() -> None:
    """
    Executes the main training script workflow.

    Returns:
        None
    """
    # restrict opencv thread count to prevent pytorch multiprocessing issues
    cv2.setNumThreads(0)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # setup and configuration
    if not DATASET_PATH:
        raise ValueError("SOCOFING_DATASET_PATH environment variable not set. Please create a .env file.")

    # ensure directory for saving model exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(HISTORY_SAVE_PATH), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # data loading and preparation
    logging.info("Loading and splitting data...")
    # test split serves as validation set during training
    train_files, val_files = load_and_split_data(DATASET_PATH, max_user_id=NUM_TRAINING_USERS)

    if not train_files or not val_files:
        logging.error("No training or validation data was loaded. Please check the dataset path in your .env file.")
        return

    quiet_preprocessor = FingerprintPreprocessor()

    # set transforms
    train_transform = SiameseFingerprintDataset.get_training_transform()
    val_transform = SiameseFingerprintDataset.get_validation_transform()

    # dynamically calculate optimal background workers based on cpu cores
    optimal_workers = min(8, max(2, (os.cpu_count() or 4) - 2))

    # setup datasets
    train_dataset = SiameseFingerprintDataset(train_files, transform=train_transform, preprocessor=quiet_preprocessor, use_cache=CACHE_IMAGES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=optimal_workers, pin_memory=True, persistent_workers=True)

    val_dataset = SiameseFingerprintDataset(val_files, transform=val_transform, preprocessor=quiet_preprocessor, use_cache=CACHE_IMAGES)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=optimal_workers, pin_memory=True, persistent_workers=True)

    logging.info(f"Training data loader size: {len(train_loader)} batches")
    logging.info(f"Validation data loader size: {len(val_loader)} batches")

    # setup model optimizer and checkpoint loading
    logging.info("Initializing model...")
    embedding_net = EmbeddingNet(embedding_dim=128).to(device)

    # compile model for avoiding windows
    if hasattr(torch, 'compile') and device.type == 'cuda' and os.name != 'nt':
        logging.info("Compiling model with torch.compile() for faster execution...")
        embedding_net = torch.compile(embedding_net)

    criterion = ContrastiveLoss(margin=CONTRASTIVE_MARGIN)
    optimizer = AdamW(embedding_net.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == 'cuda')

    logging.info("\n--- Training Configuration ---")
    logging.info(f"Batch Size: {BATCH_SIZE}")
    logging.info(f"Learning Rate: {LEARNING_RATE}")
    logging.info(f"Mixed Precision (AMP): {'Enabled' if device.type == 'cuda' else 'Disabled'}")
    logging.info(f"Model Compiled: {hasattr(torch, 'compile') and device.type == 'cuda' and os.name != 'nt'}")
    logging.info(f"Validation Frequency: Every {VALIDATION_FREQUENCY} epochs")
    logging.info("------------------------------\n")

    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0  # for early stopping checks
    training_history = []

    # load checkpoint to resume training
    if os.path.exists(MODEL_SAVE_PATH):
        logging.info(f"Resuming training from checkpoint: {MODEL_SAVE_PATH}")
        checkpoint = torch.load(MODEL_SAVE_PATH)
        embedding_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        logging.info(f"Resumed from Epoch {start_epoch}. Best validation loss: {best_val_loss:.4f}. Epochs without improvement: {epochs_without_improvement}")
    else:
        logging.info("Starting a new training session.")

    # variables to persist metric logging on skipped validation epochs
    avg_val_loss = best_val_loss if best_val_loss != float('inf') else 0.0
    val_accuracy = 0.0
    avg_pos_dist = 0.0
    avg_neg_dist = 0.0

    # execute training and validation loop
    logging.info(f"Training for {NUM_EPOCHS - start_epoch} more epochs.")
    for epoch in range(start_epoch, NUM_EPOCHS):
        logging.info(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        logging.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # training phase
        avg_train_loss = train_epoch(embedding_net, train_loader, optimizer, criterion, scaler, device, epoch)

        if (epoch + 1) % VALIDATION_FREQUENCY == 0 or (epoch + 1) == NUM_EPOCHS:
            # validation phase
            avg_val_loss, val_accuracy, avg_pos_dist, avg_neg_dist = validate_epoch(embedding_net, val_loader, criterion, device, epoch)
        else:
            logging.info(f"Epoch {epoch+1} - Validation skipped (runs every {VALIDATION_FREQUENCY} epochs).")

        # store metrics for epoch
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'avg_pos_dist': avg_pos_dist,
            'avg_neg_dist': avg_neg_dist
        })

        if (epoch + 1) % VALIDATION_FREQUENCY == 0 or (epoch + 1) == NUM_EPOCHS:
            # save model if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': embedding_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'epochs_without_improvement': epochs_without_improvement
                }, MODEL_SAVE_PATH)
                logging.info(f"Validation loss improved. Checkpoint saved to {MODEL_SAVE_PATH}")
            else:
                epochs_without_improvement += 1
                logging.info(f"Validation loss did not improve. Count: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}")

            scheduler.step(avg_val_loss)

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Stopping early as validation loss has not improved for {EARLY_STOPPING_PATIENCE} epochs.")
                break

    logging.info("Training complete.")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")

    # save training history
    logging.info(f"Saving training history to {HISTORY_SAVE_PATH}")
    if training_history:
        file_exists = os.path.exists(HISTORY_SAVE_PATH)
        write_mode = 'a' if file_exists and start_epoch > 0 else 'w'
        with open(HISTORY_SAVE_PATH, write_mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=training_history[0].keys())
            if write_mode == 'w':
                writer.writeheader()
            writer.writerows(training_history)

    logging.info("Script finished.")


if __name__ == '__main__':
    main()
