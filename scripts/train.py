import os
import csv
import torch
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from tqdm import tqdm
import cv2
import torch.nn.functional as F
import numpy as np

from data.data_loader import load_and_split_data
from data.dataset import SiameseFingerprintDataset
from models.model import EmbeddingNet, ContrastiveLoss
from preprocessing.preprocess import FingerprintPreprocessor

# Configuration
DATASET_PATH = os.getenv("SOCOFING_DATASET_PATH")
MODEL_SAVE_PATH = "trained_models/embedding_net.pth"
HISTORY_SAVE_PATH = "training_history.csv"

# Hyperparameters
NUM_EPOCHS = 100  # A reasonable upper limit; Early Stopping will likely finish sooner.
BATCH_SIZE = 64  # Increased to fully utilize GPU VRAM and process more images in parallel.
LEARNING_RATE = 0.0001
CONTRASTIVE_MARGIN = 1.0
EARLY_STOPPING_PATIENCE = 10  # Stop if val_loss doesn't improve for 10 epochs.
NUM_TRAINING_USERS = 400  # The number of users to include for training and validation.
VALIDATION_FREQUENCY = 5  # Run validation every N epochs to save time.
CACHE_IMAGES = False  # Set to True if you have 8GB+ RAM, False to save RAM.


def main():
    # Set OpenCV to use only one thread to avoid conflicts with PyTorch's multiprocessing.
    cv2.setNumThreads(0)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Setup and Configuration
    if not DATASET_PATH:
        raise ValueError("SOCOFING_DATASET_PATH environment variable not set. Please create a .env file.")

    # Ensure the directory for saving the model exists to prevent errors.
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 2. Data Loading and Preparation
    logging.info("Loading and splitting data...")
    # The 'test' split from our loader serves as the validation set during training.
    train_files, val_files = load_and_split_data(DATASET_PATH, max_user_id=NUM_TRAINING_USERS)

    if not train_files or not val_files:
        logging.error("No training or validation data was loaded. Please check the dataset path in your .env file.")
        return

    # Use a non-verbose preprocessor for cleaner logs during batch processing.
    quiet_preprocessor = FingerprintPreprocessor(verbose=False)

    # Use augmented transforms for training and standard transforms for validation.
    train_transform = SiameseFingerprintDataset.get_training_transform()
    val_transform = SiameseFingerprintDataset.get_validation_transform()

    # Dynamically calculate optimal background workers based on your CPU cores
    optimal_workers = min(8, max(2, (os.cpu_count() or 4) - 2))

    # Use the Siamese dataset for training, which generates positive/negative pairs.
    train_dataset = SiameseFingerprintDataset(train_files, transform=train_transform, preprocessor=quiet_preprocessor, use_cache=CACHE_IMAGES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=optimal_workers, pin_memory=True, persistent_workers=True)

    # The validation dataset also uses the Siamese pair-based structure.
    val_dataset = SiameseFingerprintDataset(val_files, transform=val_transform, preprocessor=quiet_preprocessor, use_cache=CACHE_IMAGES)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=optimal_workers, pin_memory=True, persistent_workers=True)

    logging.info(f"Training data loader size: {len(train_loader)} batches")
    logging.info(f"Validation data loader size: {len(val_loader)} batches")

    # 3. Model, Optimizer, and Checkpoint Loading
    logging.info("Initializing model...")
    embedding_net = EmbeddingNet(embedding_dim=128).to(device)

    # Suggestion 5: Use torch.compile() for optimized execution graph on PyTorch 2.0+
    # Disabled on Windows ('nt') as Triton backend is not natively supported.
    if hasattr(torch, 'compile') and device.type == 'cuda' and os.name != 'nt':
        logging.info("Compiling model with torch.compile() for faster execution...")
        embedding_net = torch.compile(embedding_net)

    # Use ContrastiveLoss for both training and validation.
    criterion = ContrastiveLoss(margin=CONTRASTIVE_MARGIN)
    optimizer = AdamW(embedding_net.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    # Suggestion 2: Automatic Mixed Precision (AMP) Scaler
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
    epochs_without_improvement = 0  # For early stopping
    training_history = []

    # Check if a checkpoint exists to resume training.
    if os.path.exists(MODEL_SAVE_PATH):
        logging.info(f"Resuming training from checkpoint: {MODEL_SAVE_PATH}")
        checkpoint = torch.load(MODEL_SAVE_PATH)
        embedding_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        # Load the early stopping counter if it exists, otherwise reset.
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        logging.info(f"Resumed from Epoch {start_epoch}. Best validation loss: {best_val_loss:.4f}. Epochs without improvement: {epochs_without_improvement}")
    else:
        logging.info("Starting a new training session.")

    # Variables to persist metric logging on skipped validation epochs
    avg_val_loss = best_val_loss if best_val_loss != float('inf') else 0.0
    val_accuracy = 0.0
    avg_pos_dist = 0.0
    avg_neg_dist = 0.0

    # 4. Training and Validation Loop
    logging.info(f"Training for {NUM_EPOCHS - start_epoch} more epochs.")
    for epoch in range(start_epoch, NUM_EPOCHS):
        logging.info(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        logging.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Training Phase
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

        if (epoch + 1) % VALIDATION_FREQUENCY == 0 or (epoch + 1) == NUM_EPOCHS:
            # Validation Phase
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

                    # Calculate distances for metric tracking
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
        else:
            logging.info(f"Epoch {epoch+1} - Validation skipped (runs every {VALIDATION_FREQUENCY} epochs).")

        # Store metrics for this epoch
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'avg_pos_dist': avg_pos_dist,
            'avg_neg_dist': avg_neg_dist
        })

        if (epoch + 1) % VALIDATION_FREQUENCY == 0 or (epoch + 1) == NUM_EPOCHS:
            # Save the model if validation loss has improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0  # Reset counter
                # Save a checkpoint dictionary for robust resuming.
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

            # Step the scheduler based on the validation loss
            scheduler.step(avg_val_loss)

            # Check for early stopping
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Stopping early as validation loss has not improved for {EARLY_STOPPING_PATIENCE} epochs.")
                break

    logging.info("Training complete.")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")

    # 5. Save Training History
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
