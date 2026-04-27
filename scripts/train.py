import os
import csv
import torch
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from tqdm import tqdm
import cv2
import torch.nn.functional as F

from data.data_loader import load_and_split_data
from data.dataset import SiameseFingerprintDataset, FingerprintImageDataset, BalancedBatchSampler
from models.model import EmbeddingNet, ContrastiveLoss, TripletLoss
from preprocessing.preprocess import FingerprintPreprocessor

# Configuration
DATASET_PATH = os.getenv("SOCOFING_DATASET_PATH")
MODEL_SAVE_PATH = "trained_models/embedding_net.pth"
HISTORY_SAVE_PATH = "training_history.csv"

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.00005  # A slightly lower learning rate is often better for TripletLoss
TRIPLET_MARGIN = 0.4
VALIDATION_THRESHOLD = 0.5  # Distance threshold for accuracy calculation.
NUM_TRAINING_USERS = 400  # The number of users to include for training and validation.


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

    # For online hard mining, we need a dataset that returns individual images and labels,
    # and a special sampler to create batches with multiple instances of the same class.
    train_dataset = FingerprintImageDataset(train_files, transform=train_transform, preprocessor=quiet_preprocessor)
    # Each batch will contain P fingers, with K images from each finger.
    train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=8, n_samples=4) # Batch size = 8 * 4 = 32
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=4, pin_memory=True)

    # The validation dataset remains pair-based for consistent metric calculation.
    val_dataset = SiameseFingerprintDataset(val_files, transform=val_transform, preprocessor=quiet_preprocessor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    logging.info(f"Training data loader size: {len(train_loader)} batches")
    logging.info(f"Validation data loader size: {len(val_loader)} batches")

    # 3. Model, Optimizer, and Checkpoint Loading
    logging.info("Initializing model...")
    embedding_net = EmbeddingNet(embedding_dim=128).to(device)
    # Use TripletLoss for training and ContrastiveLoss for validation metrics
    train_criterion = TripletLoss(margin=TRIPLET_MARGIN)
    val_criterion = ContrastiveLoss(margin=1.0) # Keep standard margin for validation
    optimizer = AdamW(embedding_net.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    start_epoch = 0
    best_val_loss = float('inf')
    training_history = []

    # Check if a checkpoint exists to resume training.
    if os.path.exists(MODEL_SAVE_PATH):
        logging.info(f"Resuming training from checkpoint: {MODEL_SAVE_PATH}")
        checkpoint = torch.load(MODEL_SAVE_PATH)
        embedding_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        logging.info(f"Resumed from Epoch {start_epoch}. Best validation loss so far: {best_val_loss:.4f}")
    else:
        logging.info("Starting a new training session.")


    # 4. Training and Validation Loop
    logging.info(f"Training for {NUM_EPOCHS - start_epoch} more epochs.")
    for epoch in range(start_epoch, NUM_EPOCHS):
        logging.info(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        logging.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Training Phase
        embedding_net.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training")
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            embeddings = embedding_net(images)

            # --- Online Semi-Hard Negative Mining ---
            # Calculate pairwise distance matrix
            pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

            all_triplets_loss = []
            # Iterate through each sample in the batch as an anchor
            for i in range(len(labels)):
                anchor_label = labels[i]
                dists = pairwise_dist[i]

                # Find all positives and negatives for this anchor
                pos_mask = (labels == anchor_label) & (torch.arange(len(labels)).to(device) != i)
                neg_mask = labels != anchor_label

                if not torch.any(pos_mask) or not torch.any(neg_mask):
                    continue

                # Select the hardest positive (furthest from anchor)
                hardest_pos_dist = dists[pos_mask].max()

                # Select semi-hard negatives: dist(A,P) < dist(A,N) < dist(A,P) + margin
                semi_hard_neg_mask = neg_mask & (dists > hardest_pos_dist) & (dists < (hardest_pos_dist + TRIPLET_MARGIN))
                if torch.any(semi_hard_neg_mask):
                    # Form triplets with all semi-hard negatives and calculate their loss
                    loss_for_anchor = F.relu(hardest_pos_dist - dists[semi_hard_neg_mask] + TRIPLET_MARGIN)
                    all_triplets_loss.append(loss_for_anchor)

            # Only perform backpropagation if valid triplets were found in the batch.
            if all_triplets_loss:
                loss = torch.cat(all_triplets_loss).mean()
                loss.backward()
                optimizer.step()
            else:
                # If no valid triplets are found, the loss is effectively zero for this batch.
                # We create a tensor for logging purposes but do not backpropagate.
                loss = torch.tensor(0.0)

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

        # Validation Phase
        embedding_net.eval()
        val_loss = 0.0
        all_pos_distances = []
        all_neg_distances = []
        val_correct_predictions = 0
        val_total_predictions = 0

        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc="Validating")
            for img1, img2, label in progress_bar_val:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                label_squeezed = label.unsqueeze(1)

                output1 = embedding_net(img1)
                output2 = embedding_net(img2)

                loss = val_criterion(output1, output2, label_squeezed)
                val_loss += loss.item()

                # Calculate distances for metric tracking
                distances = F.pairwise_distance(output1, output2)
                all_pos_distances.extend(distances[label == 1].cpu().numpy())
                all_neg_distances.extend(distances[label == 0].cpu().numpy())

                # Calculate accuracy for this batch
                val_correct_predictions += torch.sum(distances[label == 1] < VALIDATION_THRESHOLD).item()
                val_correct_predictions += torch.sum(distances[label == 0] >= VALIDATION_THRESHOLD).item()
                val_total_predictions += len(distances)

        avg_val_loss = val_loss / len(val_loader)
        avg_pos_dist = sum(all_pos_distances) / len(all_pos_distances) if all_pos_distances else 0
        avg_neg_dist = sum(all_neg_distances) / len(all_neg_distances) if all_neg_distances else 0
        val_accuracy = (val_correct_predictions / val_total_predictions) * 100 if val_total_predictions > 0 else 0

        logging.info(f"Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.4f}")
        logging.info(f"Epoch {epoch+1} - Validation Accuracy: {val_accuracy:.2f}%")
        logging.info(f"Epoch {epoch+1} - Avg. Positive Pair Distance: {avg_pos_dist:.4f}")
        logging.info(f"Epoch {epoch+1} - Avg. Negative Pair Distance: {avg_neg_dist:.4f}")

        # Store metrics for this epoch
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
            'avg_pos_dist': avg_pos_dist,
            'avg_neg_dist': avg_neg_dist
        })

        # Save the model if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save a checkpoint dictionary for robust resuming.
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': embedding_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, MODEL_SAVE_PATH)
            logging.info(f"Validation loss improved. Checkpoint saved to {MODEL_SAVE_PATH}")

        # Step the scheduler based on the validation loss
        scheduler.step(avg_val_loss)

    logging.info("Training complete.")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")

    # 5. Save Training History
    logging.info(f"Saving training history to {HISTORY_SAVE_PATH}")
    with open(HISTORY_SAVE_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=training_history[0].keys())
        writer.writeheader()
        writer.writerows(training_history)

    logging.info("Script finished.")


if __name__ == '__main__':
    main()
