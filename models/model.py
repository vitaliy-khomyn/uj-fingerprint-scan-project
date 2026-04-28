import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class EmbeddingNet(nn.Module):
    """
    Creates an embedding model from a pre-trained MobileNetV2 base.
    The model takes an image and outputs a normalized embedding vector.
    """
    def __init__(self, embedding_dim=128):
        """
        Initializes the embedding network base on MobileNetV2.

        Args:
            embedding_dim (int): The dimensionality of the output embedding.

        Returns:
            None
        """
        super(EmbeddingNet, self).__init__()
        self.base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # fine-tuning strategy: freeze the first 14 feature blocks of MobileNetV2
        # and allow the subsequent layers to be trained.
        for i, child in enumerate(self.base_model.features.children()):
            is_trainable = i >= 14
            for param in child.parameters():
                param.requires_grad = is_trainable

        in_features = self.base_model.classifier[1].in_features

        # replace original classifier with new linear layer for embedding generation
        self.base_model.classifier = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        """
        Performs a forward pass to generate a normalized embedding.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The normalized embedding vector.
        """
        embedding = self.base_model(x)
        # l2-normalize output embedding vector for consistent distance metrics
        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        return normalized_embedding


class SiameseNet(nn.Module):
    """
    A Siamese network that takes two images, passes them through the embedding net,
    and returns the distance between their embeddings. This class is useful for
    conceptual understanding and inference, but for training, it's often more
    efficient to compute the loss directly from the embeddings in the training loop.

    This class is not used in the current training pipeline but is kept for clarity.
    """
    def __init__(self, embedding_net):
        """
        Initializes the Siamese network logic.

        Args:
            embedding_net (torch.nn.Module): The base embedding network.

        Returns:
            None
        """
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, input1, input2):
        """
        Computes the distance between the embeddings of two inputs.

        Args:
            input1 (torch.Tensor): The first input tensor.
            input2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The calculated pairwise distance.
        """
        embedding1 = self.embedding_net(input1)
        embedding2 = self.embedding_net(input2)

        distance = F.pairwise_distance(embedding1, embedding2)
        return distance


class ContrastiveLoss(nn.Module):
    """
    Implements the contrastive loss function.
    This loss function pushes together embeddings for positive pairs (label=1)
    and pulls apart embeddings for negative pairs (label=0) up to a specified margin.
    """
    def __init__(self, margin=1.0):
        """
        Initializes the Contrastive Loss function.

        Args:
            margin (float): The required margin between negative pairs.

        Returns:
            None
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        """
        Computes the contrastive loss metric.

        Args:
            embedding1 (torch.Tensor): The embedding of the first image batch.
            embedding2 (torch.Tensor): The embedding of the second image batch.
            label (torch.Tensor): A tensor of labels (1 for positive, 0 for negative).

        Returns:
            torch.Tensor: The computed loss tensor.
        """
        euclidean_distance = F.pairwise_distance(embedding1, embedding2, keepdim=True)
        loss_positive = (label) * (euclidean_distance ** 2)
        loss_negative = (1 - label) * (torch.clamp(self.margin - euclidean_distance, min=0.0) ** 2)

        loss_contrastive = torch.mean(loss_positive + loss_negative)
        return loss_contrastive


def verify_embedding_net(device):
    """
    Verifies the outputs and properties of the EmbeddingNet.

    Args:
        device (torch.device): The device memory used for tensor execution.

    Returns:
        None
    """

    # verify the embedding network
    logging.info("\n--- Verifying EmbeddingNet ---")
    embedding_net = EmbeddingNet(embedding_dim=128).to(device)

    # create a dummy input tensor (batch_size, channels, height, width)
    dummy_image = torch.randn(1, 3, 128, 128).to(device)
    output_embedding = embedding_net(dummy_image)
    logging.info(f"Shape of output embedding: {output_embedding.shape}")
    logging.info(f"Magnitude of embedding vector (should be ~1.0): {torch.linalg.norm(output_embedding):.4f}")


def verify_contrastive_loss(device):
    """
    Verifies the ContrastiveLoss logic with positive and negative pairs.

    Args:
        device (torch.device): The device used for processing tensors.

    Returns:
        None
    """
    logging.info("\n--- Verifying ContrastiveLoss ---")
    loss_fn = ContrastiveLoss(margin=1.0)

    # test with a positive pair (similar embeddings)
    emb1_pos = F.normalize(torch.randn(4, 128)).to(device)
    emb2_pos = F.normalize(emb1_pos + torch.randn_like(emb1_pos) * 0.01)
    labels_pos = torch.ones(4, 1).to(device)
    loss_pos = loss_fn(emb1_pos, emb2_pos, labels_pos)
    logging.info(f"Loss for positive pair (should be low): {loss_pos.item():.4f}")

    # test with a negative pair inside the margin (different but close)
    emb1_neg_close = F.normalize(torch.randn(4, 128)).to(device)
    emb2_neg_close = F.normalize(emb1_neg_close + torch.randn_like(emb1_neg_close) * 0.3) # Distance < margin
    labels_neg_close = torch.zeros(4, 1).to(device)
    loss_neg_close = loss_fn(emb1_neg_close, emb2_neg_close, labels_neg_close)
    logging.info(f"Loss for negative pair inside margin (should be > 0): {loss_neg_close.item():.4f}")

    # test with a negative pair outside the margin (far apart)
    emb1_neg_far = F.normalize(torch.randn(4, 128)).to(device)
    emb2_neg_far = F.normalize(emb1_neg_far + torch.randn_like(emb1_neg_far) * 2.0) # Distance > margin
    labels_neg_far = torch.zeros(4, 1).to(device)
    loss_neg_far = loss_fn(emb1_neg_far, emb2_neg_far, labels_neg_far)
    logging.info(f"Loss for negative pair outside margin (should be 0): {loss_neg_far.item():.4f}")


def main():
    """
    Runs verification tests on the model components.
    This function is executed when the script is run directly.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running model verification on device: {device}")

    verify_embedding_net(device)
    verify_contrastive_loss(device)


if __name__ == '__main__':
    main()
