import cv2
import logging
import numpy as np
from typing import Tuple


class FingerprintPreprocessor:
    """
    Encapsulates fingerprint image preprocessing steps for a CNN-based model.
    """

    def __init__(self, image_size: Tuple[int, int] = (128, 128), clahe_clip_limit: float = 2.0, clahe_grid_size: Tuple[int, int] = (8, 8)) -> None:
        """
        Initializes the preprocessor with parameters for various steps.

        Args:
            image_size (Tuple[int, int]): Target size for network input.
            clahe_clip_limit (float): Clip limit for CLAHE algorithm.
            clahe_grid_size (Tuple[int, int]): Grid size for CLAHE algorithm.

        Returns:
            None
        """
        self.image_size = image_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size

    def _grayscale_conversion(self, image: np.ndarray) -> np.ndarray:
        """
        Converts an image to grayscale if needed.

        Args:
            image (np.ndarray): The target image to inspect and convert.

        Returns:
            np.ndarray: The resulting grayscale image.
        """
        # converted to grayscale if channels indicate color image
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _normalize_image(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Applies contrast limited adaptive histogram equalization.

        Args:
            gray_image (np.ndarray): A single-channel grayscale image array.

        Returns:
            np.ndarray: The normalized contrast image.
        """
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size)
        normalized_image = clahe.apply(gray_image)
        return normalized_image

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes an image to the target size required for the CNN.

        Args:
            image (np.ndarray): The preprocessed 2D array representation.

        Returns:
            np.ndarray: The resized, 3-channel representation.
        """
        # extra dimension added and converted to rgb for pre-trained models expecting 3-channel inputs
        resized = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Performs the complete preprocessing pipeline on a fingerprint image.

        Args:
            image_path (str): The path to the input fingerprint image.

        Returns:
            np.ndarray: The preprocessed fingerprint image ready for the model.
        """
        # read directly as grayscale to save disk bandwidth and skip conversion compute
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        normalized_image = self._normalize_image(gray_image)

        final_image = self._resize_image(normalized_image)
        return final_image


def _create_dummy_image(image_path: str) -> None:
    """
    Creates a dummy fingerprint image to fulfill testing requirements.

    Args:
        image_path (str): The target file destination output.

    Returns:
        None
    """
    dummy_img = np.full((200, 200), 128, dtype=np.uint8)
    # draw lines to simulate ridges
    for i in range(10, 190, 20):
        cv2.line(dummy_img, (10, i), (190, i), 255, 3)
    cv2.imwrite(image_path, dummy_img)


def _main() -> None:
    """
    Demonstrates module functionality independently.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # initiated example with standard cnn input size
    preprocessor = FingerprintPreprocessor(image_size=(128, 128))
    try:
        dummy_image_path = "fingerprint.png"
        try:
            test_image = cv2.imread(dummy_image_path)
            if test_image is None:
                raise FileNotFoundError
        except FileNotFoundError:
            logging.info(f"'{dummy_image_path}' not found. Creating a dummy image for demonstration.")
            _create_dummy_image(dummy_image_path)
            logging.info(f"Dummy image saved as '{dummy_image_path}'.")

        processed_fingerprint = preprocessor.preprocess(dummy_image_path)
        cv2.imwrite("processed_fingerprint.png", processed_fingerprint)
        logging.info("Processed image saved as 'processed_fingerprint.png'")

        # output results to ui
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original (Dummy)")
        plt.imshow(cv2.imread(dummy_image_path))
        plt.subplot(1, 2, 2)
        plt.title("Processed")
        plt.imshow(processed_fingerprint)
        plt.show()

    except FileNotFoundError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    _main()
