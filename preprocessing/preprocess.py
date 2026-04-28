import cv2
import numpy as np
import logging


class FingerprintPreprocessor:
    """
    A class to encapsulate fingerprint image preprocessing steps for a CNN-based model.
    """

    def __init__(self, image_size=(128, 128), clahe_clip_limit=2.0, clahe_grid_size=(8, 8), verbose=True):
        """
        Initializes the preprocessor with parameters for various steps.

        Args:
            image_size (tuple): The target size (height, width) for the network input.
            clahe_clip_limit (float): Clip limit for CLAHE algorithm.
            clahe_grid_size (tuple): Grid size for CLAHE algorithm.
            verbose (bool): If True, prints processing steps.
        """
        self.image_size = image_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.verbose = verbose

    def _grayscale_conversion(self, image):
        """Converts an image to grayscale if it's not already."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _normalize_image(self, gray_image):
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_grid_size)
        normalized_image = clahe.apply(gray_image)
        return normalized_image

    def _resize_image(self, image):
        """Resizes the image to the target size for the CNN."""
        # We add an extra dimension and then convert to RGB for pre-trained models
        # that expect 3-channel inputs.
        resized = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    def preprocess(self, image_path):
        """
        Performs the complete preprocessing pipeline on a fingerprint image.

        Args:
            image_path (str): Path to the input fingerprint image.

        Returns:
            numpy.ndarray: The preprocessed fingerprint image ready for the model.
        """
        if self.verbose:
            logging.info(f"Processing image: {image_path}")
        # Read directly as grayscale! Saves disk bandwidth and skips CPU conversion
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        if self.verbose:
            logging.info("Step 1: Grayscale Conversion - Done.")

        normalized_image = self._normalize_image(gray_image)
        if self.verbose:
            logging.info("Step 2: Normalization (CLAHE) - Done.")

        final_image = self._resize_image(normalized_image)
        if self.verbose:
            logging.info(f"Step 3: Resizing to {self.image_size} - Done.")

        if self.verbose:
            logging.info("Preprocessing complete.")
        return final_image


# Example Usage (assuming you have a fingerprint image named 'fingerprint.png')
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Example with a standard CNN input size
    preprocessor = FingerprintPreprocessor(image_size=(128, 128), verbose=True)
    try:
        # Create a dummy image for demonstration if 'fingerprint.png' doesn't exist
        dummy_image_path = "fingerprint.png"
        try:
            test_image = cv2.imread(dummy_image_path)
            if test_image is None:
                raise FileNotFoundError
        except FileNotFoundError:
            logging.info(f"'{dummy_image_path}' not found. Creating a dummy image for demonstration.")
            dummy_img = np.full((200, 200), 128, dtype=np.uint8)
            # Draw some lines to simulate ridges
            for i in range(10, 190, 20):
                cv2.line(dummy_img, (10, i), (190, i), 255, 3)
            cv2.imwrite(dummy_image_path, dummy_img)
            logging.info(f"Dummy image saved as '{dummy_image_path}'.")

        processed_fingerprint = preprocessor.preprocess(dummy_image_path)
        cv2.imwrite("processed_fingerprint.png", processed_fingerprint)
        logging.info("Processed image saved as 'processed_fingerprint.png'")

        # Display results (optional, requires matplotlib or direct imshow)
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
