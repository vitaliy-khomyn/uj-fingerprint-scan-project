import os
import sys
import pickle
import numpy as np
import torch
import cv2
import streamlit as st

# Add project root to sys.path to allow imports from our custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from models.model import EmbeddingNet
from preprocessing.preprocess import FingerprintPreprocessor
from data.dataset import SiameseFingerprintDataset


# --- Configuration ---
MODEL_PATH = os.path.join("..", "trained_models", "embedding_net.pth")
ENROLLMENT_DB_PATH = os.path.join("..", "enrollment_db.pkl")
# Use the optimal threshold found during validation (EER threshold)
MATCH_THRESHOLD = 0.1350  


st.set_page_config(page_title="Fingerprint Auth")


@st.cache_resource
def load_system_components():
    """Loads the model and database once and caches them in memory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the embedding model
    model = EmbeddingNet(embedding_dim=128).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load the enrollment database
    with open(ENROLLMENT_DB_PATH, 'rb') as f:
        enrollment_db = pickle.load(f)

    preprocessor = FingerprintPreprocessor(verbose=False)
    transform = SiameseFingerprintDataset.get_validation_transform()

    return model, enrollment_db, preprocessor, transform, device


def main():
    st.title("Fingerprint Authorization System")
    st.write("Upload a fingerprint scan to verify your identity.")

    try:
        model, enrollment_db, preprocessor, transform, device = load_system_components()
    except Exception as e:
        st.error(f"Failed to load system components. Ensure your model and database exist in the root folder. Error: {e}")
        st.stop()

    uploaded_file = st.file_uploader("Choose a fingerprint image...", type=["bmp", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Read the uploaded image into memory
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        st.image(image, caption='Uploaded Fingerprint', width=250)

        if st.button("Authenticate", type="primary"):
            with st.spinner("Processing scan and searching database..."):
                try:
                    # The preprocessor currently expects a file path, so we save a temporary file
                    temp_path = "temp_scan.bmp"
                    cv2.imwrite(temp_path, image)

                    processed_img = preprocessor.preprocess(temp_path)
                    tensor_img = transform(processed_img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        embedding = model(tensor_img).cpu().numpy()

                    os.remove(temp_path)  # Clean up temp file

                    # Search the database for the closest match
                    best_match_id = None
                    best_distance = float('inf')

                    for user_id, user_embedding in enrollment_db.items():
                        dist = np.linalg.norm(embedding - user_embedding)
                        if dist < best_distance:
                            best_distance = dist
                            best_match_id = user_id

                    # Authorization Decision
                    if best_distance <= MATCH_THRESHOLD:
                        st.success(f"**Access Granted!** Authorized as User: **{best_match_id[0]}** (Finger: {best_match_id[1]})")
                        st.info(f"Match Distance: {best_distance:.4f} (Threshold: {MATCH_THRESHOLD})")
                    else:
                        st.error("**Access Denied!** Fingerprint not recognized.")
                        st.warning(f"Closest match distance: {best_distance:.4f} (Threshold: {MATCH_THRESHOLD})")

                except Exception as e:
                    st.error(f"An error occurred during authentication: {e}")


if __name__ == '__main__':
    main()
