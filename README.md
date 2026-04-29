# Fingerprint Scanning & Authorization System

This project is a deep learning-based fingerprint recognition and authorization system. It utilizes a **Siamese Neural Network** built on top of a pre-trained **MobileNetV2** backbone to generate robust fingerprint embeddings. The model learns to map fingerprints from the same user close together while pushing fingerprints from different users apart using **Contrastive Loss**.

The system includes a complete pipeline for data preprocessing, training, user enrollment, evaluation (EER, FAR, FRR), and a Streamlit-based web application for real-time authentication.

## Project Structure

*   **`app/`**: Contains the Streamlit web application (`app.py`) for the user-facing fingerprint authentication interface.
*   **`data/`**: Handles dataset loading, splitting (`data_loader.py`), and the PyTorch Dataset class (`dataset.py`) for generating positive and negative Siamese pairs on-the-fly.
*   **`models/`**: Defines the neural network architectures (`model.py`), including the `EmbeddingNet` and the `ContrastiveLoss` function.
*   **`preprocessing/`**: Contains `preprocess.py`, responsible for image standardization (Grayscale conversion, CLAHE normalization, and resizing).
*   **`scripts/`**: 
    *   `train.py`: The main training loop with Automatic Mixed Precision (AMP) and Early Stopping.
    *   `enrollment.py`: Generates the master template database (`enrollment_db.pkl`) for authorized users.
    *   `validate.py`: Evaluates the model against Genuine and Impostor benchmarks, calculating the Equal Error Rate (EER) and optimal threshold.
    *   `plot_results.py`: Generates ROC curves, distance distributions, and learning curves.
*   **`results/`**: Output directory for trained models, enrollment databases, validation pickles, and plots.

## Dataset

This project is built around the **SOCOFing** (Sokoto Coventry Fingerprint Dataset). It contains real fingerprint scans alongside synthetically altered versions (Easy, Medium, Hard alterations like obliteration, central rotation, and z-cuts).

### Dataset License and Usage

The **SOCOFing** dataset is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license. It is provided strictly for **non-commercial, not-for-profit academic research purposes**.

If you use this dataset or this project in any publications, technical reports, or manuals, you are required to cite the original dataset authors:
> Shehu, Y.I., Ruiz-Garcia, A., Palade, V., James, A. (2018) "Detection of Fingerprint Alterations Using Deep Convolutional Neural Networks" in Proceedings of the International Conference on Artificial Neural Networks (ICANN 2018), Rhodes – Greece.

## Setup and Installation

1.  **Clone the repository** and ensure you have Python 3.8+ installed.
2.  **Install dependencies** (PyTorch, OpenCV, Scikit-learn, Pandas, Matplotlib, Streamlit, python-dotenv).
3.  **Environment Variables**: 
    *   Copy the `.env.sample` file and rename it to `.env`.
    *   Update the `SOCOFING_DATASET_PATH` to point to the root directory of your downloaded SOCOFing dataset.

## Pipeline & Usage

The project is designed to be run sequentially.

### 1. Training the Model
Train the Siamese network to learn fingerprint embeddings. The script features caching options, Mixed Precision (AMP) for speed, and generates a `.pth` model checkpoint.

```bash
python scripts/train.py
```
*Note: Training heavily utilizes the GPU. If you encounter memory issues, lower the `BATCH_SIZE` in `train.py`.*

### 2. Enrolling Users
Once the model is trained, you need to enroll authorized users. This script processes a subset of genuine users and creates an aggregated, L2-normalized master template for each finger, saving it to `results/enrollments/enrollment_db.pkl`.

```bash
python scripts/enrollment.py
```

### 3. Validating the System
To determine how accurate the system is, run the validation script. It tests the model against completely unseen Genuine (same user) and Impostor (different user) pairs.

```bash
python scripts/validate.py
```
This script will output the **Equal Error Rate (EER)** and determine the optimal authorization threshold. It also reports the False Acceptance Rate (FAR) at strict standard False Rejection Rate (FRR) benchmarks (e.g., 0.1%).

### 4. Visualizing Results
After validating, you can generate visual plots (Learning Curves, Distance Distributions, ROC Curves) to analyze model performance.

```bash
python scripts/plot_results.py
```
*Plots will be saved to the `results/plots` directory.*

### 5. Running the Web App
Launch the Streamlit interface to test the model interactively. You can upload a `.BMP` fingerprint scan, and the app will process the image, extract its embedding, compare it against the `enrollment_db.pkl`, and grant or deny access based on the EER threshold.

```bash
streamlit run app/app.py
```

## Technical Highlights

*   **CLAHE Preprocessing**: Contrast Limited Adaptive Histogram Equalization is applied to enhance the visibility of fingerprint minutiae (ridges and bifurcations).
*   **MobileNetV2 Backbone**: A lightweight, highly efficient CNN architecture fine-tuned specifically for feature extraction.
*   **Contrastive Loss**: Directly optimizes the Euclidean distance between embedding vectors.
*   **Robust Enrollment**: Master templates are created by averaging multiple authorized scans, resulting in a more stable vector resilient to minor cuts or sensor noise.