# Music Emotion Recognition System

## Overview
This repository contains a comprehensive system for music emotion recognition (MER) using deep learning. The pipeline processes audio files (using the Emotify dataset as a base), extracts acoustic features (Mel Spectrograms, MFCCs, Spectral Contrast), balances the dataset using SMOTE, trains a Convolutional Neural Network (CNN), and predicts multiple emotion labels for music tracks. The system includes extensive evaluation and visualization capabilities.

## Dataset
This project is demonstrated using the Emotify dataset, which contains music tracks with emotion annotations across multiple genres (classical, rock, electronic, pop). The 9 target emotions are:
- Amazement
- Solemnity
- Tenderness
- Nostalgia
- Calmness
- Power
- Joyful Activation
- Tension
- Sadness

## Repository Structure

- `audio_manager.py`: Audio file operations including reformatting, segmentation, and feature extraction  
- `augmentor.py`: Class balancing using SMOTE to address underrepresented emotions  
- `data_manager.py`: CSV data operations, including reformatting and aggregating emotion annotations  
- `dataset_manager.py`: Dataset splitting (track‑aware) and preprocessing while maintaining track integrity  
- `model_manager.py`: CNN model building, training, evaluation, and threshold finding  
- `utils.py`: Utility functions (e.g., natural sorting)  
- `main.py`: Main pipeline that orchestrates the entire workflow with **train** and **eval** modes  
- `visualization.ipynb`: Jupyter notebook for detailed model analysis, performance visualization, and error inspection  
- `requirements.txt`: List of required Python packages for the project  
- `music_emotion_model.keras`: Saved trained Keras model (generated after training)  
- `Music_emotion_history.json`: Training‑history logs (loss and metrics per epoch) saved as JSON (generated after training)  
- `Music_emotion_metrics.json`: Evaluation metrics on the test set saved as JSON (generated after training or evaluation)  
- `Music Data/`: Folder containing the input dataset (genre subfolders with audio files and `data.csv`)  
- `Musics/`: Reformatted audio files (e.g., `1.mp3`, `2.mp3`) created by `audio_manager.py`  
- `data.csv`: Raw annotation data file  
- `test_data/`: Folder where test‑set data and predictions are saved during evaluation  
- `test_data.npz`: NumPy archive containing test features, true labels, predicted probabilities, and thresholds  


## Performance Metrics
Detailed evaluation metrics are saved in `Music_emotion_metrics.json` after running the pipeline. Key metrics from a sample run:

- **Overall F1 Score (macro):** ~0.885
- **Hamming Loss:** ~0.053

![image](https://github.com/user-attachments/assets/03eb8155-0c4d-4b5e-9416-b10773f25431)


Per-emotion performance varies. See `Music_emotion_metrics.json` or the `visualization.ipynb` notebook for a full breakdown. Generally, emotions like Amazement, Tenderness, Calmness, Power, Joyful Activation, and Tension show high F1 scores (often > 0.90), while Solemnity and Sadness can be more challenging.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/adogan/Music-Emotion-Recognition.git
    cd music-emotion-recognition
    ```

2.  **Set up a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional - GPU Support):** If you have an NVIDIA GPU and want to use it with TensorFlow, ensure you have the compatible CUDA Toolkit and cuDNN library installed. See the [TensorFlow GPU documentation](https://www.tensorflow.org/install/gpu) for details.

5.  **Dataset:** Place your dataset (audio genre folders and `data.csv`) inside a `Music Data` directory in the project root, or update the `DATASET_PATH` and `DATA_CSV_PATH` variables in `main.py`.

## Usage

The `main.py` script orchestrates the pipeline and can be run in two modes:

1.  **Train a new model:**
    This will execute the full pipeline: data loading, preprocessing, feature extraction, augmentation, training, evaluation, and saving the model (`music_emotion_model.keras`), history (`Music_emotion_history.json`), and metrics (`Music_emotion_metrics.json`).
    ```bash
    python main.py train
    ```

2.  **Evaluate an existing model:**
    This loads the saved model (`music_emotion_model.keras`), re-runs the data preparation steps (necessary to get the test set), finds optimal thresholds on the validation set, evaluates the model on the test set, prints the metrics, saves them to `Music_emotion_metrics.json`, and saves test data/predictions to `test_data/test_data.npz`.
    ```bash
    python main.py eval
    ```

## Visualization and Analysis (`visualization.ipynb`)

A detailed Jupyter Notebook (`visualization.ipynb`) is provided for visualization of the model's performance and behavior. Run this notebook *after* executing `main.py train` or `main.py eval` to load the generated data.

Key visualizations included:

*   **Training & Validation Curves:** Monitor loss and metrics (Accuracy, AUC, Precision, Recall) over epochs to diagnose overfitting or underfitting.

    ![image](https://github.com/user-attachments/assets/08db5c09-f7f1-4dc3-9df3-83565bd6ea42)


*   **Overall Metrics Table:** A summary table of key performance indicators like Hamming Loss, F1 scores (micro, macro, sample), Precision, and Recall.
*   **Per-Emotion Performance Bars:** Bar charts comparing F1, Precision, and Recall for each of the 9 emotions, highlighting model strengths and weaknesses.

    ![image](https://github.com/user-attachments/assets/b19d604a-ad5e-4530-bee5-3328067962c0)

*   **Confusion Matrix:** A heatmap showing class-wise confusion (averaged over samples or per-emotion based on implementation) to understand misclassification patterns.

    ![image](https://github.com/user-attachments/assets/35868a8f-93bf-4e31-9548-ff62285b6b01)

*   **ROC & Precision-Recall Curves:** Per-emotion curves illustrating the trade-off between true positive rate and false positive rate (ROC) and between precision and recall. Useful for evaluating performance, especially on imbalanced classes.
*   **Calibration Curves:** Assess how well the predicted probabilities reflect the true likelihood of emotions.
*   **Threshold Sweep Analysis:** Visualize how metrics like F1-score change as the prediction threshold varies for each emotion.
*   **Embeddings Visualization:** Project high-dimensional feature representations (from intermediate layers or final predictions) into 2D or 3D using t-SNE, PCA, or UMAP to explore data structure and class separability.

    ![image](https://github.com/user-attachments/assets/041b9476-d086-4499-afcc-7b33094e57fe)

*   **Error Analysis Dashboard:** Tools like confusion heatmaps specifically for errors, and identification/display of the "hardest" examples (those the model gets wrong with high confidence) to guide further model improvement.

## Model Architecture
The system uses a Convolutional Neural Network (CNN). Key components include:
- Multiple `Conv2D` layers with ReLU activations and L2 regularization.
- `MaxPooling2D` layers for down-sampling.
- `BatchNormalization` for stabilizing and accelerating training.
- `Dropout` for regularization.
- A `Flatten` layer followed by `Dense` layers.
- A final `Dense` output layer with `sigmoid` activation for multi-label classification.
- Compiled with Adam optimizer and `binary_crossentropy` loss.

## Results and Analysis Summary
The model generally performs well, achieving high F1 scores for several emotions. Detailed results are available in `Music_emotion_metrics.json` and the `visualization.ipynb` notebook.

- **Strong performers:** Typically Amazement, Tenderness, Calmness, Power, Joyful Activation, Tension.
- **Areas for improvement:** Often Solemnity and Sadness, which may benefit from more targeted feature engineering, different augmentation strategies, or model architecture adjustments.

The visualization notebook provides tools to delve deeper into these results.

## Dataset Requirements
- Audio files (.mp3 recommended, but Librosa supports others) organized in genre subdirectories within the main dataset path (`Music Data/` by default).
- A CSV file (`data.csv` by default) in the dataset path containing annotations. It must include columns: `track id`, `genre`, and the 9 specified `EMOTION_COLUMNS` (e.g., `amazement`, `solemnity`, ...). The `track id` should correspond to the base number of the audio files after reformatting (e.g., `1`, `2`, ... for `1.mp3`, `2.mp3`).

## License
This project is licensed under the [MIT License](LICENSE).

## Citation
If you use **Music Emotion Recognition System** or the underlying Emotify dataset in academic work, please cite:

> **A. Aljanaki, F. Wiering, R. C. Veltkamp**  
> *Studying emotion induced by music through a crowdsourcing game.*  
> Information Processing & Management, 52 (1): 115‑128, 2016.  
> https://doi.org/10.1016/j.ipm.2015.03.004
