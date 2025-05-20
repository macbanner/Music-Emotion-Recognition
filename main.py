import argparse
import os
import json

import numpy as np
import tensorflow as tf

from audio_manager import AudioManager
from augmentor import Augmentor
from data_manager import DataManager
from dataset_manager import DatasetManager
from model_manager import ModelManager


class MusicEmotionPipeline:
    """
    Pipeline class to execute the music emotion recognition workflow.
    """

    def __init__(self, dataset_path, data_csv_path, emotion_columns, segment_length=3):
        self.dataset_path = dataset_path
        self.data_csv_path = data_csv_path
        self.emotion_columns = emotion_columns
        self.segment_length = segment_length

        # Instantiate the modules
        self.data_manager = DataManager(data_csv_path, emotion_columns)
        self.audio_manager = AudioManager(dataset_path)
        self.dataset_manager = DatasetManager(emotion_columns)
        self.augmentor = Augmentor(emotion_columns)

    def run(self):
        """Runs the complete music emotion recognition pipeline."""
        print("Starting music emotion recognition pipeline...")

        # Load and reformat CSV data
        data_subset = self.data_manager.reformat_data()
        if data_subset is None:
            print("Failed to load data. Exiting.")
            return
        print(f"Data loaded successfully. Shape: {data_subset.shape}")

        # Aggregate data and print track-level distribution
        aggregated_data = self.data_manager.aggregate_data(data_subset)
        save_path = os.path.join(self.dataset_path, "Aggregated_Data.csv")
        aggregated_data.to_csv(save_path, index=False)
        print(f"Aggregated data shape: {aggregated_data.shape}")
        self.data_manager.print_class_distribution("Aggregated Track-level Labels",
                                                   aggregated_data[self.emotion_columns].values)

        # Extract audio segments and corresponding labels
        audio_path = os.path.join(self.dataset_path, "Musics")
        X_segments, y_labels, track_ids = self.audio_manager.extract_audio_segments_and_labels(
            audio_path, aggregated_data, self.emotion_columns, segment_length=self.segment_length
        )
        if X_segments is None or y_labels is None:
            print("Failed to extract audio segments. Exiting.")
            return
        print(f"Audio segments extracted: {X_segments.shape}, Labels shape: {y_labels.shape}")
        self.data_manager.print_class_distribution("All Audio Segments", y_labels)

        # Extract features from the audio segments
        features = self.audio_manager.extract_features_GRU(X_segments, augment=False)
        print(f"Features extracted. Shape: {features.shape}")

        # Split data into training, validation, and test sets
        X_train, X_val, X_test, y_train, y_val, y_test = self.dataset_manager.split_data_by_track(
            features, y_labels, track_ids, train_size=0.7, val_size=0.15, test_size=0.15
        )
        self.data_manager.print_class_distribution("Training Set", y_train)
        self.data_manager.print_class_distribution("Validation Set", y_val)
        self.data_manager.print_class_distribution("Test Set", y_test)

        # Augment underrepresented classes using DataManager's print function for class distribution
        X_train_aug, y_train_aug = self.augmentor.augment_underrepresented_labels(
            X_train, y_train, target_ratio=0.4, print_func=self.data_manager.print_class_distribution
        )

        # Preprocess the dataset (standardization and reshaping for CNN)
        X_train_proc, X_val_proc, X_test_proc = self.dataset_manager.preprocess_data(X_train_aug, X_val, X_test)
        print(f"Preprocessed data shapes: {X_train_proc.shape}, {X_val_proc.shape}, {X_test_proc.shape}")

        # Build, train, and evaluate the model
        input_shape = X_train_proc.shape[1:]
        num_emotions = y_train_aug.shape[1]
        model_manager = ModelManager(num_emotions, self.emotion_columns)
        class_weights_array = model_manager.compute_class_weights(y_train_aug)
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights_array)}
        print("Computed Class Weights:", class_weights_dict)  # Optional: print weights
        model = model_manager.build_cnn_gru_model(input_shape, class_weights_dict, dropout_rate=0.6, learning_rate=0.00001, recurrent_units=64)
        history = model_manager.train_model(model, X_train_proc, y_train_aug, X_val_proc, y_val,
                                            class_weights_dict=class_weights_dict,  # Pass the dict here
                                            epochs=50, batch_size=32, patience=10)

        # Determine best thresholds and evaluate on the test set
        y_val_pred_prob = model.predict(X_val_proc)
        best_thresholds = model_manager.find_best_thresholds(y_val, y_val_pred_prob)
        print("Best thresholds per emotion:", best_thresholds)
        metrics, used_thresholds = model_manager.evaluate_model(model, X_test_proc, y_test, thresholds=best_thresholds)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.3f}")

        # Save the trained model, training history, and evaluation metrics
        model.save("music_emotion_model.keras")
        with open("Music_emotion_history.json", "w") as f:
            json.dump(history.history, f)
        with open("Music_emotion_metrics.json", "w") as f:
            json.dump(metrics, f)
        print("Model, history, and metrics saved successfully.")

        return model, history, metrics


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
DATASET_PATH = "Music Data"
DATA_CSV_PATH = os.path.join(DATASET_PATH, "data.csv")
EMOTION_COLUMNS = [
    "amazement", "solemnity", "tenderness", "nostalgia",
    "calmness", "power", "joyful_activation", "tension", "sadness"
]
SEGMENT_LENGTH = 3
MODEL_PATH = "music_emotion_model.keras"
HISTORY_PATH = "Music_emotion_history.json"
METRICS_PATH = "Music_emotion_metrics.json"

# ──────────────────────────────────────────────────────────────────────────────
def train_pipeline():
    from main import MusicEmotionPipeline
    pipeline = MusicEmotionPipeline(
        DATASET_PATH,
        DATA_CSV_PATH,
        EMOTION_COLUMNS,
        segment_length=SEGMENT_LENGTH
    )
    pipeline.run()
    # pipeline.run() already saves model, history & metrics
    print("Training complete. Model, history, and metrics have been saved.")

# ──────────────────────────────────────────────────────────────────────────────
def evaluate_pipeline():
    # Load and reformat CSV → track‐level labels
    dm = DataManager(DATA_CSV_PATH, EMOTION_COLUMNS)
    data_subset = dm.reformat_data()
    agg = dm.aggregate_data(data_subset)

    # Extract segments & labels
    am = AudioManager(DATASET_PATH)
    X_seg, y_lbl, track_ids = am.extract_audio_segments_and_labels(
        os.path.join(DATASET_PATH, "Musics"),
        agg,
        EMOTION_COLUMNS,
        segment_length=SEGMENT_LENGTH
    )

    # Extract features (no augmentation)
    features = am.extract_features_GRU(X_seg, augment=False)

    # Split by track
    dsm = DatasetManager(EMOTION_COLUMNS)
    X_train, X_val, X_test, y_train, y_val, y_test = dsm.split_data_by_track(
        features, y_lbl, track_ids,
        train_size=0.7, val_size=0.15, test_size=0.15
    )

    # Preprocess
    X_train_p, X_val_p, X_test_p = dsm.preprocess_data(X_train, X_val, X_test)

    # Load your saved model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")

    # Find best thresholds on validation set
    mm = ModelManager(len(EMOTION_COLUMNS), EMOTION_COLUMNS)
    y_val_prob = model.predict(X_val_p)
    best_thresholds = mm.find_best_thresholds(y_val, y_val_prob)
    print("Best thresholds:", best_thresholds)

    # Evaluate on test set
    metrics, used_thresholds = mm.evaluate_model(model, X_test_p, y_test, thresholds=best_thresholds)
    print(json.dumps(metrics, indent=2))

    out_dir = "test_data"
    os.makedirs(out_dir, exist_ok=True)

    # Prepare file path
    fp = os.path.join(out_dir, "test_data.npz")

    # Collect arrays
    X_test_proc = X_test_p
    y_pred_prob = model.predict(X_test_proc)
    cols = np.array(EMOTION_COLUMNS)
    ths = np.array(used_thresholds)

    # Save into that folder
    np.savez(
        fp,
        X_test_proc=X_test_proc,
        y_test=y_test,
        emotion_cols=cols,
        y_pred_prob=y_pred_prob,
        thresholds=ths,
    )
    print(f"→ Saved all test data to '{fp}'")

    # (Re)save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)
    print(f"Test metrics written to {METRICS_PATH}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_pipeline()


