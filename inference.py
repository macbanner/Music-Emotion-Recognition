import argparse
import numpy as np
import tensorflow as tf

from audio_manager import AudioManager
from model_manager import ModelManager

# Your emotion names
EMOTIONS = [
    "amazement", "solemnity", "tenderness", "nostalgia",
    "calmness", "power", "joyful_activation", "tension", "sadness"
]

DEFAULT_AUDIO = r"deneme\Guns N' Roses - Don't Cry.mp3"
DEFAULT_MODEL_PATH = {
    "cnn":      r"test_data\cnn\CNN_best_model.keras",
    "cnn_gru":  r"test_data\gru\GRU_best_model.keras"
}
DEFAULT_THRESH_FILE = {
    "cnn":      r"test_data\cnn\CNN_test_data.npz",
    "cnn_gru":  r"test_data\gru\GRU_test_data.npz"
}

# Your emotion labels, in the same order you trained/validated
EMOTIONS = [
    "amazement", "solemnity", "tenderness", "nostalgia",
    "calmness", "power", "joyful_activation", "tension", "sadness"
]

def load_model(model_type: str, model_path: str):
    """
    model_type: 'cnn' or 'cnn_gru' (for clarity only)
    model_path: path to .keras file
    """
    model = tf.keras.models.load_model(model_path)
    return model

def infer_track(audio_path: str,
                model_type: str,
                model_path: str,
                thresholds: np.ndarray = None,
                segment_length: int = 3):
    """
    Runs inference on one track. Returns dict of emotion→(prob, pred_label).
    """
    # 1) Segment the file
    am       = AudioManager(dataset_path=None)
    segments = am.extract_segments_from_file(audio_path, segment_length=segment_length)

    # 2) Feature extraction
    if model_type == "cnn":
        feats = am.extract_features_CNN(segments, augment=False)
    else:
        feats = am.extract_features_GRU(segments, augment=False)

    # 3) Normalize per‐track
    mu = feats.mean(axis=(0, 2), keepdims=True)
    sigma = feats.std(axis=(0, 2), keepdims=True) + 1e-6
    X = ((feats - mu) / sigma)[..., np.newaxis]

    # 4) Add channel dimension and predict
    model      = tf.keras.models.load_model(model_path)
    all_probs  = model.predict(X, verbose=0)      # shape (n_segments, n_emotions)
    track_probs = all_probs.mean(axis=0)          # shape (n_emotions,)

    # 5) Pick thresholds
    if thresholds is None:
        thresholds = np.full_like(track_probs, 0.5)

    # 6) Multi‐label decision: all emotions above threshold
    labels = (track_probs >= thresholds).astype(int)
    predicted = [emo for i, emo in enumerate(EMOTIONS) if labels[i]]

    # 7) Pack results
    return {
        "probabilities": {emo: float(track_probs[i]) for i, emo in enumerate(EMOTIONS)},
        "predicted_emotions": predicted
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infer emotions on one track (multi-label) using CNN or CNN+GRU."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Path to local file or URL (YouTube/Spotify)."
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "cnn_gru"],
        default="cnn_gru",
        help="Which pretrained model to use."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to your saved .keras model."
    )
    parser.add_argument(
        "--thresholds-file",
        default=None,
        help="Path to your saved .npz with thresholds + columns."
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=3,
        help="Segment length in seconds."
    )
    args = parser.parse_args()

    # Fill in PyCharm defaults if no CLI args provided
    if args.audio is None:
        args.audio = DEFAULT_AUDIO
    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATH[args.model]
    if args.thresholds_file is None:
        args.thresholds_file = DEFAULT_THRESH_FILE[args.model]

    # Load thresholds (and override EMOTIONS order if provided)
    data = np.load(args.thresholds_file, allow_pickle=True)
    thresholds = data["thresholds"]
    if "emotion_cols" in data:
        EMOTIONS[:] = list(data["emotion_cols"])

    # Run inference
    results = infer_track(
        audio_path=args.audio,
        model_type=args.model,
        model_path=args.model_path,
        thresholds=thresholds,
        segment_length=args.segment_length
    )

    # Print results
    print("\n=== Detected emotions ===")
    if results["predicted_emotions"]:
        print("→", ", ".join(results["predicted_emotions"]))
    else:
        print("→ None above threshold")

    print("\n=== Probabilities ===")
    for emo, prob in results["probabilities"].items():
        print(f"{emo:20}: {prob:.3f}")
