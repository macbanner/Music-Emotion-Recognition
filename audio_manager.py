import os
import shutil
import librosa
import numpy as np
from utils import natural_key
from tqdm import tqdm


class AudioManager:
    """
    Class for managing audio file operations such as reformatting, segmentation,
    and feature extraction.
    """

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def reformat_music_files(self):
        """
        Copies and renames music files from genre subdirectories into a unified directory.
        """
        track_id = 1
        copy_path = os.path.join(self.dataset_path, "Musics")
        os.makedirs(copy_path, exist_ok=True)

        genre_list = ["classical", "rock", "electronic", "pop"]
        for genre in genre_list:
            genre_path = os.path.join(self.dataset_path, genre)

            # Iterate through the files in the genre directory and copy with new names
            files = sorted(os.listdir(genre_path), key=natural_key)
            for file in files:
                source_file = os.path.join(genre_path, file)
                if os.path.isfile(source_file):
                    new_name = f"{track_id}.mp3"
                    new_file_path = os.path.join(copy_path, new_name)
                    shutil.copyfile(source_file, new_file_path)
                    track_id += 1

    def extract_audio_segments_and_labels(self, audio_path, aggregated_data, emotion_columns, sr=44100,
                                          segment_length=3):
        X_segments = []
        y_labels = []
        track_ids = []
        segment_samples = segment_length * sr
        n_fft = 2048
        hop_length = 512

        try:
            track_id = 1
            audios = sorted(os.listdir(audio_path), key=natural_key)
            print(f"Extracting segments from {len(audios)} audio files...")
            # Store labels temporarily to avoid repeated lookups
            track_label_cache = {}

            for audio in tqdm(audios, desc="Processing Audio Files"):
                file_path = os.path.join(audio_path, audio)
                try:
                    y_audio, _ = librosa.load(file_path, sr=sr, mono=True)
                except Exception as load_e:
                    print(f"Warning: Could not load {file_path}: {load_e}. Skipping.")
                    track_id += 1
                    continue

                # Check for labels *before* segmenting to potentially skip files faster
                if track_id not in track_label_cache:
                    if track_id in aggregated_data["track id"].values:
                        track_labels = aggregated_data[aggregated_data["track id"] == track_id][emotion_columns].values[0]
                        track_label_cache[track_id] = track_labels
                    else:
                        # Mark as having no labels found to avoid future lookups for this ID
                        track_label_cache[track_id] = None
                        # print(f"Warning: No labels found for track ID {track_id}") # Optional warning

                # If no labels found for this track, skip its segmentation
                if track_label_cache[track_id] is None:
                    track_id += 1
                    continue

                current_track_labels = track_label_cache[track_id]

                # Split audio into segments
                for i in range(0, len(y_audio) - segment_samples + 1, segment_samples):
                    segment = y_audio[i : i + segment_samples]
                    if len(segment) == segment_samples:
                        X_segments.append(segment)
                        y_labels.append(current_track_labels) # Append the cached label
                        track_ids.append(track_id)    # Append the track id

                track_id += 1 # Increment track_id after processing the file

            if not X_segments:
                 print("Warning: No valid segments found or matched with labels.")
                 return None, None, None

            X_segments = np.array(X_segments)
            y_labels = np.array(y_labels)
            track_ids = np.array(track_ids)

            print(f"Finished segment extraction. Found {len(X_segments)} segments.")
            return X_segments, y_labels, track_ids
        except Exception as e:
            print(f"Error during segment extraction: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def extract_segments_from_file(self,
                                   audio_path: str,
                                   sr: int = 44100,
                                   segment_length: int = 3):
        """
        Load one audio file and split into fixed-length segments.

        Returns:
            np.ndarray of shape (n_segments, segment_samples)
        """
        segment_samples = segment_length * sr
        # load the full track
        y_audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        segments = []
        for start in range(0, len(y_audio) - segment_samples + 1, segment_samples):
            seg = y_audio[start:start + segment_samples]
            if len(seg) == segment_samples:
                segments.append(seg)
        if not segments:
            raise ValueError(f"No {segment_length}s segments could be extracted from {audio_path}")
        return np.stack(segments, axis=0)


    def extract_features_GRU(self, audio_segments, sr=44100, n_mels=128, n_mfcc=20, n_chroma=12, augment=False):
        """
        Extract combined audio features from segments with optional augmentation.
        Includes Mel Spec, MFCC, Delta MFCC, Spectral Contrast, RMSE, ZCR, and Chroma.

        Parameters:
            audio_segments (list or np.ndarray): Audio segments.
            sr (int): Sampling rate.
            n_mels (int): Number of mel bands.
            n_mfcc (int): Number of MFCC coefficients.
            n_chroma (int): Number of chroma bins (usually 12).
            augment (bool): Whether to perform pitch-shift augmentation.

        Returns:
            np.ndarray: Array of combined features. Shape: (num_segments, num_features, num_frames)
        """
        features = []
        n_fft = 2048
        hop_length = 512

        print(f"Extracting features (incl. Chroma) from {len(audio_segments)} segments...")
        for segment in tqdm(audio_segments, desc="Extracting Features"):
            try:
                # --- Base Spectral/Temporal Features ---
                mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=20, fmax=8000)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), sr=sr, n_mfcc=n_mfcc)
                delta_mfcc = librosa.feature.delta(mfcc)
                contrast = librosa.feature.spectral_contrast(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length)
                rmse = librosa.feature.rms(y=segment, frame_length=n_fft, hop_length=hop_length)[0]
                zcr = librosa.feature.zero_crossing_rate(y=segment, frame_length=n_fft, hop_length=hop_length)[0]

                # --- Harmonic Feature ---
                chroma = librosa.feature.chroma_stft(y=segment, sr=sr, n_chroma=n_chroma, n_fft=n_fft, hop_length=hop_length)

                # --- Alignment & Stacking ---
                min_frames = min(mel_spec_db.shape[1], mfcc.shape[1], delta_mfcc.shape[1], contrast.shape[1], rmse.shape[0], zcr.shape[0], chroma.shape[1]) # Add chroma here

                mel_spec_db = mel_spec_db[:, :min_frames]
                mfcc = mfcc[:, :min_frames]
                delta_mfcc = delta_mfcc[:, :min_frames]
                contrast = contrast[:, :min_frames]
                rmse = rmse[:min_frames].reshape(1, min_frames)
                zcr = zcr[:min_frames].reshape(1, min_frames)
                chroma = chroma[:, :min_frames] # Align chroma

                # Combine features vertically
                combined = np.vstack([mel_spec_db, mfcc, delta_mfcc, contrast, rmse, zcr, chroma]) # Add chroma here
                # Shape: (n_mels + n_mfcc*2 + contrast_bands + 1 + 1 + n_chroma, min_frames)

                features.append(combined)

                # --- Optional Augmentation ---
                if augment:
                    try:
                        segment_shifted = librosa.effects.pitch_shift(segment, sr=sr, n_steps=1)
                        # Recalculate ALL features for shifted audio
                        mel_spec_shifted = librosa.feature.melspectrogram(y=segment_shifted, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=20, fmax=8000)
                        mel_spec_db_shifted = librosa.power_to_db(mel_spec_shifted, ref=np.max)
                        mfcc_shifted = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec_shifted), sr=sr, n_mfcc=n_mfcc)
                        delta_mfcc_shifted = librosa.feature.delta(mfcc_shifted)
                        contrast_shifted = librosa.feature.spectral_contrast(y=segment_shifted, sr=sr, n_fft=n_fft, hop_length=hop_length)
                        rmse_shifted = librosa.feature.rms(y=segment_shifted, frame_length=n_fft, hop_length=hop_length)[0]
                        zcr_shifted = librosa.feature.zero_crossing_rate(y=segment_shifted, frame_length=n_fft, hop_length=hop_length)[0]
                        chroma_shifted = librosa.feature.chroma_stft(y=segment_shifted, sr=sr, n_chroma=n_chroma, n_fft=n_fft, hop_length=hop_length) # Add chroma

                        # Align & Stack Shifted Features
                        min_frames_shifted = min(mel_spec_db_shifted.shape[1], mfcc_shifted.shape[1], delta_mfcc_shifted.shape[1], contrast_shifted.shape[1], rmse_shifted.shape[0], zcr_shifted.shape[0], chroma_shifted.shape[1]) # Add chroma
                        mel_spec_db_shifted = mel_spec_db_shifted[:, :min_frames_shifted]
                        mfcc_shifted = mfcc_shifted[:, :min_frames_shifted]
                        delta_mfcc_shifted = delta_mfcc_shifted[:, :min_frames_shifted]
                        contrast_shifted = contrast_shifted[:, :min_frames_shifted]
                        rmse_shifted = rmse_shifted[:min_frames_shifted].reshape(1, min_frames_shifted)
                        zcr_shifted = zcr_shifted[:min_frames_shifted].reshape(1, min_frames_shifted)
                        chroma_shifted = chroma_shifted[:, :min_frames_shifted] # Align chroma

                        combined_shifted = np.vstack([mel_spec_db_shifted, mfcc_shifted, delta_mfcc_shifted, contrast_shifted, rmse_shifted, zcr_shifted, chroma_shifted]) # Add chroma

                        # Ensure consistency
                        final_min_frames = min(combined.shape[1], combined_shifted.shape[1])
                        features[-1] = combined[:, :final_min_frames]
                        features.append(combined_shifted[:, :final_min_frames])

                    except Exception as aug_e:
                        print(f"Warning: Error during augmentation for a segment: {aug_e}. Skipping augmentation.")
                        # Ensure original has original min_frames
                        if len(features) > 0 and features[-1].shape[1] != min_frames:
                             features[-1] = features[-1][:, :min_frames]


            except Exception as feat_e:
                print(f"Warning: Error extracting features for a segment: {feat_e}. Skipping.")
                continue

        if not features:
             print("Error: No features could be extracted.")
             return None

        # Final alignment across all segments
        if features:
            min_frames_overall = min(f.shape[1] for f in features)
            features = [f[:, :min_frames_overall] for f in features]
        else:
             print("Warning: No features extracted successfully.")
             return None


        print(f"Finished feature extraction. Final feature shape per segment: {features[0].shape}")
        return np.array(features)

    def extract_features_CNN(self, audio_segments, sr=44100, n_mels=128, augment=False):
        """
        Extract combined audio features from segments with optional augmentation.

        Parameters:
            audio_segments (list or np.ndarray): Audio segments.
            sr (int): Sampling rate.
            n_mels (int): Number of mel bands.
            augment (bool): Whether to perform pitch-shift augmentation.

        Returns:
            np.ndarray: Array of combined features.
        """
        try:
            features = []

            for segment in audio_segments:
                # Basic feature - mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=segment,
                    sr=sr,
                    n_mels=n_mels,
                    n_fft=2048,
                    hop_length=512,
                    fmin=20,
                    fmax=8000
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                # MFCC
                mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=20)

                # Spectral contrast
                contrast = librosa.feature.spectral_contrast(y=segment, sr=sr)

                # Combine features
                combined = np.vstack([mel_spec_db, mfcc, contrast])

                # Optional augmentation
                if augment:
                    # Pitch shift (mild)
                    segment_shifted = librosa.effects.pitch_shift(segment, sr=sr, n_steps=1)
                    mel_spec_shifted = librosa.feature.melspectrogram(
                        y=segment_shifted, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512
                    )
                    mel_spec_db_shifted = librosa.power_to_db(mel_spec_shifted, ref=np.max)
                    mfcc_shifted = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec_shifted), n_mfcc=20)
                    contrast_shifted = librosa.feature.spectral_contrast(y=segment_shifted, sr=sr)
                    combined_shifted = np.vstack([mel_spec_db_shifted, mfcc_shifted, contrast_shifted])

                    features.append(combined)
                    features.append(combined_shifted)
                else:
                    features.append(combined)

            return np.array(features)
        except Exception as e:
            print(f"Error in extract_features: {e}")
            return None


