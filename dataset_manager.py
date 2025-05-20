import numpy as np


class DatasetManager:
    """
    Class for splitting and preprocessing the dataset while ensuring
    segments from the same track remain together.
    """

    def __init__(self, emotion_columns):
        self.emotion_columns = emotion_columns

    def split_data_by_track(self, X_segments, y_labels, track_ids,
                            train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        """
        Splits data into training, validation, and test sets.

        Parameters:
            X_segments (np.ndarray): Audio segments or features.
            y_labels (np.ndarray): Corresponding emotion labels.
            track_ids (np.ndarray): Track IDs.
            train_size (float): Training split ratio.
            val_size (float): Validation split ratio.
            test_size (float): Test split ratio.
            random_state (int): Random seed.

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-10

        # Get unique tracks and their associated emotion labels
        unique_tracks = np.unique(track_ids)
        track_to_emotions = {}
        for track in unique_tracks:
            # Find indices where this track appears
            track_indices = np.where(track_ids == track)[0]
            # Get the emotion labels for this track (should all be the same)
            track_emotions = y_labels[track_indices[0]]
            track_to_emotions[track] = track_emotions

        # Initialize track lists for each split
        train_tracks = []
        val_tracks = []
        test_tracks = []
        np.random.seed(random_state)

        # For each emotion, ensure proportional representation
        for emotion_idx, emotion in enumerate(self.emotion_columns):
            # Find tracks that have this emotion
            positive_tracks = [
                track for track, emotions in track_to_emotions.items() if emotions[emotion_idx] == 1
            ]
            # Skip if no tracks have this emotion
            if not positive_tracks:
                print(f"Warning: No tracks found with emotion '{emotion}'")
                continue

            # Shuffle tracks with this emotion
            np.random.shuffle(positive_tracks)

            # Calculate split sizes
            n_tracks = len(positive_tracks)
            n_train = max(1, int(n_tracks * train_size))
            n_val = max(1, int(n_tracks * val_size))

            # Ensure at least 1 track in test if there are enough tracks
            if n_tracks > 2:  # We need at least 3 tracks to distribute across 3 splits
                # Adjust to ensure we have at least one track in each split
                if n_train + n_val >= n_tracks:
                    n_train = max(1, n_tracks - 2)
                    n_val = 1
                train_tracks.extend(positive_tracks[:n_train])
                val_tracks.extend(positive_tracks[n_train: n_train + n_val])
                test_tracks.extend(positive_tracks[n_train + n_val:])
            else:
                # If we have only 1-2 tracks, prioritize training data
                train_tracks.extend(positive_tracks)
                print(f"Warning: Only {n_tracks} tracks with emotion '{emotion}', all added to training")

        # Remove duplicates
        train_tracks = list(set(train_tracks))
        val_tracks = list(set(val_tracks))
        test_tracks = list(set(test_tracks))

        # Handle tracks without any of the target emotions
        remaining_tracks = [
            track
            for track in unique_tracks
            if track not in train_tracks and track not in val_tracks and track not in test_tracks
        ]

        # Distribute remaining tracks proportionally
        np.random.shuffle(remaining_tracks)
        n_remaining = len(remaining_tracks)
        n_train_remaining = int(n_remaining * train_size)
        n_val_remaining = int(n_remaining * val_size)

        train_tracks.extend(remaining_tracks[:n_train_remaining])
        val_tracks.extend(remaining_tracks[n_train_remaining: n_train_remaining + n_val_remaining])
        test_tracks.extend(remaining_tracks[n_train_remaining + n_val_remaining:])

        # Create masks for each split
        train_mask = np.isin(track_ids, train_tracks)
        val_mask = np.isin(track_ids, val_tracks)
        test_mask = np.isin(track_ids, test_tracks)

        X_train = X_segments[train_mask]
        y_train = y_labels[train_mask]
        X_val = X_segments[val_mask]
        y_val = y_labels[val_mask]
        X_test = X_segments[test_mask]
        y_test = y_labels[test_mask]

        # Print statistics
        print(f"Training set:   {X_train.shape[0]} samples from {len(train_tracks)} tracks")
        print(f"Validation set: {X_val.shape[0]} samples from {len(val_tracks)} tracks")
        print(f"Test set:       {X_test.shape[0]} samples from {len(test_tracks)} tracks")

        for i, emotion in enumerate(self.emotion_columns):
            train_dist = np.mean(y_train[:, i])
            val_dist = np.mean(y_val[:, i])
            test_dist = np.mean(y_test[:, i])
            total_dist = np.mean(y_labels[:, i])
            print(
                f"  {emotion:<20}: Total: {total_dist:.3f}, Train: {train_dist:.3f}, Val: {val_dist:.3f}, Test: {test_dist:.3f}"
            )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess_data(self, X_train, X_val, X_test):
        # X shapes are likely (num_samples, num_features, num_timesteps) before reshaping for CNN
        print(f"Raw shapes before preprocessing: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

        # Standardize each feature channel (row) independently
        # Calculate mean/std per feature channel ON TRAINING DATA ONLY
        # Axis 0 is samples, Axis 1 is features, Axis 2 is time
        mean_per_feature = np.mean(X_train, axis=(0, 2),
                                   keepdims=True)  # Mean across samples and time, keep feature dim
        std_per_feature = np.std(X_train, axis=(0, 2), keepdims=True)  # Std across samples and time, keep feature dim

        # Add epsilon to std to avoid division by zero
        std_per_feature += 1e-6

        print(f"Shape of mean/std per feature: {mean_per_feature.shape}")  # Should be (1, num_features, 1)

        # Apply to all sets
        X_train_norm = (X_train - mean_per_feature) / std_per_feature
        X_val_norm = (X_val - mean_per_feature) / std_per_feature
        X_test_norm = (X_test - mean_per_feature) / std_per_feature

        # Reshape for 2D CNN (adding channel dimension)
        # Input for CNN should be (batch, height, width, channels)
        # Your features are (num_samples, num_features, num_timesteps)
        # Let num_features be 'height', num_timesteps be 'width'
        X_train_proc = X_train_norm[..., np.newaxis]  # Add channel dim -> (samples, features, time, 1)
        X_val_proc = X_val_norm[..., np.newaxis]
        X_test_proc = X_test_norm[..., np.newaxis]

        print(
            f"Shapes after preprocessing and reshape: Train={X_train_proc.shape}, Val={X_val_proc.shape}, Test={X_test_proc.shape}")
        return X_train_proc, X_val_proc, X_test_proc
