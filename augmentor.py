import numpy as np
from imblearn.over_sampling import SMOTE


class Augmentor:
    """
    Class for augmenting the training dataset for underrepresented emotion classes using SMOTE.
    """

    def __init__(self, emotion_columns):
        self.emotion_columns = emotion_columns

    def augment_underrepresented_labels(self, X_train, y_train, target_ratio=0.3, print_func=None):
        """
        Apply SMOTE to balance underrepresented classes in the dataset.

        Parameters:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            target_ratio (float): Target ratio for minority classes relative to majority.
            print_func (callable, optional): Function to print class distributions.
                                             Expected signature: print_func(description, y_labels)

        Returns:
            tuple: (X_resampled, y_resampled)
        """
        if print_func:
            print_func("Before SMOTE", y_train)
        else:
            print("Before SMOTE - class distribution (default print):", np.sum(y_train, axis=0))

        original_shape = X_train.shape
        X_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_combined = X_reshaped.copy()
        y_combined = y_train.copy()

        for i in range(y_train.shape[1]):
            positive_indices = np.where(y_train[:, i] == 1)[0]
            negative_indices = np.where(y_train[:, i] == 0)[0]
            current_ratio = (
                len(positive_indices) / len(negative_indices)
                if len(negative_indices) > 0
                else 1.0
            )

            if len(positive_indices) < 5:
                print(
                    f"Skipping emotion {self.emotion_columns[i]} - not enough positive samples (minimum 5 required)"
                )
                continue

            if current_ratio < target_ratio:
                print(f"Applying SMOTE for emotion {self.emotion_columns[i]}")
                print(
                    f"  Current ratio: {current_ratio:.3f} (Target: {target_ratio:.3f})\n"
                    f"  Current counts: {len(positive_indices)} positive, {len(negative_indices)} negative"
                )
                X_emotion = X_reshaped.copy()
                y_emotion = y_train[:, i].copy()
                target_samples = int(len(negative_indices) * target_ratio)
                samples_to_add = target_samples - len(positive_indices)
                print(f"  Target samples: {target_samples}, Will add: {samples_to_add}")

                sampling_strategy = {0: len(negative_indices), 1: target_samples}

                try:
                    k_neighbors = min(5, len(positive_indices) - 1)
                    smote = SMOTE(
                        sampling_strategy=sampling_strategy,
                        k_neighbors=k_neighbors,
                        random_state=42,
                    )
                    X_resampled, y_emotion_resampled = smote.fit_resample(
                        X_emotion, y_emotion
                    )
                    new_samples_mask = np.ones(len(X_resampled), dtype=bool)
                    new_samples_mask[: len(X_emotion)] = False
                    X_new = X_resampled[new_samples_mask]

                    y_new = np.zeros((X_new.shape[0], y_train.shape[1]))
                    y_new[:, i] = 1

                    from sklearn.neighbors import NearestNeighbors

                    nn = NearestNeighbors(n_neighbors=min(3, len(positive_indices)))
                    nn.fit(X_reshaped[positive_indices])
                    for j in range(X_new.shape[0]):
                        neighbors = nn.kneighbors(X_new[j].reshape(1, -1), return_distance=False)[0]
                        for k in range(y_train.shape[1]):
                            if k != i:
                                neighbor_labels = y_train[positive_indices[neighbors], k]
                                y_new[j, k] = 1 if np.mean(neighbor_labels) >= 0.5 else 0

                    X_combined = np.vstack([X_combined, X_new])
                    y_combined = np.vstack([y_combined, y_new])
                    print(f"  Added {len(X_new)} synthetic samples for emotion {self.emotion_columns[i]}")

                except Exception as e:
                    print(f"  Error applying SMOTE for emotion {self.emotion_columns[i]}: {e}")
                    continue
            else:
                print(
                    f"Skipping emotion {self.emotion_columns[i]} - ratio already sufficient ({current_ratio:.3f} >= {target_ratio:.3f})"
                )

        X_resampled = X_combined.reshape(-1, *original_shape[1:])
        print(f"Final dataset shape after oversampling: {X_resampled.shape}, Labels: {y_combined.shape}")
        if print_func:
            print_func("After SMOTE", y_combined)

        return X_resampled, y_combined
