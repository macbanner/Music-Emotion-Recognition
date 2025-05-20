import pandas as pd
import numpy as np


class DataManager:
    """
    Class for managing CSV data operations including reading, reformatting,
    aggregating emotion annotations, and printing class distributions.
    """

    def __init__(self, csv_path, emotion_columns):
        self.csv_path = csv_path
        self.emotion_columns = emotion_columns

    def reformat_data(self):
        """
        Reads and reformats the CSV data file.

        Returns:
            pd.DataFrame: DataFrame containing selected columns.
        """
        try:
            data_file = pd.read_csv(self.csv_path)

            # Specify the columns to use
            columns_to_use = [
                "track id",  # Music file identifier
                "genre",  # Genre of the music file
                "amazement",  # Emotion annotation 1
                "solemnity",  # Emotion annotation 2
                "tenderness",  # Emotion annotation 3
                "nostalgia",  # Emotion annotation 4
                "calmness",  # Emotion annotation 5
                "power",  # Emotion annotation 6
                "joyful_activation",  # Emotion annotation 7
                "tension",  # Emotion annotation 8
                "sadness",  # Emotion annotation 9
            ]
            # Strip any leading/trailing spaces in column names
            data_file.columns = data_file.columns.str.strip()

            # Create a new DataFrame with only the desired columns
            data_subset = data_file[columns_to_use]
            return data_subset
        except Exception as e:
            print(f"Error: {e}")
            return None

    def aggregate_data(self, data_subset):
        """
        Aggregates emotion annotations using dynamic thresholds.

        Parameters:
            data_subset (pd.DataFrame): DataFrame with raw emotion annotations.

        Returns:
            pd.DataFrame: Aggregated DataFrame with binary labels.
        """
        aggregated = data_subset.groupby(["track id", "genre"], as_index=False).mean()

        # Find optimal threshold for each emotion based on distribution
        thresholds = {}
        for emotion in self.emotion_columns:
            values = aggregated[emotion].values
            # Use median as threshold if distribution is skewed; otherwise, fixed value
            if np.std(values) > 0.2:
                threshold = np.median(values)
            else:
                threshold = 0.4
            thresholds[emotion] = threshold
            print(f"Emotion: {emotion}, Threshold: {threshold:.3f}")

        # Apply thresholds
        for emotion in self.emotion_columns:
            aggregated[emotion] = (aggregated[emotion] > thresholds[emotion]).astype(int)
        return aggregated

    def print_class_distribution(self, description, y_labels):
        """
        Prints detailed class distribution for each emotion label.

        Parameters:
            description (str): Context description.
            y_labels (np.ndarray): Array of emotion labels.
        """
        print(f"\n{description} Class Distribution:")
        total_samples = y_labels.shape[0]
        print(f"Total samples: {total_samples}")

        for i, emotion in enumerate(self.emotion_columns):
            positive = np.sum(y_labels[:, i])
            negative = total_samples - positive
            pos_ratio = positive / total_samples
            neg_ratio = negative / total_samples
            print(
                f"  {emotion:<20}: Positive: {positive:4} ({pos_ratio:.3f}), Negative: {negative:4} ({neg_ratio:.3f})")
