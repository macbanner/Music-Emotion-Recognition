# --- START OF FILE model_manager.py ---

import os # <-- Import os for directory creation
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score


class ModelManager:
    """
    Class for building, training, and evaluating emotion recognition models.
    Includes a pure CNN and a CNN+GRU architecture.
    """

    def __init__(self, num_emotions, emotion_columns):
        self.num_emotions = num_emotions
        self.emotion_columns = emotion_columns # emotion_columns is needed for saving

    def compute_class_weights(self, y_train):
        """
        Computes class weights for balancing the loss function.

        Parameters:
            y_train (np.ndarray): Training labels.

        Returns:
            np.ndarray: Array of class weights.
        """
        num_samples = y_train.shape[0]
        num_classes = y_train.shape[1]
        class_weights = np.zeros(num_classes)

        for i in range(num_classes):
            count_pos = np.sum(y_train[:, i])
            # Add small epsilon to avoid division by zero and handle cases with 0 positive samples better
            class_weights[i] = num_samples / (2.0 * count_pos + 1e-6)

        # Normalize weights to prevent overly large values if counts are very small
        # class_weights = class_weights / np.sum(class_weights) * num_classes

        return class_weights

    def weighted_binary_crossentropy(self, class_weights):
        """
        Creates a weighted binary crossentropy loss function.
        NOTE: The standard 'binary_crossentropy' loss in Keras can often handle
              class weights directly via the `class_weight` parameter in `model.fit()`.
              This custom function might be needed for specific weighting schemes,
              but let's try the standard approach first.

        Parameters:
            class_weights (array-like): Pre-computed class weights.

        Returns:
            function: Weighted binary crossentropy loss function.
        """
        class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

        def loss(y_true, y_pred):
            # Clip predictions to avoid log(0) errors
            y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
            # Calculate loss per sample
            bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
            # Apply weights based on true labels
            weights = y_true * class_weights_tensor + (1.0 - y_true) # No weight for negative class needed if balanced this way
            weighted_bce = weights * bce
            return tf.keras.backend.mean(weighted_bce)

        return loss

    def build_model(self, input_shape, class_weights, dropout_rate=0.5, learning_rate=0.0005):
        """
        Builds and compiles the original pure CNN model.

        Parameters:
            input_shape (tuple): Shape of the input (e.g., freq_bins, time_steps, channels).
            class_weights (array-like): Class weights for loss function.
            dropout_rate (float): Dropout rate.
            learning_rate (float): Learning rate.

        Returns:
            tf.keras.Model: The compiled CNN model.
        """
        print("Building Pure CNN Model...")
        l2_reg = 0.001
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=input_shape),

            # First Conv2D block
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate / 2),

            # Second Conv2D block
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate / 2),

            # Third Conv2D block
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),

            # Flatten and dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),

            # Output layer
            tf.keras.layers.Dense(self.num_emotions, activation='sigmoid', name='output')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Use class_weights directly in compile if loss supports it or pass to fit
        # Using standard BCE loss here. Class weights will be passed to fit().
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy', # Standard binary crossentropy
            metrics=[
                "binary_accuracy",
                tf.keras.metrics.AUC(name='auc'), # Name AUC metric for clarity
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
            ],
        )
        model.summary()
        return model

    def build_cnn_gru_model(self, input_shape, class_weights, dropout_rate=0.5, learning_rate=0.0005, recurrent_units=64):
        """
        Builds and compiles a CNN + GRU model.

        Parameters:
            input_shape (tuple): Shape of the input (e.g., freq_bins, time_steps, channels).
            class_weights (array-like): Class weights for loss function.
            dropout_rate (float): Dropout rate.
            learning_rate (float): Learning rate.
            recurrent_units (int): Number of units in the GRU layer.

        Returns:
            tf.keras.Model: The compiled CNN+GRU model.
        """
        print(f"Building CNN + GRU Model (GRU units: {recurrent_units})...")
        l2_reg = 0.002

        # Input layer
        inputs = tf.keras.layers.Input(shape=input_shape)

        # CNN Feature Extractor Part
        # Block 1
        x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2_reg))(inputs)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate / 2)(x)

        # Block 2
        x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2_reg))(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate / 2)(x)

        # Block 3
        x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.L2(l2_reg))(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dropout(dropout_rate)(x) # Optional dropout here

        # Reshape for RNN: (batch, freq_reduced, time_reduced, channels) -> (batch, time_reduced, features)
        # The 'time' dimension is typically the last dimension after Conv2D layers if input is (freq, time, ch)
        # Check the shape after the last pooling layer to confirm dimensions
        # Assuming Keras default 'channels_last': (batch, height, width, channels)
        # If input_shape is (freq_bins, time_steps, 1), then after pooling:
        # shape will be (batch, freq_bins_reduced, time_steps_reduced, num_filters)
        # We want to feed the GRU sequences along the time dimension.
        # Reshape combines frequency and channel information for each time step.
        # Shape after last MaxPooling2D: x.shape = (batch_size, H', W', C')
        # We want input to GRU as (batch_size, T, F) where T=W' (time steps) and F = H'*C' (features per step)
        current_shape = x.shape
        # Expected shape: (batch_size, pooled_freq_bins, pooled_time_steps, channels)
        # Example: If input is (128, 130, 1) and 3x pooling (2,2) -> (16, 16, 128)
        target_shape_dim1 = current_shape[2] # pooled_time_steps
        target_shape_dim2 = current_shape[1] * current_shape[3] # pooled_freq_bins * channels
        x = tf.keras.layers.Reshape((target_shape_dim1, target_shape_dim2))(x)


        # GRU Layer to process sequence
        # return_sequences=False because we want the output after processing the whole sequence
        x = tf.keras.layers.GRU(recurrent_units, return_sequences=False, dropout=dropout_rate, recurrent_dropout=dropout_rate, kernel_regularizer=tf.keras.regularizers.L2(l2_reg))(x)
        x = tf.keras.layers.BatchNormalization()(x) # Normalize GRU output

        # Dense layers for classification
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2_reg))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        # Output layer
        outputs = tf.keras.layers.Dense(self.num_emotions, activation='sigmoid', name='output')(x)

        # Build the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy', # Use standard loss, weights passed to fit()
            metrics=[
                "binary_accuracy",
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
            ],
        )
        model.summary()
        return model

    def train_model(self, model, X_train, y_train, X_val, y_val, class_weights_dict, # Pass class weights dict here
                    epochs=50, batch_size=32, patience=10):
        """
        Trains the model (either CNN or CNN+GRU).

        Parameters:
            model (tf.keras.Model): Compiled model.
            X_train, y_train: Training data and labels.
            X_val, y_val: Validation data and labels.
            class_weights_dict (dict): Dictionary mapping class indices to weights.
            epochs (int): Maximum epochs.
            batch_size (int): Batch size.
            patience (int): Early stopping patience.

        Returns:
            History: Training history.
        """
        # Data Augmentation (Optional, consider based on dataset size/performance)
        use_augmentation = False # Set to True if needed, e.g., if X_train.shape[0] < 1000
        if use_augmentation:
            print("Using data augmentation...")
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=5,      # Reduced rotation
                width_shift_range=0.05, # Reduced shift
                height_shift_range=0.05,# Reduced shift
                zoom_range=0.05,      # Reduced zoom
                horizontal_flip=False, # Usually not meaningful for spectrograms
                fill_mode="nearest",
            )
            train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
            steps_per_epoch = len(X_train) // batch_size
        else:
            train_generator = (X_train, y_train) # Use arrays directly
            steps_per_epoch = None # Not needed when not using generator flow

        # Callbacks
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        )
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras", monitor="val_loss", save_best_only=True, verbose=1
        )

        # Train model
        history = model.fit(
            train_generator if use_augmentation else X_train,
            y=None if use_augmentation else y_train, # y is part of generator if using flow
            steps_per_epoch=steps_per_epoch,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size if not use_augmentation else None, # Batch size handled by generator
            callbacks=[lr_scheduler, early_stopping, model_checkpoint],
            class_weight=class_weights_dict, # Pass the class weights here
            verbose=1,
        )

        return history

    def find_best_thresholds(self, y_true, y_pred_prob, thresholds=np.arange(0.3, 0.71, 0.05)):
        """
        Finds optimal thresholds per emotion based on F1 score on validation set.

        Parameters:
            y_true (np.ndarray): True labels.
            y_pred_prob (np.ndarray): Predicted probabilities.
            thresholds (iterable): Threshold values to try.

        Returns:
            list: Best threshold for each emotion.
        """
        best_thresholds = []
        num_classes = y_true.shape[1]

        print("\nFinding best thresholds...")
        for i in range(num_classes):
            best_f1 = -1.0 # Initialize with -1 to ensure any valid F1 is better
            best_threshold = 0.5 # Default threshold

            f1_scores = []
            for t in thresholds:
                y_pred = (y_pred_prob[:, i] >= t).astype(int)
                # Use macro average F1 for the single class (equivalent to binary F1)
                # Handle cases with no true positives or no predicted positives gracefully
                f1 = f1_score(y_true[:, i], y_pred, average='binary', zero_division=0)
                f1_scores.append(f1)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t

            best_thresholds.append(best_threshold)
            # Optional: print per-class threshold search results
            # print(f"  Emotion '{self.emotion_columns[i]}': Best F1={best_f1:.4f} at threshold={best_threshold:.2f}")


        print(f"Selected thresholds: {best_thresholds}")
        return best_thresholds

    def evaluate_model(self, model, X_test, y_test, thresholds=None):
        """
        Evaluates the model on test data using specified or default thresholds.

        Parameters:
            model (tf.keras.Model): Trained model.
            X_test (np.ndarray): Test data features.
            y_test (np.ndarray): Test data true labels.
            thresholds (list, optional): Thresholds per emotion. If None, uses 0.5 for all.

        Returns:
            tuple: (metrics_dict, used_thresholds_list)
        """
        print("\nEvaluating model on test set...")
        # Get predicted probabilities
        y_pred_prob = model.predict(X_test)

        # Determine thresholds to use
        if thresholds is None:
            print("Using default threshold of 0.5 for all emotions.")
            used_thresholds = [0.5] * y_test.shape[1]
        else:
            print(f"Using specified thresholds: {thresholds}")
            if len(thresholds) != y_test.shape[1]:
                 raise ValueError(f"Number of thresholds ({len(thresholds)}) must match number of emotions ({y_test.shape[1]})")
            used_thresholds = thresholds

        # Apply thresholds to get binary predictions
        y_pred = np.zeros_like(y_pred_prob)
        for i, t in enumerate(used_thresholds):
            y_pred[:, i] = (y_pred_prob[:, i] >= t).astype(int)

        # Calculate standard multi-label metrics
        print("Calculating metrics...")
        metrics = {
            "hamming_loss": hamming_loss(y_test, y_pred),
            "sample_f1": f1_score(y_test, y_pred, average="samples", zero_division=0),
            "micro_f1": f1_score(y_test, y_pred, average="micro", zero_division=0),
            "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "precision_micro": precision_score(y_test, y_pred, average="micro", zero_division=0),
            "recall_micro": recall_score(y_test, y_pred, average="micro", zero_division=0),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        }

        # Calculate per-emotion metrics
        for i, emotion in enumerate(self.emotion_columns):
            metrics[f"{emotion}_f1"] = f1_score(y_test[:, i], y_pred[:, i], average='binary', zero_division=0)
            metrics[f"{emotion}_precision"] = precision_score(y_test[:, i], y_pred[:, i], average='binary', zero_division=0)
            metrics[f"{emotion}_recall"] = recall_score(y_test[:, i], y_pred[:, i], average='binary', zero_division=0)

        print("Evaluation complete.")
        return metrics, used_thresholds

    # --- NEW METHOD ---
    def evaluate_and_save_test_data(self, model_path, X_val_proc, y_val, X_test_proc, y_test, output_npz_path):
        """
        Loads a pre-trained model, finds optimal thresholds on validation data,
        predicts on test data, and saves the required data to an NPZ file.

        Parameters:
            model_path (str): Path to the saved Keras model file (e.g., "music_emotion_model.keras").
            X_val_proc (np.ndarray): Preprocessed validation features.
            y_val (np.ndarray): Validation labels.
            X_test_proc (np.ndarray): Preprocessed test features.
            y_test (np.ndarray): Test labels.
            output_npz_path (str): Path to save the output NPZ file (e.g., "test_data/test_data.npz").
        """
        print(f"Loading pre-trained model from: {model_path}")
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        print("Predicting on validation set to find thresholds...")
        y_val_pred_prob = model.predict(X_val_proc)
        best_thresholds = self.find_best_thresholds(y_val, y_val_pred_prob) # Uses the method defined above

        print("Predicting on test set...")
        y_test_pred_prob = model.predict(X_test_proc)

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_npz_path)
        if output_dir: # Only create if path includes a directory
            os.makedirs(output_dir, exist_ok=True)

        print(f"Saving test data and predictions to: {output_npz_path}")
        # Save the required arrays using the specified keys
        np.savez(
            output_npz_path,
            X_test_proc=X_test_proc,            # The preprocessed test features
            y_test=y_test,                      # The true test labels
            emotion_cols=np.array(self.emotion_columns), # Emotion names as numpy array
            y_pred_prob=y_test_pred_prob,       # Predicted probabilities for test set
            thresholds=np.array(best_thresholds) # The best thresholds found using validation set
        )
        print("Data saved successfully.")
        # Optionally, you could also run evaluate_model here and return metrics
        # metrics, _ = self.evaluate_model(model, X_test_proc, y_test, thresholds=best_thresholds)
        # print("Test Set Metrics (using found thresholds):", metrics)


# --- END OF FILE model_manager.py ---