import tkinter as tk
from tkinter import ttk, filedialog
import customtkinter as ctk  # pip install customtkinter
from PIL import Image, ImageTk  # pip install pillow
import os
import webbrowser
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import time
import numpy as np
import tempfile
import sys
from pytube import YouTube  # For YouTube video downloading
import spotipy  # For Spotify integration
from spotipy.oauth2 import SpotifyClientCredentials

# Import from inference module
from inference import infer_track, EMOTIONS

# Default model paths
DEFAULT_MODEL_PATH = {
    "cnn":      r"test_data\cnn\CNN_best_model.keras",
    "cnn_gru":  r"test_data\gru\GRU_best_model.keras"
}
DEFAULT_THRESH_FILE = {
    "cnn":      r"test_data\cnn\CNN_test_data.npz",
    "cnn_gru":  r"test_data\gru\GRU_test_data.npz"
}

# Spotify API credentials - you would need to replace these with your own
# Get credentials from: https://developer.spotify.com/dashboard
SPOTIFY_CLIENT_ID = "YOUR_CLIENT_ID"  # Replace with your client ID
SPOTIFY_CLIENT_SECRET = "YOUR_CLIENT_SECRET"  # Replace with your client secret

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")  # Options: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Options: "blue" (default), "green", "dark-blue"


class MusicEmotionRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Music Emotion Recognition")
        self.geometry("900x700")
        self.minsize(800, 650)

        # Define colors and styles
        self.colors = {
            "primary": "#3B82F6",  # Blue
            "secondary": "#10B981",  # Green
            "accent": "#8B5CF6",  # Purple
            "warning": "#F59E0B",  # Amber
            "error": "#EF4444",  # Red
            "background": "#F9FAFB",  # Light gray
            "card": "#FFFFFF",  # White
            "text": "#1F2937",  # Dark gray
            "text_secondary": "#6B7280",  # Medium gray

            # Emotion colors
            "amazement": "#8B5CF6",  # Purple
            "solemnity": "#3B82F6",  # Blue
            "tenderness": "#EC4899",  # Pink
            "nostalgia": "#F59E0B",  # Amber
            "calmness": "#10B981",  # Green
            "power": "#EF4444",  # Red
            "joyful_activation": "#F97316",  # Orange
            "tension": "#6366F1",  # Indigo
            "sadness": "#6B7280",  # Gray
        }
        
        # Setup UI components
        self.setup_ui()
        
        # Load Spotify credentials after UI is initialized
        self.load_spotify_credentials()
        
        # Check for required audio libraries
        self.after(1000, self.check_audio_dependencies)

    def setup_ui(self):
        # Create main container
        self.main_container = ctk.CTkFrame(self, corner_radius=0)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Create header
        self.create_header()

        # Create content area with two columns
        self.content = ctk.CTkFrame(self.main_container)
        self.content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left column for inputs
        self.left_column = ctk.CTkFrame(self.content)
        self.left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=0)

        # Right column for results
        self.right_column = ctk.CTkFrame(self.content)
        self.right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=0)

        # Create input section
        self.create_input_section()

        # Create model selection section
        self.create_model_selection()

        # Create analyze button
        self.create_analyze_button()

        # Create results section
        self.create_results_section()

        # Footer with attribution
        self.create_footer()

    def create_header(self):
        header = ctk.CTkFrame(self.main_container, height=80, corner_radius=0)
        header.pack(fill=tk.X, padx=0, pady=0)

        # Logo and title
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side=tk.LEFT, padx=20, pady=10)

        # App logo (would normally be an icon)
        logo_label = ctk.CTkLabel(
            title_frame,
            text="🎵",
            font=ctk.CTkFont(family="Helvetica", size=36)
        )
        logo_label.pack(side=tk.LEFT, padx=(0, 10))

        # App title
        title_label = ctk.CTkLabel(
            title_frame,
            text="Music Emotion Recognition",
            font=ctk.CTkFont(family="Helvetica", size=24, weight="bold")
        )
        title_label.pack(side=tk.LEFT)

        # Theme toggle button
        theme_button = ctk.CTkButton(
            header,
            text="Toggle Theme",
            command=self.toggle_theme,
            width=120,
            height=32
        )
        theme_button.pack(side=tk.RIGHT, padx=20, pady=20)

    def create_input_section(self):
        # Input Section Frame
        input_frame = ctk.CTkFrame(self.left_column)
        input_frame.pack(fill=tk.X, padx=0, pady=(0, 20))

        # Section title
        input_title = ctk.CTkLabel(
            input_frame,
            text="Input Source",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        input_title.pack(anchor=tk.W, padx=15, pady=(15, 10))

        # Tab view for different input methods
        self.input_tabs = ctk.CTkTabview(input_frame, height=280)
        self.input_tabs.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        # Create tabs
        self.input_tabs.add("Local File")
        self.input_tabs.add("Spotify URL")
        self.input_tabs.add("YouTube URL")
        self.input_tabs.add("Search")

        # --- Local File Tab ---
        local_frame = ctk.CTkFrame(self.input_tabs.tab("Local File"), fg_color="transparent")
        local_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=20)

        # File path display
        self.file_path_var = tk.StringVar(value="No file selected")
        file_path_label = ctk.CTkLabel(
            local_frame,
            textvariable=self.file_path_var,
            font=ctk.CTkFont(size=12),
            wraplength=300
        )
        file_path_label.pack(fill=tk.X, padx=10, pady=(10, 20))

        # Browse button
        browse_button = ctk.CTkButton(
            local_frame,
            text="Browse Audio Files",
            command=self.browse_file,
            height=40
        )
        browse_button.pack(pady=10)

        # Supported formats info
        formats_label = ctk.CTkLabel(
            local_frame,
            text="Supported formats: .mp3, .wav, .ogg, .flac",
            font=ctk.CTkFont(size=12),
            text_color=self.colors["text_secondary"]
        )
        formats_label.pack(pady=(10, 0))

        # --- Spotify URL Tab ---
        spotify_frame = ctk.CTkFrame(self.input_tabs.tab("Spotify URL"), fg_color="transparent")
        spotify_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=20)

        # Spotify URL entry
        spotify_label = ctk.CTkLabel(
            spotify_frame,
            text="Enter Spotify Track URL:",
            anchor=tk.W
        )
        spotify_label.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.spotify_url_var = tk.StringVar()
        spotify_entry = ctk.CTkEntry(
            spotify_frame,
            textvariable=self.spotify_url_var,
            placeholder_text="https://open.spotify.com/track/...",
            height=40
        )
        spotify_entry.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Example tracks
        examples_frame = ctk.CTkFrame(spotify_frame, fg_color="transparent")
        examples_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        
        # Example 1
        example1_button = ctk.CTkButton(
            examples_frame,
            text="Example: Ed Sheeran",
            command=lambda: self.spotify_url_var.set("https://open.spotify.com/track/0V3wPSX9ygBnCm8psDIegu"),
            height=35,
            fg_color=self.colors["secondary"]
        )
        example1_button.pack(side=tk.LEFT, expand=True, padx=(0, 5), fill=tk.X)
        
        # Example 2
        example2_button = ctk.CTkButton(
            examples_frame,
            text="Example: Taylor Swift",
            command=lambda: self.spotify_url_var.set("https://open.spotify.com/track/0V3wPSX9ygBnCm8psDIegu"),
            height=35,
            fg_color=self.colors["secondary"]
        )
        example2_button.pack(side=tk.RIGHT, expand=True, padx=(5, 0), fill=tk.X)
        
        # API Credentials frame
        creds_frame = ctk.CTkFrame(spotify_frame)
        creds_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        # Title
        creds_title = ctk.CTkLabel(
            creds_frame,
            text="Spotify API Credentials",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        creds_title.pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # Client ID
        client_id_label = ctk.CTkLabel(
            creds_frame,
            text="Client ID:",
            anchor=tk.W
        )
        client_id_label.pack(fill=tk.X, padx=10, pady=(5, 2))
        
        self.client_id_var = tk.StringVar(value=SPOTIFY_CLIENT_ID)
        client_id_entry = ctk.CTkEntry(
            creds_frame,
            textvariable=self.client_id_var,
            placeholder_text="Enter your Spotify Client ID",
            height=30
        )
        client_id_entry.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Client Secret
        client_secret_label = ctk.CTkLabel(
            creds_frame,
            text="Client Secret:",
            anchor=tk.W
        )
        client_secret_label.pack(fill=tk.X, padx=10, pady=(5, 2))
        
        self.client_secret_var = tk.StringVar(value=SPOTIFY_CLIENT_SECRET)
        client_secret_entry = ctk.CTkEntry(
            creds_frame,
            textvariable=self.client_secret_var,
            placeholder_text="Enter your Spotify Client Secret",
            height=30,
            show="*"  # Hide the secret value
        )
        client_secret_entry.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Save credentials button
        save_creds_button = ctk.CTkButton(
            creds_frame,
            text="Save Credentials",
            command=self.save_spotify_credentials,
            height=30,
            fg_color=self.colors["primary"]
        )
        save_creds_button.pack(padx=10, pady=(0, 10))
        
        # Instructions
        instructions_label = ctk.CTkLabel(
            spotify_frame,
            text="To use Spotify integration, you need to create a Spotify Developer account and get API credentials from https://developer.spotify.com",
            font=ctk.CTkFont(size=12),
            text_color=self.colors["text_secondary"],
            wraplength=300,
            justify="left"
        )
        instructions_label.pack(pady=(5, 0), padx=10)
        
        # Note about previews
        preview_note = ctk.CTkLabel(
            spotify_frame,
            text="Note: Spotify analysis uses 30-second preview clips, which may not be available for all tracks. If unavailable, you'll be prompted to use YouTube instead.",
            font=ctk.CTkFont(size=11),
            text_color=self.colors["text_secondary"],
            wraplength=300,
            justify="left"
        )
        preview_note.pack(pady=(5, 0), padx=10)

        # --- YouTube URL Tab ---
        youtube_frame = ctk.CTkFrame(self.input_tabs.tab("YouTube URL"), fg_color="transparent")
        youtube_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=20)

        # YouTube URL entry
        youtube_label = ctk.CTkLabel(
            youtube_frame,
            text="Enter YouTube Video URL:",
            anchor=tk.W
        )
        youtube_label.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.youtube_url_var = tk.StringVar()
        youtube_entry = ctk.CTkEntry(
            youtube_frame,
            textvariable=self.youtube_url_var,
            placeholder_text="https://www.youtube.com/watch?v=...",
            height=40
        )
        youtube_entry.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Example buttons with working examples
        youtube_examples_frame = ctk.CTkFrame(youtube_frame, fg_color="transparent")
        youtube_examples_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        
        # Example 1: Adele - Hello
        youtube_example1_button = ctk.CTkButton(
            youtube_examples_frame,
            text="Example: Adele - Hello",
            command=lambda: self.youtube_url_var.set("https://www.youtube.com/watch?v=YQHsXMglC9A"),
            height=35,
            fg_color=self.colors["secondary"]
        )
        youtube_example1_button.pack(side=tk.LEFT, expand=True, padx=(0, 5), fill=tk.X)
        
        # Example 2: Relaxing music
        youtube_example2_button = ctk.CTkButton(
            youtube_examples_frame,
            text="Example: Relaxing Music",
            command=lambda: self.youtube_url_var.set("https://www.youtube.com/watch?v=lFcSrYw-ARY"),
            height=35,
            fg_color=self.colors["secondary"]
        )
        youtube_example2_button.pack(side=tk.RIGHT, expand=True, padx=(5, 0), fill=tk.X)
        
        # Format help
        youtube_format_label = ctk.CTkLabel(
            youtube_frame,
            text="Supported formats: Standard YouTube URLs (youtube.com/watch?v=... or youtu.be/...)",
            font=ctk.CTkFont(size=12),
            text_color=self.colors["text_secondary"],
            wraplength=300
        )
        youtube_format_label.pack(pady=(10, 0))

        # --- Search Tab ---
        search_frame = ctk.CTkFrame(self.input_tabs.tab("Search"), fg_color="transparent")
        search_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=20)

        # Search entry
        search_label = ctk.CTkLabel(
            search_frame,
            text="Search for a song:",
            anchor=tk.W
        )
        search_label.pack(fill=tk.X, padx=10, pady=(10, 5))

        search_entry_frame = ctk.CTkFrame(search_frame, fg_color="transparent")
        search_entry_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.search_var = tk.StringVar()
        search_entry = ctk.CTkEntry(
            search_entry_frame,
            textvariable=self.search_var,
            placeholder_text="Artist name - Song title",
            height=40
        )
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        search_button = ctk.CTkButton(
            search_entry_frame,
            text="Search",
            command=self.search_songs,
            width=80,
            height=40
        )
        search_button.pack(side=tk.RIGHT)

        # Search results (simulated)
        self.search_results_frame = ctk.CTkFrame(search_frame)
        self.search_results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Initially hide results
        self.search_results_frame.pack_forget()

    def create_model_selection(self):
        # Model Selection Frame
        model_frame = ctk.CTkFrame(self.left_column)
        model_frame.pack(fill=tk.X, padx=0, pady=(0, 20))

        # Section title
        model_title = ctk.CTkLabel(
            model_frame,
            text="Model Selection",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        model_title.pack(anchor=tk.W, padx=15, pady=(15, 10))

        # Model options
        self.model_var = tk.StringVar(value="CNN + GRU")

        # Container for radio buttons
        radio_frame = ctk.CTkFrame(model_frame, fg_color="transparent")
        radio_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Create radio buttons
        cnn_radio = ctk.CTkRadioButton(
            radio_frame,
            text="CNN Model",
            variable=self.model_var,
            value="CNN",
            font=ctk.CTkFont(size=14)
        )
        cnn_radio.pack(side=tk.LEFT, padx=(0, 40), pady=10)

        cnn_gru_radio = ctk.CTkRadioButton(
            radio_frame,
            text="CNN + GRU Model",
            variable=self.model_var,
            value="CNN + GRU",
            font=ctk.CTkFont(size=14)
        )
        cnn_gru_radio.pack(side=tk.LEFT, padx=0, pady=10)

        # Model info accordion
        model_info_frame = ctk.CTkFrame(model_frame)
        model_info_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Expand/collapse button
        self.info_expanded = False
        self.info_button = ctk.CTkButton(
            model_info_frame,
            text="ⓘ Model Information",
            command=self.toggle_model_info,
            fg_color="transparent",
            text_color=self.colors["primary"],
            hover_color=("#E1E5EA", "#2D333B"),  # Light/Dark mode hover colors
            anchor="w"
        )
        self.info_button.pack(fill=tk.X, padx=10, pady=5)

        # Hidden info content
        self.info_content = ctk.CTkTextbox(
            model_info_frame,
            height=0,
            wrap="word",
            font=ctk.CTkFont(size=12)
        )
        self.info_content.insert(
            "1.0",
            "CNN Model: Convolutional Neural Network optimized for spectral analysis of audio. "
            "Better for shorter tracks with consistent features.\n\n"
            "CNN + GRU Model: Combines CNN with Gated Recurrent Units for temporal pattern recognition. "
            "Better for capturing emotional progression in longer pieces."
        )
        self.info_content.configure(state="disabled")

    def create_analyze_button(self):
        # Analyze button frame (at bottom of left column)
        analyze_frame = ctk.CTkFrame(self.left_column, fg_color="transparent")
        analyze_frame.pack(fill=tk.X, padx=0, pady=(0, 10))

        # Large, prominent analyze button
        self.analyze_button = ctk.CTkButton(
            analyze_frame,
            text="Analyze Music",
            command=self.analyze_music,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=self.colors["primary"],
            hover_color="#2563EB"  # Darker blue on hover
        )
        self.analyze_button.pack(fill=tk.X, padx=15, pady=15)

    def create_results_section(self):
        # Results Frame
        results_frame = ctk.CTkFrame(self.right_column)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Section title
        results_title = ctk.CTkLabel(
            results_frame,
            text="Emotion Analysis Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        results_title.pack(anchor=tk.W, padx=15, pady=(15, 10))

        # Placeholder when no results
        self.placeholder_frame = ctk.CTkFrame(results_frame, fg_color="transparent")
        self.placeholder_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        placeholder_label = ctk.CTkLabel(
            self.placeholder_frame,
            text="Upload an audio file and click 'Analyze Music'\nto see emotion analysis results",
            font=ctk.CTkFont(size=14),
            text_color=self.colors["text_secondary"]
        )
        placeholder_label.pack(expand=True)

        # Results content (initially hidden)
        self.results_content = ctk.CTkFrame(results_frame)
        self.results_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.results_content.pack_forget()  # Hide initially

        # Track info
        self.track_info_frame = ctk.CTkFrame(self.results_content)
        self.track_info_frame.pack(fill=tk.X, padx=0, pady=(0, 15))

        self.track_title_var = tk.StringVar(value="Unknown Track")
        track_title_label = ctk.CTkLabel(
            self.track_info_frame,
            textvariable=self.track_title_var,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        track_title_label.pack(anchor=tk.W, padx=10, pady=(10, 5))

        self.track_artist_var = tk.StringVar(value="Unknown Artist")
        track_artist_label = ctk.CTkLabel(
            self.track_info_frame,
            textvariable=self.track_artist_var,
            font=ctk.CTkFont(size=14)
        )
        track_artist_label.pack(anchor=tk.W, padx=10, pady=(0, 10))

        # Detected emotions
        detected_emotions_frame = ctk.CTkFrame(self.results_content)
        detected_emotions_frame.pack(fill=tk.X, padx=0, pady=(0, 15))

        emotions_label = ctk.CTkLabel(
            detected_emotions_frame,
            text="Detected Emotions:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        emotions_label.pack(anchor=tk.W, padx=10, pady=(10, 10))

        # Emotion badges container
        self.emotion_badges_frame = ctk.CTkFrame(detected_emotions_frame, fg_color="transparent")
        self.emotion_badges_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Probability chart
        chart_frame = ctk.CTkFrame(self.results_content)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=(0, 0))

        chart_label = ctk.CTkLabel(
            chart_frame,
            text="Emotion Probabilities:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        chart_label.pack(anchor=tk.W, padx=10, pady=(10, 10))

        # Create matplotlib figure for the chart
        self.figure_frame = ctk.CTkFrame(chart_frame)
        self.figure_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.figure_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial empty chart
        self.update_chart([])

    def create_footer(self):
        # Footer
        footer = ctk.CTkFrame(self.main_container, height=30, corner_radius=0)
        footer.pack(fill=tk.X, padx=0, pady=0)

        # Create footer content
        footer_label = ctk.CTkLabel(
            footer,
            text="Music Emotion Recognition v1.0",
            font=ctk.CTkFont(size=12),
            text_color=self.colors["text_secondary"]
        )
        footer_label.pack(side=tk.RIGHT, padx=20, pady=5)

    # ========== Event Handlers ==========

    def toggle_theme(self):
        # Toggle between light and dark mode
        current_mode = ctk.get_appearance_mode()
        new_mode = "Light" if current_mode == "Dark" else "Dark"
        ctk.set_appearance_mode(new_mode)

    def toggle_model_info(self):
        # Expand or collapse model info
        if self.info_expanded:
            # Collapse
            self.info_content.configure(height=0)
            self.info_button.configure(text="ⓘ Model Information")
        else:
            # Expand
            self.info_content.configure(height=80)
            self.info_button.configure(text="ⓧ Hide Model Information")

        self.info_expanded = not self.info_expanded

    def browse_file(self):
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=(
                ("Audio Files", "*.mp3 *.wav *.ogg *.flac"),
                ("All Files", "*.*")
            )
        )

        if file_path:
            self.file_path_var.set(file_path)
            # Extract filename for display
            filename = os.path.basename(file_path)
            self.track_title_var.set(filename)
            self.track_artist_var.set("Local File")

    def search_songs(self):
        # Simulate search (would connect to Spotify API in real implementation)
        search_query = self.search_var.get()

        if not search_query:
            return

        # Show loading
        self.search_results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        for widget in self.search_results_frame.winfo_children():
            widget.destroy()

        loading_label = ctk.CTkLabel(
            self.search_results_frame,
            text="Searching...",
            font=ctk.CTkFont(size=12)
        )
        loading_label.pack(pady=20)
        self.update()

        # Simulate search delay
        threading.Thread(target=self._delayed_search, args=(search_query,)).start()

    def _delayed_search(self, query):
        # Simulate API delay
        time.sleep(1.5)

        # Demo results based on query
        results = []
        if "rock" in query.lower():
            results = [
                {"title": "Sweet Child O' Mine", "artist": "Guns N' Roses"},
                {"title": "Bohemian Rhapsody", "artist": "Queen"},
                {"title": "Stairway to Heaven", "artist": "Led Zeppelin"}
            ]
        elif "pop" in query.lower():
            results = [
                {"title": "Bad Guy", "artist": "Billie Eilish"},
                {"title": "Shape of You", "artist": "Ed Sheeran"},
                {"title": "Blinding Lights", "artist": "The Weeknd"}
            ]
        else:
            results = [
                {"title": f"{query} (Best Match)", "artist": "Popular Artist"},
                {"title": f"{query} Remix", "artist": "DJ Someone"},
                {"title": f"The {query} Song", "artist": "Various Artists"}
            ]

        # Update UI in main thread
        self.after(0, lambda: self._update_search_results(results))

    def _update_search_results(self, results):
        # Clear previous results
        for widget in self.search_results_frame.winfo_children():
            widget.destroy()

        if not results:
            no_results = ctk.CTkLabel(
                self.search_results_frame,
                text="No results found",
                font=ctk.CTkFont(size=12)
            )
            no_results.pack(pady=20)
            return

        # Create scrollable frame for results
        scroll_frame = ctk.CTkScrollableFrame(
            self.search_results_frame,
            height=150,
            label_text="Search Results"
        )
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Add results as buttons
        for i, result in enumerate(results):
            result_btn = ctk.CTkButton(
                scroll_frame,
                text=f"{result['title']} - {result['artist']}",
                command=lambda r=result: self.select_search_result(r),
                height=30,
                anchor="w",
                fg_color=("gray90", "gray20"),
                text_color=self.colors["text"],
                hover_color=("gray80", "gray30")
            )
            result_btn.pack(fill=tk.X, padx=5, pady=(5 if i > 0 else 0))

    def select_search_result(self, result):
        # Set track info from search result
        self.track_title_var.set(result["title"])
        self.track_artist_var.set(result["artist"])

        # Give visual feedback
        for tab in self.input_tabs._tab_dict.values():
            tab.configure(fg_color=self.input_tabs._fg_color)
        self.input_tabs.set("Search")

        # Show a selected indicator
        feedback = ctk.CTkLabel(
            self.search_results_frame,
            text=f"Selected: {result['title']}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=self.colors["secondary"]
        )
        feedback.pack(side=tk.BOTTOM, pady=5)

    def analyze_music(self):
        """Analyzes the selected music file using inference module."""
        # Get input source based on active tab
        active_tab = self.input_tabs.get()
        input_source = None
        is_youtube = False
        is_spotify = False
        spotify_suggestion = None

        if active_tab == "Local File":
            input_source = self.file_path_var.get()
            if input_source == "No file selected":
                self.show_message("Please select an audio file first.")
                return
        elif active_tab == "Spotify URL":
            input_source = self.spotify_url_var.get()
            if not input_source:
                self.show_message("Please enter a Spotify URL.")
                return
            
            # Update global variables from UI entries
            global SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
            SPOTIFY_CLIENT_ID = self.client_id_var.get()
            SPOTIFY_CLIENT_SECRET = self.client_secret_var.get()
            
            # Check if credentials are provided
            if SPOTIFY_CLIENT_ID == "YOUR_CLIENT_ID" or SPOTIFY_CLIENT_SECRET == "YOUR_CLIENT_SECRET":
                self.show_message("Please enter your Spotify API credentials first.")
                return
                
            # Validate Spotify URL
            if not self.validate_spotify_url(input_source):
                self.show_message("Invalid Spotify URL format. Please use a standard Spotify track URL.")
                return
                
            is_spotify = True
        elif active_tab == "YouTube URL":
            input_source = self.youtube_url_var.get()
            if not input_source:
                self.show_message("Please enter a YouTube URL.")
                return
            
            # Validate YouTube URL
            if not self.validate_youtube_url(input_source):
                self.show_message("Invalid YouTube URL format. Please use a standard YouTube URL.")
                return
                
            is_youtube = True
        elif active_tab == "Search":
            # Check if a track was selected from search
            title = self.track_title_var.get()
            if title == "Unknown Track":
                self.show_message("Please search and select a track first.")
                return
            self.show_message("Search feature not fully implemented yet.")
            return  # Not implemented yet

        # Get selected model
        model_type = self.model_var.get().lower().replace(" + ", "_")  # Convert "CNN + GRU" to "cnn_gru"
        model_path = DEFAULT_MODEL_PATH.get(model_type)
        thresholds_file = DEFAULT_THRESH_FILE.get(model_type)

        if not model_path or not os.path.exists(model_path):
            self.show_message(f"Model file not found: {model_path}")
            return
            
        if not thresholds_file or not os.path.exists(thresholds_file):
            self.show_message(f"Thresholds file not found: {thresholds_file}")
            return

        # Show analyzing indicator and create progress UI
        self.analyze_button.configure(
            text="Analyzing...",
            state="disabled",
            fg_color=self.colors["text_secondary"]
        )

        # Hide placeholder, show results with progress indicator
        self.placeholder_frame.pack_forget()
        self.results_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Create a progress frame
        self.progress_frame = ctk.CTkFrame(self.results_content)
        self.progress_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Add progress label
        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="Initializing analysis...",
            font=ctk.CTkFont(size=14)
        )
        self.progress_label.pack(pady=(10, 5))
        
        # Add progress bar
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=(0, 10))
        self.progress_bar.set(0)
        
        self.update()
        
        # Start analysis in a separate thread to keep UI responsive
        threading.Thread(target=self._run_analysis, args=(input_source, model_type, model_path, thresholds_file, is_youtube, is_spotify)).start()

    def _run_analysis(self, audio_path, model_type, model_path, thresholds_file, is_youtube=False, is_spotify=False):
        """Runs the music analysis in background thread and updates UI with progress."""
        temp_file = None
        try:
            # Handle YouTube URL
            if is_youtube:
                try:
                    # Try alternative download method first (more reliable)
                    self.progress_label.configure(text="Downloading YouTube audio...")
                    self.progress_bar.set(0.1)
                    self.update()
                    
                    alt_audio_path = self.try_alternative_youtube_download(audio_path)
                    if alt_audio_path:
                        audio_path = alt_audio_path
                        temp_file = alt_audio_path
                    else:
                        # If alternative method fails, try primary method
                        self.progress_label.configure(text="Alternative method failed, trying primary download method...")
                        self.update()
                        try:
                            # Try primary method as fallback
                            audio_path = self.download_youtube_audio(audio_path)
                            temp_file = audio_path
                            
                            # Convert to WAV for compatibility
                            converted_path = self.convert_to_wav(audio_path)
                            if converted_path != audio_path:
                                audio_path = converted_path
                                # Don't delete the original yet since temp_file is used for cleanup
                        except Exception as yt_error:
                            # Both methods failed
                            self.after(0, lambda: self.show_message(
                                f"YouTube download failed: {str(yt_error)}\nTry a different video or URL format.",
                                duration=5000
                            ))
                            self.after(0, lambda: self.analyze_button.configure(
                                text="Analyze Music",
                                state="normal",
                                fg_color=self.colors["primary"]
                            ))
                            # Clean up any progress UI
                            self.after(0, lambda: self.progress_frame.destroy() if hasattr(self, 'progress_frame') else None)
                            return
                except Exception as yt_error:
                    # Handle all YouTube download errors
                    self.after(0, lambda: self.show_message(
                        f"YouTube error: {str(yt_error)}",
                        duration=5000
                    ))
                    self.after(0, lambda: self.analyze_button.configure(
                        text="Analyze Music",
                        state="normal",
                        fg_color=self.colors["primary"]
                    ))
                    # Clean up any progress UI
                    self.after(0, lambda: self.progress_frame.destroy() if hasattr(self, 'progress_frame') else None)
                    return
            
            # Handle Spotify URL
            elif is_spotify:
                try:
                    # Extract Spotify track
                    spotify_result = self.extract_spotify_track(audio_path)
                    
                    # Check if we got a valid audio file or just track info with no preview
                    if spotify_result:
                        if isinstance(spotify_result, tuple) and len(spotify_result) == 2:
                            spotify_audio_path, track_info = spotify_result
                            
                            if spotify_audio_path:
                                # We have a valid preview audio
                                audio_path = spotify_audio_path
                                temp_file = spotify_audio_path
                                
                                # Convert to WAV for compatibility
                                converted_path = self.convert_to_wav(audio_path)
                                if converted_path != audio_path:
                                    audio_path = converted_path
                            else:
                                # No preview available, prompt to search on YouTube instead
                                self.after(0, lambda: self.progress_label.configure(
                                    text=f"No preview available for '{track_info.get('track_name', 'this track')}'"))
                                
                                # Suggest YouTube search
                                youtube_query = track_info.get('youtube_query', '')
                                
                                # Create a suggestion button
                                suggestion_frame = ctk.CTkFrame(self.progress_frame)
                                suggestion_frame.pack(fill=tk.X, padx=10, pady=5)
                                
                                suggestion_label = ctk.CTkLabel(
                                    suggestion_frame,
                                    text="Try searching on YouTube instead:",
                                    font=ctk.CTkFont(size=12),
                                    anchor="w"
                                )
                                suggestion_label.pack(fill=tk.X, padx=5, pady=(5, 0))
                                
                                def search_on_youtube():
                                    # Switch to YouTube tab and set the query
                                    self.input_tabs.set("YouTube URL")
                                    self.youtube_url_var.set("")  # Clear first
                                    
                                    # Open YouTube search in browser
                                    youtube_search_url = f"https://www.youtube.com/results?search_query={youtube_query.replace(' ', '+')}"
                                    webbrowser.open(youtube_search_url)
                                    
                                    # Reset analyze button
                                    self.analyze_button.configure(
                                        text="Analyze Music",
                                        state="normal",
                                        fg_color=self.colors["primary"]
                                    )
                                    
                                    # Clean up progress frame
                                    if hasattr(self, 'progress_frame'):
                                        self.progress_frame.destroy()
                                
                                youtube_button = ctk.CTkButton(
                                    suggestion_frame,
                                    text=f"Search: {track_info.get('track_name', '')} - {track_info.get('artists', '')}",
                                    command=search_on_youtube,
                                    height=30,
                                    fg_color=self.colors["warning"]
                                )
                                youtube_button.pack(fill=tk.X, padx=5, pady=(5, 5))
                                
                                # Reset analyze button
                                self.after(0, lambda: self.analyze_button.configure(
                                    text="Analyze Music",
                                    state="normal",
                                    fg_color=self.colors["primary"]
                                ))
                                
                                return
                except Exception as spotify_error:
                    self.after(0, lambda: self.show_message(
                        f"Spotify error: {str(spotify_error)}\nCheck your API credentials or try a different track.",
                        duration=5000
                    ))
                    self.after(0, lambda: self.analyze_button.configure(
                        text="Analyze Music",
                        state="normal",
                        fg_color=self.colors["primary"]
                    ))
                    # Clean up any progress UI
                    self.after(0, lambda: self.progress_frame.destroy() if hasattr(self, 'progress_frame') else None)
                    return
            else:
                # For local files, make sure they're in a compatible format
                if os.path.exists(audio_path) and not audio_path.lower().endswith('.wav'):
                    # Convert to WAV if it's not already
                    converted_path = self.convert_to_wav(audio_path)
                    if converted_path != audio_path:
                        audio_path = converted_path
                        temp_file = converted_path  # Mark for cleanup
            
            # Update progress
            self.after(0, lambda: self.progress_label.configure(text="Loading thresholds..."))
            self.after(0, lambda: self.progress_bar.set(0.1))
            
            # Load thresholds
            data = np.load(thresholds_file, allow_pickle=True)
            thresholds = data["thresholds"]
            
            # Update progress
            self.after(0, lambda: self.progress_label.configure(text="Segmenting audio file..."))
            self.after(0, lambda: self.progress_bar.set(0.2))
            
            # Create a list to store progress updates
            progress_steps = [
                (0.3, "Extracting audio features..."),
                (0.5, "Normalizing features..."),
                (0.7, "Running model prediction..."),
                (0.9, "Processing results...")
            ]
            
            # Function to simulate progress updates during analysis
            def update_progress():
                for progress, message in progress_steps:
                    time.sleep(1)  # Simulate processing time
                    self.after(0, lambda p=progress, m=message: (
                        self.progress_label.configure(text=m),
                        self.progress_bar.set(p)
                    ))
            
            # Start progress updates in parallel
            progress_thread = threading.Thread(target=update_progress)
            progress_thread.daemon = True
            progress_thread.start()
            
            # Run inference
            results = infer_track(
                audio_path=audio_path,
                model_type=model_type,
                model_path=model_path,
                thresholds=thresholds,
                segment_length=3  # Default segment length
            )
            
            # Update UI with results in main thread
            self.after(0, lambda: self.progress_label.configure(text="Analysis complete!"))
            self.after(0, lambda: self.progress_bar.set(1.0))
            self.after(500, lambda: self._display_results(results))
            
        except Exception as e:
            # Handle any errors
            error_message = f"Analysis error: {str(e)}"
            self.after(0, lambda: self.show_message(error_message, duration=5000))
            self.after(0, lambda: self.analyze_button.configure(
                text="Analyze Music",
                state="normal",
                fg_color=self.colors["primary"]
            ))
            
            # Log the full error
            print(f"Error in analysis: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Clean up temporary file if it exists
            if temp_file and os.path.exists(temp_file):
                try:
                    temp_dir = os.path.dirname(temp_file)
                    # Remove the file
                    os.remove(temp_file)
                    # Try to remove the directory
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temporary files: {cleanup_error}")

    def _display_results(self, results):
        """Displays the emotion analysis results."""
        # Reset analyze button
        self.analyze_button.configure(
            text="Analyze Music",
            state="normal",
            fg_color=self.colors["primary"]
        )

        # Remove progress frame
        if hasattr(self, 'progress_frame'):
            self.progress_frame.destroy()

        # Clear previous emotion badges
        for widget in self.emotion_badges_frame.winfo_children():
            widget.destroy()

        # Extract predicted emotions and probabilities from results
        predicted_emotions = results["predicted_emotions"]
        probabilities = results["probabilities"]

        # Add emotion badges for detected emotions
        if predicted_emotions:
            for i, emotion in enumerate(predicted_emotions):
                badge = ctk.CTkButton(
                    self.emotion_badges_frame,
                    text=emotion.capitalize(),
                    font=ctk.CTkFont(size=12, weight="bold"),
                    fg_color=self.colors[emotion],
                    hover_color=self.colors[emotion],
                    height=28,
                    width=120,
                    corner_radius=14
                )
                badge.pack(side=tk.LEFT, padx=(0 if i == 0 else 5), pady=5)
        else:
            # No emotions detected
            no_emotion_label = ctk.CTkLabel(
                self.emotion_badges_frame,
                text="No dominant emotions detected",
                font=ctk.CTkFont(size=12),
                text_color=self.colors["text_secondary"]
            )
            no_emotion_label.pack(pady=5)

        # Update probability chart
        emotion_data = [
            (emotion.capitalize(), prob)
            for emotion, prob in probabilities.items()
        ]
        self.update_chart(emotion_data)

    def update_chart(self, emotion_data):
        # Clear the previous chart
        self.ax.clear()

        if not emotion_data:
            self.ax.text(0.5, 0.5, "No data to display",
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=self.ax.transAxes)
            self.fig.tight_layout()
            self.canvas.draw()
            return

        # Sort data by probability value (descending)
        emotion_data.sort(key=lambda x: x[1], reverse=True)

        # Extract emotions and values
        emotions = [item[0] for item in emotion_data]
        values = [item[1] for item in emotion_data]

        # Get colors for each emotion (use default if not in our color map)
        colors = [self.colors.get(emotion.lower(), "#6B7280") for emotion in emotions]

        # Create horizontal bar chart
        bars = self.ax.barh(emotions, values, color=colors)

        # Add value labels to the right of each bar
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.01
            self.ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
                         va='center', fontsize=8)

        # Add labels and title
        self.ax.set_xlabel('Probability')
        self.ax.set_title('Emotion Probabilities')

        # Set the x-axis limit to 1.1 to accommodate the labels
        self.ax.set_xlim(0, 1.1)

        # Customize grid
        self.ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        self.ax.set_axisbelow(True)

        # Customize chart appearance based on the current UI theme
        theme_mode = ctk.get_appearance_mode()
        if theme_mode == "Dark":
            self.fig.patch.set_facecolor('#2B2B2B')
            self.ax.set_facecolor('#2B2B2B')
            self.ax.spines['bottom'].set_color('#555555')
            self.ax.spines['top'].set_color('#555555')
            self.ax.spines['right'].set_color('#555555')
            self.ax.spines['left'].set_color('#555555')
            self.ax.tick_params(axis='x', colors='#CCCCCC')
            self.ax.tick_params(axis='y', colors='#CCCCCC')
            self.ax.xaxis.label.set_color('#CCCCCC')
            self.ax.title.set_color('#CCCCCC')
        else:
            self.fig.patch.set_facecolor('#F9FAFB')
            self.ax.set_facecolor('#F9FAFB')
            self.ax.spines['bottom'].set_color('#DDDDDD')
            self.ax.spines['top'].set_color('#DDDDDD')
            self.ax.spines['right'].set_color('#DDDDDD')
            self.ax.spines['left'].set_color('#DDDDDD')
            self.ax.tick_params(axis='x', colors='#333333')
            self.ax.tick_params(axis='y', colors='#333333')
            self.ax.xaxis.label.set_color('#333333')
            self.ax.title.set_color('#333333')

        # Update the plot
        self.fig.tight_layout()
        self.canvas.draw()

    def show_message(self, message, duration=2000):
        """Show a temporary message to the user"""
        # Create a message overlay
        overlay = ctk.CTkFrame(self)
        overlay.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Message label
        msg_label = ctk.CTkLabel(
            overlay,
            text=message,
            font=ctk.CTkFont(size=14),
            wraplength=400,  # Allow wrapping for longer messages
            justify="center",
            padx=20,
            pady=10
        )
        msg_label.pack(padx=10, pady=10)
        
        # If it's a longer message (error), add a dismiss button
        if len(message) > 80 or duration > 3000:
            dismiss_button = ctk.CTkButton(
                overlay,
                text="Dismiss",
                command=overlay.destroy,
                width=100,
                height=30
            )
            dismiss_button.pack(pady=(0, 10))
            # Use longer duration for errors
            duration = max(duration, 5000)

        # Auto-dismiss after duration
        self.after(duration, overlay.destroy)

    def create_emotion_color_legend(self):
        """Creates an expandable legend explaining emotion colors"""
        legend_frame = ctk.CTkFrame(self.right_column)
        legend_frame.pack(fill=tk.X, padx=0, pady=(0, 15))

        # Create a button to expand/collapse
        self.legend_expanded = False
        self.legend_button = ctk.CTkButton(
            legend_frame,
            text="ⓘ Emotion Color Legend",
            command=self.toggle_legend,
            fg_color="transparent",
            text_color=self.colors["primary"],
            hover_color=("#E1E5EA", "#2D333B"),  # Light/Dark mode hover colors
            anchor="w"
        )
        self.legend_button.pack(fill=tk.X, padx=10, pady=5)

        # Create container for color squares
        self.legend_content = ctk.CTkFrame(legend_frame, height=0)

        # Create a grid of color indicators
        emotions = [
            ("amazement", "Purple", "Feeling of wonder or awe"),
            ("solemnity", "Blue", "Serious, profound feeling"),
            ("tenderness", "Pink", "Gentle, affectionate emotion"),
            ("nostalgia", "Amber", "Sentimental longing for the past"),
            ("calmness", "Green", "State of tranquility"),
            ("power", "Red", "Strong, energetic feeling"),
            ("joyful_activation", "Orange", "Happy, uplifting emotion"),
            ("tension", "Indigo", "Feeling of anxiety or suspense"),
            ("sadness", "Gray", "Feeling of sorrow or unhappiness")
        ]

        row = 0
        col = 0
        max_cols = 3

        for emotion, color_name, description in emotions:
            # Frame for each emotion
            emotion_frame = ctk.CTkFrame(self.legend_content, fg_color="transparent")
            emotion_frame.grid(row=row, column=col, padx=5, pady=5, sticky="w")

            # Color square
            color_square = ctk.CTkFrame(
                emotion_frame,
                width=16,
                height=16,
                corner_radius=4,
                fg_color=self.colors[emotion]
            )
            color_square.pack(side=tk.LEFT, padx=(0, 5))

            # Label with tooltip effect
            label = ctk.CTkLabel(
                emotion_frame,
                text=f"{emotion.capitalize()} ({color_name})",
                font=ctk.CTkFont(size=12),
                width=120,
                anchor="w"
            )
            label.pack(side=tk.LEFT)

            # Create tooltip effect
            self.create_tooltip(label, description)

            # Update grid position
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

    def create_tooltip(self, widget, text):
        """Create a simple tooltip effect for a widget"""
        tooltip = None

        def enter(event):
            nonlocal tooltip
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25

            # Create tooltip window
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x}+{y}")

            label = ctk.CTkLabel(
                tooltip,
                text=text,
                font=ctk.CTkFont(size=11),
                padx=5,
                pady=2
            )
            label.pack()

        def leave(event):
            nonlocal tooltip
            if tooltip:
                tooltip.destroy()
                tooltip = None

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def toggle_legend(self):
        """Toggle emotion color legend visibility"""
        if self.legend_expanded:
            # Collapse
            self.legend_content.configure(height=0)
            self.legend_button.configure(text="ⓘ Emotion Color Legend")
            self.legend_content.pack_forget()
        else:
            # Expand
            self.legend_content.pack(fill=tk.X, padx=10, pady=(0, 10))
            self.legend_button.configure(text="ⓧ Hide Emotion Color Legend")

        self.legend_expanded = not self.legend_expanded

    def export_results(self):
        """Export analysis results to a file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Analysis Results"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w') as f:
                # Write header
                f.write("Music Emotion Analysis Results\n")
                f.write("=============================\n\n")

                # Track info
                f.write(f"Track: {self.track_title_var.get()}\n")
                f.write(f"Artist: {self.track_artist_var.get()}\n\n")

                # Model used
                f.write(f"Model: {self.model_var.get()}\n\n")

                # Get emotion data from chart
                f.write("Detected Emotions:\n")
                for widget in self.emotion_badges_frame.winfo_children():
                    if isinstance(widget, ctk.CTkButton):
                        f.write(f"- {widget.cget('text')}\n")
                    elif isinstance(widget, ctk.CTkLabel):
                        # This is the "No emotions detected" label
                        f.write(f"- {widget.cget('text')}\n")
                
                f.write("\nEmotion Probabilities:\n")
                if hasattr(self.ax, 'containers') and self.ax.containers:
                    for container in self.ax.containers:
                        for i, patch in enumerate(container):
                            emotion = self.ax.get_yticklabels()[i].get_text()
                            value = patch.get_width()
                            f.write(f"- {emotion}: {value:.2f}\n")

                f.write("\nExported on: " + time.strftime("%Y-%m-%d %H:%M:%S"))

            self.show_message(f"Results exported to {os.path.basename(file_path)}")
        except Exception as e:
            self.show_message(f"Export failed: {str(e)}")

    def add_export_button(self):
        """Add an export button to the results section"""
        export_button = ctk.CTkButton(
            self.results_content,
            text="Export Results",
            command=self.export_results,
            height=36,
            fg_color=self.colors["secondary"],
            hover_color="#0CA678"  # Darker green on hover
        )
        export_button.pack(anchor=tk.E, padx=10, pady=(15, 0))

    def show_about_dialog(self):
        """Show an about dialog with app information"""
        about_window = ctk.CTkToplevel(self)
        about_window.title("About Music Emotion Recognition")
        about_window.geometry("500x400")
        about_window.resizable(False, False)

        # Make modal
        about_window.transient(self)
        about_window.grab_set()

        # Center on parent
        x = self.winfo_x() + (self.winfo_width() / 2) - (500 / 2)
        y = self.winfo_y() + (self.winfo_height() / 2) - (400 / 2)
        about_window.geometry(f"+{int(x)}+{int(y)}")

        # Content frame
        content = ctk.CTkFrame(about_window)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Logo
        logo_label = ctk.CTkLabel(
            content,
            text="🎵",
            font=ctk.CTkFont(family="Helvetica", size=48)
        )
        logo_label.pack(pady=(20, 5))

        # App title
        title_label = ctk.CTkLabel(
            content,
            text="Music Emotion Recognition",
            font=ctk.CTkFont(family="Helvetica", size=24, weight="bold")
        )
        title_label.pack(pady=(0, 5))

        # Version
        version_label = ctk.CTkLabel(
            content,
            text="Version 1.0",
            font=ctk.CTkFont(size=12),
            text_color=self.colors["text_secondary"]
        )
        version_label.pack(pady=(0, 20))

        # Description
        description = (
            "This application analyzes audio tracks to detect emotional content "
            "using deep learning models. It can process local audio files, "
            "Spotify tracks, YouTube videos, or search for songs online."
        )
        desc_label = ctk.CTkLabel(
            content,
            text=description,
            font=ctk.CTkFont(size=12),
            wraplength=460,
            justify="center"
        )
        desc_label.pack(pady=(0, 20))

        # Technical info
        tech_frame = ctk.CTkFrame(content)
        tech_frame.pack(fill=tk.X, pady=(0, 20))

        tech_label = ctk.CTkLabel(
            tech_frame,
            text="Technical Information",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        tech_label.pack(anchor=tk.W, padx=10, pady=(10, 5))

        tech_info = (
            "• GUI: CustomTkinter\n"
            "• Audio Processing: Librosa\n"
            "• Models: CNN and CNN+GRU architectures\n"
            "• Visualization: Matplotlib"
        )
        tech_details = ctk.CTkLabel(
            tech_frame,
            text=tech_info,
            font=ctk.CTkFont(size=12),
            justify="left"
        )
        tech_details.pack(anchor=tk.W, padx=10, pady=(0, 10))

        # Close button
        close_button = ctk.CTkButton(
            content,
            text="Close",
            command=about_window.destroy,
            width=120
        )
        close_button.pack(pady=(0, 10))

    def add_help_menu(self):
        """Add a help menu to the app"""
        # Create menu button in header
        help_menu_button = ctk.CTkButton(
            self.main_container.winfo_children()[0],  # Header frame
            text="Help",
            width=80,
            height=32,
            command=self.show_help_menu
        )
        help_menu_button.pack(side=tk.RIGHT, padx=(0, 10), pady=20)

    def show_help_menu(self):
        """Show a dropdown help menu"""
        # Create a simple popup menu
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="User Guide", command=self.show_user_guide)
        menu.add_command(label="About", command=self.show_about_dialog)

        # Show menu below the help button
        help_button = self.main_container.winfo_children()[0].winfo_children()[-1]
        x = help_button.winfo_rootx()
        y = help_button.winfo_rooty() + help_button.winfo_height()
        menu.post(x, y)

    def show_user_guide(self):
        """Show a user guide window"""
        guide_window = ctk.CTkToplevel(self)
        guide_window.title("User Guide")
        guide_window.geometry("600x500")

        # Make modal
        guide_window.transient(self)
        guide_window.grab_set()

        # Center on parent
        x = self.winfo_x() + (self.winfo_width() / 2) - (600 / 2)
        y = self.winfo_y() + (self.winfo_height() / 2) - (500 / 2)
        guide_window.geometry(f"+{int(x)}+{int(y)}")

        # Create a scrollable frame for the content
        scroll_frame = ctk.CTkScrollableFrame(guide_window)
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        title_label = ctk.CTkLabel(
            scroll_frame,
            text="Music Emotion Recognition - User Guide",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(anchor=tk.W, pady=(0, 20))

        # Guide content
        sections = [
            {
                "title": "Getting Started",
                "content": (
                    "This application allows you to analyze the emotional content of music using "
                    "deep learning models. You can analyze local audio files, Spotify tracks, "
                    "YouTube videos, or search for tracks online."
                )
            },
            {
                "title": "Input Sources",
                "content": (
                    "• Local File: Upload MP3, WAV, OGG, or FLAC files from your computer.\n"
                    "• Spotify URL: Paste the URL of a Spotify track.\n"
                    "• YouTube URL: Paste the URL of a YouTube video.\n"
                    "• Search: Enter an artist or song name to search for tracks."
                )
            },
            {
                "title": "Model Selection",
                "content": (
                    "• CNN Model: Optimized for spectral analysis of audio. "
                    "Better for shorter tracks with consistent features.\n"
                    "• CNN + GRU Model: Combines CNN with Gated Recurrent Units for temporal pattern recognition. "
                    "Better for capturing emotional progression in longer pieces."
                )
            },
            {
                "title": "Analyzing Music",
                "content": (
                    "1. Select your input source and provide the audio track.\n"
                    "2. Choose a model type.\n"
                    "3. Click the 'Analyze Music' button.\n"
                    "4. Wait for the analysis to complete.\n"
                    "5. Review the detected emotions and probability chart."
                )
            },
            {
                "title": "Understanding Results",
                "content": (
                    "The analysis results show:\n"
                    "• Detected Emotions: The most dominant emotions found in the track.\n"
                    "• Emotion Probabilities: A chart showing the probability scores for all emotions.\n\n"
                    "The nine emotion categories are:\n"
                    "• Amazement: Feeling of wonder or awe\n"
                    "• Solemnity: Serious, profound feeling\n"
                    "• Tenderness: Gentle, affectionate emotion\n"
                    "• Nostalgia: Sentimental longing for the past\n"
                    "• Calmness: State of tranquility\n"
                    "• Power: Strong, energetic feeling\n"
                    "• Joyful Activation: Happy, uplifting emotion\n"
                    "• Tension: Feeling of anxiety or suspense\n"
                    "• Sadness: Feeling of sorrow or unhappiness"
                )
            },
            {
                "title": "Exporting Results",
                "content": (
                    "You can export analysis results to a text or CSV file by clicking the 'Export Results' button "
                    "in the results section."
                )
            }
        ]

        # Add each section
        for i, section in enumerate(sections):
            # Section title
            section_label = ctk.CTkLabel(
                scroll_frame,
                text=section["title"],
                font=ctk.CTkFont(size=16, weight="bold")
            )
            section_label.pack(anchor=tk.W, pady=(15 if i > 0 else 0, 5))

            # Section content
            content_label = ctk.CTkLabel(
                scroll_frame,
                text=section["content"],
                font=ctk.CTkFont(size=12),
                justify="left",
                wraplength=540
            )
            content_label.pack(anchor=tk.W, pady=(0, 10))

        # Close button
        close_button = ctk.CTkButton(
            scroll_frame,
            text="Close",
            command=guide_window.destroy,
            width=120
        )
        close_button.pack(pady=(20, 10))

    def download_youtube_audio(self, youtube_url):
        """
        Downloads audio from a YouTube video and returns the path to the temporary file.
        
        Args:
            youtube_url (str): The YouTube video URL
            
        Returns:
            str: Path to the downloaded audio file
        """
        try:
            # Update progress
            self.progress_label.configure(text="Fetching YouTube video information...")
            self.progress_bar.set(0.1)
            self.update()
            
            # Try to fix common URL issues
            if "youtu.be" in youtube_url:
                video_id = youtube_url.split("/")[-1].split("?")[0]
                youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            elif "youtube.com/watch" in youtube_url and "v=" not in youtube_url:
                self.progress_label.configure(text="Invalid YouTube URL format")
                raise ValueError("Invalid YouTube URL format. URL must contain 'v=' parameter.")
            
            # Workaround for pytube issues with cipher
            try:
                # Create a YouTube object
                yt = YouTube(youtube_url)
                # Fetch the video title to ensure connection works
                video_title = yt.title
            except Exception as e:
                if "HTTP Error 400" in str(e):
                    # Try an alternative approach by modifying the URL
                    if "?" in youtube_url:
                        base_url = youtube_url.split("?")[0]
                        params = youtube_url.split("?")[1]
                        video_id = None
                        for param in params.split("&"):
                            if param.startswith("v="):
                                video_id = param[2:]
                                break
                        if video_id:
                            fixed_url = f"https://www.youtube.com/watch?v={video_id}"
                            yt = YouTube(fixed_url)
                        else:
                            raise ValueError("Could not extract video ID from URL")
                    else:
                        # Provide a more helpful error message
                        raise ValueError(f"Invalid YouTube URL format: {youtube_url}")
                else:
                    # Re-raise other exceptions
                    raise
            
            # Update progress with video title
            self.track_title_var.set(yt.title)
            self.track_artist_var.set(yt.author)
            self.progress_label.configure(text=f"Downloading: {yt.title}")
            self.progress_bar.set(0.3)
            self.update()
            
            # Get the audio stream with highest quality
            try:
                audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
                if audio_stream is None:
                    raise ValueError("No audio stream found for this video")
            except Exception as stream_error:
                self.progress_label.configure(text="Error: No available audio streams")
                raise ValueError(f"Could not find audio streams: {str(stream_error)}")
            
            # Create a temporary directory to store the audio
            temp_dir = tempfile.mkdtemp()
            output_file = os.path.join(temp_dir, "youtube_audio.mp4")
            
            # Download the audio
            audio_stream.download(output_path=temp_dir, filename="youtube_audio.mp4")
            
            # Verify the file was downloaded successfully
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                raise ValueError("Download failed: File not created or empty")
            
            self.progress_label.configure(text="Download complete!")
            self.progress_bar.set(0.5)
            self.update()
            
            return output_file
            
        except Exception as e:
            error_message = f"Error downloading YouTube video: {str(e)}"
            print(f"YouTube download error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(error_message)

    def validate_youtube_url(self, url):
        """
        Validates a YouTube URL and extracts the video ID
        
        Args:
            url (str): The YouTube URL to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        import re
        
        # Common YouTube URL patterns
        patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, url)
            if match:
                video_id = match.group(1)
                if len(video_id) == 11:  # Standard YouTube video ID length
                    return True
        
        return False

    def try_alternative_youtube_download(self, youtube_url):
        """
        Alternative method to download YouTube audio using yt-dlp if pytube fails.
        This method first checks if yt-dlp is installed and installs it if needed.
        
        Args:
            youtube_url (str): The YouTube URL
            
        Returns:
            str: Path to downloaded audio file or None if failed
        """
        try:
            # Try to import yt-dlp
            try:
                import yt_dlp
            except ImportError:
                self.progress_label.configure(text="Installing yt-dlp (one-time setup)...")
                self.update()
                
                # Import subprocess to install yt-dlp
                import subprocess
                import sys
                
                # Install yt-dlp
                subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
                
                # Now import it
                import yt_dlp
            
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Set up optimal options for audio extraction
            # Directly download as mp3 for better compatibility
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(temp_dir, 'youtube_audio.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': True,
                'nooverwrites': False,
                'noplaylist': True,
                'prefer_ffmpeg': False  # Don't rely on FFmpeg yet
            }
            
            # Download video
            self.progress_label.configure(text="Downloading with yt-dlp...")
            self.progress_bar.set(0.2)
            self.update()
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Try extraction without download first to get metadata
                try:
                    info = ydl.extract_info(youtube_url, download=False)
                    
                    # Update track info in UI
                    if info:
                        title = info.get('title', 'Unknown Title')
                        uploader = info.get('uploader', 'Unknown Artist')
                        
                        self.track_title_var.set(title)
                        self.track_artist_var.set(uploader)
                except Exception as info_error:
                    print(f"Error extracting video info: {str(info_error)}")
                
                # Now download the audio
                try:
                    self.progress_label.configure(text="Downloading audio...")
                    self.progress_bar.set(0.3)
                    self.update()
                    ydl.download([youtube_url])
                except Exception as download_error:
                    print(f"Download error: {str(download_error)}")
                    raise
            
            # Check if download was successful
            self.progress_label.configure(text="Checking downloaded file...")
            self.progress_bar.set(0.4)
            self.update()
            
            # Look for the downloaded file - should be an mp3 now
            downloaded_file = None
            for filename in os.listdir(temp_dir):
                if os.path.splitext(filename)[1].lower() in ('.mp3', '.m4a', '.wav', '.webm'):
                    downloaded_file = os.path.join(temp_dir, filename)
                    break
            
            if not downloaded_file:
                raise ValueError("Download completed but no audio file was found")
            
            self.progress_label.configure(text="Download complete!")
            self.progress_bar.set(0.5)
            self.update()
            
            # If it's not already a wav file, we'll need to convert it
            if not downloaded_file.lower().endswith('.wav'):
                self.progress_label.configure(text="Converting to WAV format...")
                self.progress_bar.set(0.6)
                self.update()
                return downloaded_file  # Let the main convert_to_wav process handle it
            
            return downloaded_file
            
        except Exception as e:
            print(f"Alternative download failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def install_ffmpeg(self):
        """
        Attempts to install FFmpeg for the user.
        Shows a message with results.
        """
        try:
            self.show_message("Attempting to install FFmpeg. This may take a few minutes...", duration=5000)
            
            import platform
            import subprocess
            import os
            import sys
            
            system = platform.system().lower()
            
            if system == 'windows':
                # For Windows: Download FFmpeg using pip-based package
                self.show_message("Installing FFmpeg for Windows...", duration=5000)
                try:
                    # Try to use pip to install ffmpeg-python (wrapper, not actual ffmpeg)
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
                    
                    # Download the actual FFmpeg binaries using a script
                    temp_dir = tempfile.mkdtemp()
                    script_path = os.path.join(temp_dir, "get_ffmpeg.py")
                    
                    # Write a script to download and unzip FFmpeg
                    with open(script_path, 'w') as f:
                        f.write("""
import os
import sys
import urllib.request
import zipfile
import shutil

# Download FFmpeg
url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
zip_path = os.path.join(os.path.dirname(sys.executable), "ffmpeg.zip")
extract_path = os.path.join(os.path.dirname(sys.executable), "FFmpeg")

print("Downloading FFmpeg...")
urllib.request.urlretrieve(url, zip_path)

print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Find the bin directory in extracted folder
bin_dir = None
for root, dirs, files in os.walk(extract_path):
    if "bin" in dirs:
        bin_dir = os.path.join(root, "bin")
        break

if bin_dir:
    # Copy binaries to Python directory so they're in PATH
    for file in os.listdir(bin_dir):
        if file.endswith('.exe'):
            shutil.copy2(os.path.join(bin_dir, file), 
                         os.path.join(os.path.dirname(sys.executable), file))
    print("FFmpeg installed successfully!")
else:
    print("FFmpeg bin directory not found in the extracted files.")

# Clean up
os.remove(zip_path)
shutil.rmtree(extract_path)
""")
                    
                    # Run the script
                    subprocess.check_call([sys.executable, script_path])
                    
                    # Clean up
                    os.remove(script_path)
                    os.rmdir(temp_dir)
                    
                    self.show_message("FFmpeg installed successfully! Please restart the application.", duration=5000)
                    return True
                    
                except Exception as e:
                    # If automatic install fails, give download instructions
                    error_msg = f"Automatic install failed: {str(e)}\n\nPlease download FFmpeg manually from:\nhttps://ffmpeg.org/download.html"
                    self.show_message(error_msg, duration=10000)
                    return False
                    
            elif system == 'darwin':  # macOS
                # Recommend homebrew for macOS
                self.show_message("For macOS, please install FFmpeg using Homebrew:\n\nbrew install ffmpeg", duration=10000)
                return False
                
            elif system == 'linux':
                # For Linux, try apt-get
                try:
                    subprocess.check_call(["sudo", "apt", "update"])
                    subprocess.check_call(["sudo", "apt", "install", "-y", "ffmpeg"])
                    self.show_message("FFmpeg installed successfully!", duration=5000)
                    return True
                except Exception as e:
                    self.show_message(f"Could not install FFmpeg: {str(e)}\n\nPlease install FFmpeg using your distribution's package manager.", duration=10000)
                    return False
            else:
                self.show_message(f"Unsupported operating system: {system}. Please install FFmpeg manually.", duration=5000)
                return False
                
        except Exception as e:
            self.show_message(f"Error attempting to install FFmpeg: {str(e)}", duration=5000)
            return False

    def validate_spotify_url(self, url):
        """
        Validates if the provided URL is a valid Spotify track URL.
        
        Args:
            url (str): The URL to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        import re
        
        # Pattern for Spotify track URLs
        patterns = [
            r'https?://open\.spotify\.com/track/([a-zA-Z0-9]{22})(\?.*)?',
            r'spotify:track:([a-zA-Z0-9]{22})'
        ]
        
        for pattern in patterns:
            if re.match(pattern, url):
                return True
        
        return False

    def extract_spotify_track(self, spotify_url):
        """
        Extracts track information from Spotify URL and downloads a preview if available.
        If a preview is not available, suggests searching for the track on YouTube.
        
        Args:
            spotify_url (str): Spotify track URL
            
        Returns:
            tuple: (audio_path, track_info) or (None, track_info) if preview not available
        """
        try:
            # Update progress
            self.progress_label.configure(text="Connecting to Spotify API...")
            self.progress_bar.set(0.1)
            self.update()
            
            # Extract track ID from URL
            if 'track/' not in spotify_url:
                raise ValueError("Invalid Spotify URL. Must be a track URL (spotify.com/track/...)")
            
            track_id = spotify_url.split('track/')[1].split('?')[0].split('/')[0]
            
            # Initialize Spotify client
            try:
                client_credentials_manager = SpotifyClientCredentials(
                    client_id=SPOTIFY_CLIENT_ID,
                    client_secret=SPOTIFY_CLIENT_SECRET
                )
                sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            except Exception as auth_error:
                raise ValueError(f"Spotify authentication failed. Please check your API credentials: {str(auth_error)}")
            
            # Fetch track information
            self.progress_label.configure(text="Fetching track information...")
            self.progress_bar.set(0.2)
            self.update()
            
            try:
                track_info = sp.track(track_id)
            except Exception as track_error:
                raise ValueError(f"Failed to fetch track information: {str(track_error)}")
            
            # Extract track details
            track_name = track_info['name']
            artists = ', '.join([artist['name'] for artist in track_info['artists']])
            album = track_info['album']['name']
            preview_url = track_info['preview_url']
            
            # Update track info in UI
            self.track_title_var.set(track_name)
            self.track_artist_var.set(artists)
            
            # Check if preview is available
            if not preview_url:
                self.progress_label.configure(text="No preview available. Try searching on YouTube instead.")
                self.progress_bar.set(0)
                
                # Create a suggestion for YouTube search
                youtube_query = f"{artists} - {track_name} official"
                suggestion = {
                    "track_name": track_name,
                    "artists": artists,
                    "album": album,
                    "youtube_query": youtube_query
                }
                
                return None, suggestion
            
            # Download preview
            self.progress_label.configure(text=f"Downloading preview: {track_name}")
            self.progress_bar.set(0.4)
            self.update()
            
            # Create a temporary file to store the preview
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, "spotify_preview.mp3")
            
            # Download the preview using requests
            import requests
            response = requests.get(preview_url, stream=True)
            if response.status_code == 200:
                with open(audio_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                
                self.progress_label.configure(text="Preview downloaded successfully!")
                self.progress_bar.set(0.5)
                self.update()
                
                return audio_path, {
                    "track_name": track_name,
                    "artists": artists,
                    "album": album
                }
            else:
                raise ValueError(f"Failed to download preview: HTTP {response.status_code}")
            
        except Exception as e:
            error_message = f"Spotify extraction error: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            raise Exception(error_message)

    def save_spotify_credentials(self):
        """
        Saves Spotify API credentials to a file and updates global variables.
        """
        try:
            # Get values from UI
            client_id = self.client_id_var.get()
            client_secret = self.client_secret_var.get()
            
            # Update global variables
            global SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
            SPOTIFY_CLIENT_ID = client_id
            SPOTIFY_CLIENT_SECRET = client_secret
            
            # Save credentials to file
            with open("spotify_credentials.txt", "w") as f:
                f.write(f"Client ID: {client_id}\n")
                f.write(f"Client Secret: {client_secret}")
            
            self.show_message("Spotify credentials saved successfully!")
        except Exception as e:
            self.show_message(f"Error saving Spotify credentials: {str(e)}")

    def load_spotify_credentials(self):
        """
        Loads Spotify API credentials from a file.
        """
        try:
            # Check if file exists
            if not os.path.exists("spotify_credentials.txt"):
                return
                
            with open("spotify_credentials.txt", "r") as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    # Update global variables for API credentials
                    global SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
                    
                    client_id = lines[0].split(": ")[1].strip()
                    client_secret = lines[1].split(": ")[1].strip()
                    
                    # Update global variables
                    SPOTIFY_CLIENT_ID = client_id
                    SPOTIFY_CLIENT_SECRET = client_secret
                    
                    # Update UI variables if they exist
                    if hasattr(self, 'client_id_var'):
                        self.client_id_var.set(client_id)
                    if hasattr(self, 'client_secret_var'):
                        self.client_secret_var.set(client_secret)
        except Exception as e:
            print(f"Error loading Spotify credentials: {str(e)}")

    def convert_to_wav(self, input_file):
        """
        Converts any audio file to WAV format for better compatibility with librosa.
        Uses pydub which relies on ffmpeg under the hood.
        
        Args:
            input_file (str): Path to input audio file
            
        Returns:
            str: Path to converted WAV file
        """
        try:
            self.progress_label.configure(text="Converting audio format...")
            self.progress_bar.set(0.3)
            self.update()
            
            # Get directory and filename
            input_dir = os.path.dirname(input_file)
            filename = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(input_dir, f"{filename}.wav")
            
            # First, make sure FFmpeg is available before attempting conversions
            have_ffmpeg = self.is_ffmpeg_available()
            if not have_ffmpeg:
                self.progress_label.configure(text="FFmpeg not found. Attempting to download...")
                self.update()
                have_ffmpeg = self.download_ffmpeg_windows()
                if have_ffmpeg:
                    self.progress_label.configure(text="FFmpeg downloaded successfully. Continuing conversion...")
                    self.update()
            
            # For webm files, try using direct ffmpeg command first if available
            if input_file.lower().endswith('.webm') and have_ffmpeg:
                try:
                    self.progress_label.configure(text="Converting webm using FFmpeg...")
                    self.update()
                    
                    import subprocess
                    import shlex
                    
                    # Build ffmpeg command
                    ffmpeg_cmd = "ffmpeg"
                    if hasattr(self, 'ffmpeg_path') and self.ffmpeg_path:
                        ffmpeg_cmd = self.ffmpeg_path
                        
                    cmd = f'{ffmpeg_cmd} -i "{input_file}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{output_file}" -y'
                    
                    # Run the command
                    subprocess.check_call(shlex.split(cmd), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                    
                    # Check if conversion was successful
                    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                        self.progress_label.configure(text="Conversion complete!")
                        self.progress_bar.set(0.4)
                        self.update()
                        return output_file
                except Exception as ffmpeg_error:
                    print(f"Direct FFmpeg conversion failed: {str(ffmpeg_error)}")
            
            # Try using moviepy if available
            try:
                if input_file.lower().endswith(('.webm', '.mp4', '.mkv')):
                    self.progress_label.configure(text="Converting video using moviepy...")
                    self.update()
                    
                    # Using try/except for each import to better diagnose issues
                    try:
                        import moviepy
                    except ImportError:
                        print("Cannot import moviepy. Installing it...")
                        import subprocess
                        import sys
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy"])
                    
                    try:
                        from moviepy.editor import VideoFileClip
                    except ImportError as e:
                        print(f"Error importing VideoFileClip: {str(e)}")
                        raise
                    
                    video = VideoFileClip(input_file)
                    if video.audio is not None:
                        video.audio.write_audiofile(output_file, verbose=False, logger=None)
                        video.close()
                        if os.path.exists(output_file):
                            self.progress_label.configure(text="Conversion complete!")
                            self.progress_bar.set(0.4)
                            self.update()
                            return output_file
            except Exception as moviepy_error:
                print(f"MoviePy conversion failed: {str(moviepy_error)}")
            
            # Try pydub as another approach
            try:
                self.progress_label.configure(text="Converting using pydub...")
                self.update()
                
                # Import pydub - install if needed
                try:
                    from pydub import AudioSegment
                    AudioSegment.converter = r"path\to\ffmpeg.exe"
                    AudioSegment.ffprobe = r"path\to\ffprobe.exe"
                except ImportError:
                    print("Pydub not found. Installing...")
                    import subprocess
                    import sys
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub"])
                    from pydub import AudioSegment
                    AudioSegment.converter = r"path\to\ffmpeg.exe"
                    AudioSegment.ffprobe = r"path\to\ffprobe.exe"
                
                # For webm files, skip format detection and use 'webm' directly
                if input_file.lower().endswith('.webm'):
                    audio = AudioSegment.from_file(input_file, format="webm")
                else:
                    # For other files, try to determine format from extension
                    file_format = os.path.splitext(input_file)[1][1:].lower()
                    if not file_format:
                        file_format = "mp3"  # Default format
                    audio = AudioSegment.from_file(input_file, format=file_format)
                
                # Export as WAV
                audio.export(output_file, format="wav")
                
                self.progress_label.configure(text="Conversion complete!")
                self.progress_bar.set(0.4)
                self.update()
                
                return output_file
            except Exception as pydub_error:
                print(f"Pydub conversion failed: {str(pydub_error)}")
            
            # If all automated conversions fail, prompt user to install FFmpeg
            if not have_ffmpeg:
                self.progress_label.configure(text="Automatic conversion failed. Please install FFmpeg manually.")
                self.show_message(
                    "Conversion failed. Please install FFmpeg manually from https://ffmpeg.org/download.html",
                    duration=8000
                )
            else:
                self.progress_label.configure(text="All conversion methods failed.")
                self.show_message(
                    f"Could not convert file: {os.path.basename(input_file)}\nThe app will attempt to use the original file.",
                    duration=5000
                )
            
            # Return the original file as a last resort
            return input_file
                    
        except Exception as e:
            print(f"Error converting audio: {str(e)}")
            import traceback
            traceback.print_exc()
            return input_file  # Return original file if conversion fails
            
    def is_ffmpeg_available(self):
        """Check if FFmpeg is available in the system path"""
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE, 
                                    text=True, 
                                    shell=True)
            return result.returncode == 0
        except Exception:
            return False
            
    def download_ffmpeg_windows(self):
        """Download FFmpeg for Windows and add to app directory"""
        try:
            self.progress_label.configure(text="Downloading FFmpeg...")
            self.progress_bar.set(0.1)
            self.update()
            
            import urllib.request
            import zipfile
            import shutil
            
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "ffmpeg.zip")
            
            # Updated working URL for FFmpeg
            download_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
            
            # Download the zip file
            try:
                self.progress_label.configure(text="Downloading FFmpeg... (this may take a minute)")
                self.update()
                urllib.request.urlretrieve(download_url, zip_path)
            except Exception as e:
                print(f"Download error: {str(e)}")
                # Try alternative URL if first one fails
                try:
                    alt_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
                    self.progress_label.configure(text="Trying alternative download source...")
                    self.update()
                    urllib.request.urlretrieve(alt_url, zip_path)
                except Exception as alt_e:
                    print(f"Alternative download error: {str(alt_e)}")
                    return False
                
            # Extract the zip file
            if not os.path.exists(zip_path):
                print("Download failed - zip file not found")
                return False
                
            try:
                self.progress_label.configure(text="Extracting FFmpeg...")
                self.progress_bar.set(0.5)
                self.update()
                
                # Create the destination directory in the app directory
                app_dir = os.path.dirname(os.path.abspath(__file__))
                ffmpeg_dir = os.path.join(app_dir, "ffmpeg")
                os.makedirs(ffmpeg_dir, exist_ok=True)
                
                # Extract the zip
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find the ffmpeg.exe file in the extracted content
                ffmpeg_exe = None
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower() == "ffmpeg.exe":
                            ffmpeg_exe = os.path.join(root, file)
                            break
                    if ffmpeg_exe:
                        break
                
                if not ffmpeg_exe:
                    # If ffmpeg.exe wasn't found, look for any .exe files
                    bin_files = []
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.endswith('.exe'):
                                bin_files.append(os.path.join(root, file))
                                
                    if bin_files:
                        for bin_file in bin_files:
                            basename = os.path.basename(bin_file)
                            shutil.copy2(bin_file, os.path.join(ffmpeg_dir, basename))
                            if basename.lower() == "ffmpeg.exe":
                                ffmpeg_exe = os.path.join(ffmpeg_dir, basename)
                else:
                    # Copy ffmpeg.exe to our ffmpeg directory
                    shutil.copy2(ffmpeg_exe, os.path.join(ffmpeg_dir, "ffmpeg.exe"))
                    ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
                
                # Make sure we found and copied ffmpeg.exe
                if not ffmpeg_exe or not os.path.exists(ffmpeg_exe):
                    print("FFmpeg.exe not found in the downloaded package")
                    return False
                
                # Set the FFmpeg path for the application
                self.ffmpeg_path = ffmpeg_exe
                
                # Add the directory to the system PATH temporarily for this process
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
                
                self.progress_label.configure(text="FFmpeg downloaded and installed!")
                self.progress_bar.set(0.7)
                self.update()
                return True
                
            except Exception as e:
                print(f"Extraction error: {str(e)}")
                return False
            finally:
                # Clean up the temporary directory
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                    
        except Exception as e:
            print(f"FFmpeg download error: {str(e)}")
            return False

    def check_audio_dependencies(self):
        """
        Checks for required audio libraries and offers to install them if missing.
        """
        missing_deps = []
        
        try:
            import pydub
        except ImportError:
            missing_deps.append("pydub")
        
        try:
            import ffmpeg
        except ImportError:
            missing_deps.append("ffmpeg-python")
            
        try:
            import audioread
        except ImportError:
            missing_deps.append("audioread")
            
        # Check for moviepy for webm conversion    
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            missing_deps.append("moviepy")
            
        if missing_deps:
            # Create a pop-up dialog to install dependencies
            dialog = ctk.CTkToplevel(self)
            dialog.title("Missing Audio Libraries")
            dialog.geometry("400x250")
            dialog.resizable(False, False)
            dialog.transient(self)
            dialog.grab_set()
            
            # Center dialog
            self.update_idletasks()
            width = dialog.winfo_width()
            height = dialog.winfo_height()
            x = (dialog.winfo_screenwidth() // 2) - (width // 2)
            y = (dialog.winfo_screenheight() // 2) - (height // 2)
            dialog.geometry('{}x{}+{}+{}'.format(width, height, x, y))
            
            # Header
            header = ctk.CTkLabel(
                dialog,
                text="Missing Audio Libraries Detected",
                font=ctk.CTkFont(size=16, weight="bold")
            )
            header.pack(padx=20, pady=(20, 10))
            
            # Message
            message = ctk.CTkLabel(
                dialog,
                text=f"The following audio processing libraries are missing:\n• {', '.join(missing_deps)}\n\n"
                     f"These libraries are needed for YouTube and Spotify audio processing. "
                     f"Would you like to install them now?",
                wraplength=350,
                justify="center"
            )
            message.pack(padx=20, pady=10)
            
            # Button Frame
            button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            button_frame.pack(padx=20, pady=(10, 20), fill=tk.X)
            
            # Install button
            def install_deps():
                import subprocess
                import sys
                
                # Show installation progress
                progress_label.configure(text="Installing libraries...")
                progress_bar.configure(mode="indeterminate")
                progress_bar.start()
                button_frame.pack_forget()  # Hide buttons during install
                
                try:
                    # Run pip install command for missing dependencies
                    cmd = [sys.executable, "-m", "pip", "install"] + missing_deps
                    subprocess.check_call(cmd)
                    
                    # Update progress
                    progress_label.configure(text="Installation complete!")
                    progress_bar.stop()
                    progress_bar.configure(mode="determinate")
                    progress_bar.set(1.0)
                    
                    # Close button
                    close_btn = ctk.CTkButton(
                        dialog,
                        text="Close",
                        command=dialog.destroy
                    )
                    close_btn.pack(pady=(10, 20))
                    
                except Exception as e:
                    # Show error
                    progress_label.configure(text=f"Installation failed: {str(e)}")
                    progress_bar.stop()
                    progress_bar.configure(mode="determinate")
                    progress_bar.set(0)
                    
                    # Show buttons again
                    button_frame.pack(padx=20, pady=(10, 20), fill=tk.X)
            
            # Progress bar (hidden initially)
            progress_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            progress_frame.pack(padx=20, pady=(0, 10), fill=tk.X)
            
            progress_label = ctk.CTkLabel(
                progress_frame,
                text="",
                font=ctk.CTkFont(size=12)
            )
            progress_label.pack(anchor=tk.W, pady=(0, 5))
            
            progress_bar = ctk.CTkProgressBar(progress_frame)
            progress_bar.pack(fill=tk.X)
            progress_bar.set(0)
            
            # Install button
            install_btn = ctk.CTkButton(
                button_frame,
                text="Install",
                command=install_deps,
                fg_color=self.colors["primary"]
            )
            install_btn.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
            
            # Skip button
            skip_btn = ctk.CTkButton(
                button_frame,
                text="Skip",
                command=dialog.destroy,
                fg_color=self.colors["text_secondary"]
            )
            skip_btn.pack(side=tk.RIGHT, padx=(5, 0), fill=tk.X, expand=True)


# Run the application
if __name__ == "__main__":
    app = MusicEmotionRecognitionApp()

    # Add additional UI elements
    app.create_emotion_color_legend()
    app.add_export_button()
    app.add_help_menu()

    app.mainloop()