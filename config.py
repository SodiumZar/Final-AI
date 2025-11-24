# Configuration file for Retina Analysis System

# Google GenAI Configuration
GENAI_API_KEY = "AIzaSyAVK6N6iaN3yDDZVBoDe9isPVY0UD8IqvA"
GENAI_MODEL = "gemini-2.5-flash"

# Flask Configuration
DEBUG_MODE = True
HOST = "0.0.0.0"
PORT = 5000

# File Paths
UPLOAD_FOLDER = "static/uploads"
PREDICTION_FOLDER = "static/predictions"
OVERLAY_FOLDER = "static/overlays"
CHECKPOINT_PATH = "files/checkpoint.pth"

# Image Processing
IMAGE_SIZE = 512  # Model expects 512x512 images
OVERLAY_ALPHA = 0.5  # Transparency for overlay (0-1)

# Vessel Analysis Thresholds
NORMAL_DENSITY_MIN = 10.0  # Minimum normal vessel density %
NORMAL_DENSITY_MAX = 20.0  # Maximum normal vessel density %
NORMAL_TORTUOSITY_MAX = 1.3  # Maximum normal tortuosity score

# AI Analysis Configuration
AI_TIMEOUT = 30  # Timeout for GenAI requests (seconds)
USE_FALLBACK_SUMMARY = True  # Use fallback if GenAI fails
