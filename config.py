"""
Configuration settings for the YouTube Automation Pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

# Create necessary directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# AI Models Configuration
MODELS = {
    # LLM configs
    "llm": {
        "model_name": os.getenv("LLM_MODEL_NAME", "EleutherAI/gpt-neo-1.3B"),  # GPT-Neo or GPT-J
        "max_length": int(os.getenv("LLM_MAX_LENGTH", 1000)),
        "temperature": float(os.getenv("LLM_TEMPERATURE", 0.8)),
        "top_p": float(os.getenv("LLM_TOP_P", 0.9)),
    },
    
    # Gemini API configs
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY", "AIzaSyBs_YOa0m9dAhDKw45esHSlXbZ3WIDP6MI"),  # Default key from example
        "model_name": os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro"),  # For text generation
        "image_model": os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.0-flash-exp-image-generation"),  # For image generation
    },
    
    # Stable Diffusion configs
    "stable_diffusion": {
        "model_id": os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5"),
        "controlnet_model": os.getenv("CONTROLNET_MODEL", "lllyasviel/sd-controlnet-canny"),
        "height": int(os.getenv("SD_HEIGHT", 512)),
        "width": int(os.getenv("SD_WIDTH", 512)),
        "num_inference_steps": int(os.getenv("SD_STEPS", 50)),
        "guidance_scale": float(os.getenv("SD_GUIDANCE_SCALE", 7.5)),
    },
    
    # Voice Generation configs
    "voice": {
        "model_name": os.getenv("VOICE_MODEL", "tts_models/en/ljspeech/tacotron2-DDC"),  # Coqui TTS
        "use_bark": os.getenv("USE_BARK", "False").lower() == "true",  # Whether to use Bark AI instead
    }
}

# Animation Settings
ANIMATION = {
    "fps": int(os.getenv("ANIMATION_FPS", 24)),
    "use_rife": os.getenv("USE_RIFE", "True").lower() == "true",  # Use RIFE for frame interpolation
    "rife_factor": int(os.getenv("RIFE_FACTOR", 2)),  # 2x frame interpolation
    "use_depth": os.getenv("USE_DEPTH", "False").lower() == "true",  # Use depth mapping for parallax
}

# Video Compilation Settings
VIDEO = {
    "resolution": (int(os.getenv("VIDEO_WIDTH", 1920)), int(os.getenv("VIDEO_HEIGHT", 1080))),
    "bitrate": os.getenv("VIDEO_BITRATE", "5000k"),
    "codec": os.getenv("VIDEO_CODEC", "libx264"),
    "audio_codec": os.getenv("AUDIO_CODEC", "aac"),
    "audio_bitrate": os.getenv("AUDIO_BITRATE", "192k"),
}

# Google Drive Settings
GOOGLE_DRIVE = {
    "credentials_file": os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json"),
    "token_file": os.getenv("GOOGLE_TOKEN_FILE", "token.json"),
    "folder_name": os.getenv("DRIVE_FOLDER_NAME", "YoutubeAutomation"),
}

# Content Generation
CONTENT = {
    "genres": ["SciFi", "Fantasy", "Thriller", "Comedy", "Drama", "Educational"],
    "default_genre": os.getenv("DEFAULT_GENRE", "SciFi"),
    "content_type": os.getenv("CONTENT_TYPE", "story"),  # story, animation, news, comedy
    "video_length": int(os.getenv("VIDEO_LENGTH", 60)),  # Target video length in seconds
    "use_gemini": os.getenv("USE_GEMINI", "True").lower() == "true",  # Whether to use Gemini API instead of separate models
    "visual_style": os.getenv("VISUAL_STYLE", "3d cartoon animation style"),  # Visual style for image generation
}

# Logging Configuration
LOGGING = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "file": os.getenv("LOG_FILE", BASE_DIR / "automation.log"),
}
