# YouTube Automation Pipeline

A comprehensive end-to-end pipeline for generating YouTube content using AI models.

## Overview

This project implements a multi-model automation pipeline for YouTube content creation, integrating several open-source AI models:

1. **Multi-Model Prompt Generation**: Uses GPT-Neo or GPT-J to generate creative prompts and scripts.
2. **Multi-Modal Story and Image Generation**: Leverages Stable Diffusion with ControlNet for consistent image generation.
3. **Hyper-Realistic Image Animation**: Animates static images using Deforms/EbSynth and enhances with RIFE for smooth transitions.
4. **AI Voiceover Generation**: Creates realistic voiceovers with emotion using Coqui TTS or Bark AI.
5. **Advanced Video Compilation**: Combines all elements with FFmpeg and MoviePy for professional video production.
6. **Google Drive Integration**: Automatically uploads content with SEO-optimized metadata.

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/youtube-automation-pipeline.git
cd youtube-automation-pipeline
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your configuration (see `.env.example`).

4. Set up Google Drive API:
   - Create a project in the [Google Cloud Console](https://console.cloud.google.com/)
   - Enable the Google Drive API
   - Create OAuth 2.0 credentials and download as `credentials.json`
   - Place `credentials.json` in the project root directory

## Usage

Run the pipeline with default settings:

```
python main.py
```

Specify a genre and content type:

```
python main.py --genre SciFi --content_type story
```

For full command-line options:

```
python main.py --help
```

## Configuration

All configuration settings can be adjusted in the `.env` file or passed as command-line arguments. Key configuration sections include:

- AI Models (LLM, Stable Diffusion, ControlNet, Voice)
- Animation settings (FPS, frame interpolation)
- Video compilation parameters (resolution, bitrate)
- Google Drive integration settings

See `config.py` for all available options.

## Output

The pipeline generates:

1. Text scripts and prompts in the `output/scripts/` directory
2. Generated images in `output/images/`
3. Animated sequences in `output/animations/`
4. Audio files in `output/audio/`
5. Final videos in `output/videos/`
6. All content is automatically uploaded to Google Drive if configured

## Running on Google Colab

To run this pipeline on Google Colab, follow these steps:

1. Create a new Colab notebook at [colab.research.google.com](https://colab.research.google.com)

2. Clone the repository in a Colab cell:
   ```
   !git clone https://github.com/STSARC001/videopipline.git
   %cd videopipline
   ```

3. Install the required dependencies:
   ```
   !pip install -r requirements.txt
   ```

4. **Fix dependency conflicts** (crucial to prevent errors):
   ```python
   # Fix NumPy version incompatibility
   !pip uninstall -y numpy
   !pip install numpy==1.24.3
   
   # Restart runtime (run this cell, then continue from the next cell after runtime restarts)
   import os
   os.kill(os.getpid(), 9)
   ```

5. Use the simplified Google Drive authentication for Colab:
   ```python
   # Configure Google Drive authentication for Colab
   def setup_colab_drive_auth():
       """Set up Google Drive authentication for Colab"""
       from google.colab import auth
       from googleapiclient.discovery import build

       # Authenticate with Google
       auth.authenticate_user()

       # Build the Drive service directly
       drive_service = build('drive', 'v3')

       # Patch the GoogleDriveStorage class
       import modules.storage as storage
       storage.GoogleDriveStorage._authenticate = lambda self: drive_service

       print("✅ Google Drive authentication configured for Colab")

   # Run the function
   setup_colab_drive_auth()
   ```

6. Install NLTK resources:
   ```python
   # Install NLTK resources
   import nltk
   nltk.download('punkt')
   
   # Fix for punkt_tab error
   import ssl
   try:
       _create_unverified_https_context = ssl._create_unverified_context
   except AttributeError:
       pass
   else:
       ssl._create_default_https_context = _create_unverified_https_context
       
   # Download additional required NLTK resources
   nltk.download('punkt_tab')
   print("✅ NLTK resources downloaded")
   ```

7. Mount Google Drive to save output:
   ```
   from google.colab import drive
   drive.mount('/content/drive')
   ```

8. Create a `.env` file with your configuration:
   ```
   %%writefile .env
   DRIVE_FOLDER_NAME=YouTubeAutomation
   OUTPUT_DIR=/content/drive/MyDrive/youtube_automation_output
   ```

9. Run the pipeline:
   ```
   !python main.py --genre SciFi --content_type story --upload
   ```

10. Access your generated content in the specified Google Drive folder.

**Note**: Colab provides free GPU acceleration which is perfect for running the AI models in this pipeline. Select Runtime > Change runtime type > Hardware accelerator > GPU for best performance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project utilizes several open-source models and libraries:
- GPT-Neo/GPT-J by EleutherAI
- Stable Diffusion by CompVis/Stability AI
- ControlNet by lllyasviel
- Coqui TTS and Bark AI
- FFmpeg and MoviePy
