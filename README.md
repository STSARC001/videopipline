# YouTube Automation Pipeline

A comprehensive end-to-end pipeline for generating YouTube content using AI models.

## Overview

This project implements a multi-model automation pipeline for YouTube content creation, integrating several AI models:

1. **Integrated Story & Image Generation with Google's Gemini** (NEW!): Generate both story and corresponding images in one step using Google's state-of-the-art Gemini API.
2. **Multi-Model Prompt Generation**: Alternatively uses GPT-Neo or GPT-J to generate creative prompts and scripts.
3. **Multi-Modal Image Generation**: Can also leverage Stable Diffusion with ControlNet for consistent image generation.
4. **Hyper-Realistic Image Animation**: Animates static images using Deforms/EbSynth and enhances with RIFE for smooth transitions.
5. **AI Voiceover Generation**: Creates realistic voiceovers with emotion using Coqui TTS or Bark AI.
6. **Advanced Video Compilation**: Combines all elements with FFmpeg and MoviePy for professional video production.
7. **Google Drive Integration**: Automatically uploads content with SEO-optimized metadata.

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

5. Set up Gemini API (Optional but recommended):
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey) to get your Gemini API key
   - Add the API key to your `.env` file as `GEMINI_API_KEY=your_api_key_here`

## Usage

Run the pipeline with default settings:

```
python main.py
```

### Using the Gemini Integration (Recommended)

Generate content with Gemini API (combines story and image generation):

```
python main.py --use_gemini --genre Fantasy
```

Use a custom prompt with Gemini:

```
python main.py --use_gemini --custom_prompt "Generate a story about a white baby goat going on an adventure in a farm in a 3d cartoon animation style. For each scene, generate an image."
```

Specify a visual style for image generation:

```
python main.py --use_gemini --visual_style "cinematic 3D animation style" --genre Comedy
```

### Using the Traditional Pipeline

Run the traditional pipeline with separate story and image generation:

```
python main.py --use_gemini=False --genre SciFi --content_type story
```

For full command-line options:

```
python main.py --help
```

## Configuration

All configuration settings can be adjusted in the `.env` file or passed as command-line arguments. Key configuration sections include:

- AI Models (Gemini, LLM, Stable Diffusion, ControlNet, Voice)
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

To run this pipeline on Google Colab with Gemini integration, follow these steps:

1. Create a new Colab notebook at [colab.research.google.com](https://colab.research.google.com)

2. Clone the repository in a Colab cell:
   ```
   !git clone https://github.com/STSARC001/videopipline.git
   %cd videopipline
   ```

3. Install the required dependencies:
   ```
   !pip install -r requirements.txt
   
   # Ensure Google Generative AI library is installed for Gemini
   !pip install google-generativeai
   ```

4. **Fix dependency conflicts** (crucial to prevent errors):
   ```python
   # Fix NumPy version incompatibility
   !pip uninstall -y numpy
   !pip install numpy==1.24.3
   
   # Install additional dependencies that may not be in requirements.txt
   !pip install moviepy tqdm nltk librosa soundfile
   
   # Install xformers for memory-efficient attention in Stable Diffusion (if using)
   !pip install xformers
   
   # Restart runtime (run this cell, then continue from the next cell after runtime restarts)
   import os
   os.kill(os.getpid(), 9)
   ```

5. Create a `.env` file with your Gemini API key:
   ```
   %%writefile .env
   GEMINI_API_KEY=your_api_key_here
   USE_GEMINI=True
   VIDEO_LENGTH=60
   ANIMATION_FPS=24
   USE_RIFE=True
   ```

6. Mount Google Drive to save output:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Create output directory in Drive
   !mkdir -p /content/drive/MyDrive/youtube_automation_output
   ```

7. Modify the config.py file to use Google Drive for output:
   ```python
   # Run this to update the config
   %%writefile config_override.py
   import os
   from pathlib import Path
   
   # Update output directory to use Google Drive
   OUTPUT_DIR = Path("/content/drive/MyDrive/youtube_automation_output")
   os.makedirs(OUTPUT_DIR, exist_ok=True)
   
   # Execute the override
   %run config_override.py
   ```

8. Test the Gemini integration:
   ```python
   # Run the Gemini test script
   !python test_gemini_generator.py
   
   # Check the output
   !ls -la output/gemini_test/images/
   !ls -la output/gemini_test/scripts/
   ```

9. Run the full pipeline with Gemini:
   ```python
   # Example: Generate a story about a baby goat on a farm adventure
   !python main.py --use_gemini --genre Fantasy --custom_prompt "Generate a story about a white baby goat going on an adventure in a farm in a 3d cartoon animation style. For each scene, generate an image." --output_dir /content/drive/MyDrive/youtube_automation_output
   ```

10. Organize the output in Google Drive:
    ```python
    # Automatically organize the content in Google Drive
    !python -c "
    import os, shutil
    from pathlib import Path
    
    # Define paths
    output_base = Path('/content/drive/MyDrive/youtube_automation_output')
    latest_run = sorted([d for d in output_base.iterdir() if d.is_dir() and not d.name.startswith('.')])[-1]
    
    # Create organized folder
    organized_dir = output_base / 'organized' / latest_run.name
    os.makedirs(organized_dir, exist_ok=True)
    
    # Copy final video and metadata
    if (latest_run / 'videos' / 'final_video.mp4').exists():
        shutil.copy2(latest_run / 'videos' / 'final_video.mp4', organized_dir)
    if (latest_run / 'videos' / 'metadata.json').exists():
        shutil.copy2(latest_run / 'videos' / 'metadata.json', organized_dir)
    if (latest_run / 'videos' / 'thumbnail.jpg').exists():
        shutil.copy2(latest_run / 'videos' / 'thumbnail.jpg', organized_dir)
    
    print(f'Content organized at: {organized_dir}')
    "
    ```

11. Additional Colab-specific optimizations:

    a. Use GPU acceleration (recommended for faster processing):
    ```python
    # Check if GPU is available and enabled
    !nvidia-smi
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"GPU device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    ```

    b. Increase Colab memory limits for video processing:
    ```python
    # Increase Colab memory limits
    import torch
    torch.cuda.empty_cache()
    
    # Limit memory usage for video processing
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
    # Set environment variables for better performance
    os.environ["PYTHONUNBUFFERED"] = "1"
    ```

12. Share your generated content:
    ```python
    # Generate a shareable link for your video
    from google.colab import files
    
    # Download video locally if you want
    files.download(f"{organized_dir}/final_video.mp4")
    
    # Or create a shareable link using Drive API
    from google.colab import auth
    from googleapiclient.discovery import build
    
    auth.authenticate_user()
    drive_service = build('drive', 'v3')
    
    # Get file ID from path
    import re
    organized_dir_str = str(organized_dir)
    drive_pattern = r'/content/drive/MyDrive/(.+)'
    rel_path = re.search(drive_pattern, organized_dir_str).group(1)
    
    # Search for the file
    query = f"name = 'final_video.mp4' and '{rel_path}' in parents"
    response = drive_service.files().list(q=query).execute()
    
    if response.get('files'):
        file_id = response['files'][0]['id']
        # Make the file publicly accessible
        drive_service.permissions().create(
            fileId=file_id,
            body={'type': 'anyone', 'role': 'reader'},
        ).execute()
        print(f"Shareable link: https://drive.google.com/file/d/{file_id}/view")
    else:
        print("File not found. Make sure the video was generated successfully.")
    ```

> **Important Notes for Google Colab:**
> 
> 1. **Memory Management:** Colab has memory limitations. For longer videos or higher resolution, consider:
>    - Reducing the number of frames
>    - Lowering the resolution in config.py
>    - Processing fewer scenes at once
>
> 2. **Session Timeouts:** Google Colab sessions timeout after idle periods or extended runs. To prevent losing work:
>    - Save intermediate outputs to Google Drive
>    - Use `%%capture` for long-running cells to reduce browser load
>    - Stay active in the notebook to prevent idle timeouts
>
> 3. **GPU Allocation:** Colab may allocate different GPUs each session. Check with `!nvidia-smi` and adjust parameters accordingly.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project utilizes several open-source models and libraries:
- Google's Gemini API for integrated story and image generation
- GPT-Neo/GPT-J by EleutherAI
- Stable Diffusion by CompVis/Stability AI
- ControlNet by lllyasviel
- Coqui TTS and Bark AI
- FFmpeg and MoviePy
