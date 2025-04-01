#!/usr/bin/env python3
"""
YouTube Automation Pipeline - Main Entry Point
A complete pipeline for generating YouTube content using AI models
"""

import os
import argparse
import logging
from pathlib import Path
import uuid
import time

from modules.prompt_generator import PromptGenerator
from modules.image_generator import ImageGenerator
from modules.animator import Animator
from modules.voice_generator import VoiceGenerator
from modules.video_compiler import VideoCompiler
from modules.storage import GoogleDriveStorage
import config

def setup_logging():
    """Set up logging configuration"""
    log_level = getattr(logging, config.LOGGING["level"])
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config.LOGGING["file"]),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="YouTube Automation Pipeline")
    parser.add_argument(
        "--genre", 
        choices=config.CONTENT["genres"],
        default=config.CONTENT["default_genre"],
        help="Content genre (e.g., SciFi, Fantasy)"
    )
    parser.add_argument(
        "--content_type", 
        choices=["story", "animation", "news", "comedy"],
        default=config.CONTENT["content_type"],
        help="Type of content to generate"
    )
    parser.add_argument(
        "--video_length", 
        type=int,
        default=config.CONTENT["video_length"],
        help="Target video length in seconds"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path,
        default=config.OUTPUT_DIR,
        help="Directory to store generated content"
    )
    parser.add_argument(
        "--upload", 
        action="store_true",
        help="Upload to Google Drive after completion"
    )
    parser.add_argument(
        "--skip_steps", 
        nargs="+",
        choices=["prompt", "image", "animation", "voice", "video", "upload"],
        help="Skip specific pipeline steps (useful for testing)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with additional logging"
    )
    return parser.parse_args()

def main():
    """Main function to run the pipeline"""
    args = parse_arguments()
    logger = setup_logging()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create a unique run ID for this pipeline execution
    run_id = str(uuid.uuid4())[:8]
    output_dir = args.output_dir / run_id
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting YouTube Automation Pipeline (ID: {run_id})")
    logger.info(f"Genre: {args.genre}, Content Type: {args.content_type}")
    
    start_time = time.time()
    pipeline_data = {
        "run_id": run_id,
        "output_dir": output_dir,
        "genre": args.genre,
        "content_type": args.content_type,
        "video_length": args.video_length,
    }
    
    skip_steps = args.skip_steps or []
    
    try:
        # Step 1: Generate prompts and script
        if "prompt" not in skip_steps:
            logger.info("Step 1: Generating prompts and script")
            prompt_generator = PromptGenerator()
            pipeline_data.update(prompt_generator.generate(
                genre=args.genre,
                content_type=args.content_type,
                output_dir=output_dir
            ))
        
        # Step 2: Generate images based on script
        if "image" not in skip_steps:
            logger.info("Step 2: Generating images")
            image_generator = ImageGenerator()
            pipeline_data.update(image_generator.generate(
                script=pipeline_data.get("script", ""),
                scene_descriptions=pipeline_data.get("scene_descriptions", []),
                output_dir=output_dir
            ))
        
        # Step 3: Animate the images
        if "animation" not in skip_steps:
            logger.info("Step 3: Animating images")
            animator = Animator()
            pipeline_data.update(animator.animate(
                images=pipeline_data.get("images", []),
                scene_timings=pipeline_data.get("scene_timings", []),
                output_dir=output_dir
            ))
        
        # Step 4: Generate voiceover
        if "voice" not in skip_steps:
            logger.info("Step 4: Generating voiceover")
            voice_generator = VoiceGenerator()
            pipeline_data.update(voice_generator.generate(
                script=pipeline_data.get("script", ""),
                scene_dialogs=pipeline_data.get("scene_dialogs", []),
                output_dir=output_dir
            ))
        
        # Step 5: Compile final video
        if "video" not in skip_steps:
            logger.info("Step 5: Compiling video")
            video_compiler = VideoCompiler()
            pipeline_data.update(video_compiler.compile(
                animations=pipeline_data.get("animations", []),
                audio_files=pipeline_data.get("audio_files", []),
                script=pipeline_data.get("script", ""),
                output_dir=output_dir
            ))
        
        # Step 6: Upload to Google Drive
        if "upload" not in skip_steps and args.upload:
            logger.info("Step 6: Uploading to Google Drive")
            drive_storage = GoogleDriveStorage()
            pipeline_data.update(drive_storage.upload(
                video_file=pipeline_data.get("final_video", ""),
                script=pipeline_data.get("script", ""),
                images=pipeline_data.get("images", []),
                metadata=pipeline_data.get("metadata", {}),
                output_dir=output_dir
            ))
        
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
        logger.info(f"Output directory: {output_dir}")
        
        if "final_video" in pipeline_data:
            logger.info(f"Final video: {pipeline_data['final_video']}")
        
        if "drive_url" in pipeline_data:
            logger.info(f"Google Drive URL: {pipeline_data['drive_url']}")
        
        return pipeline_data
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise
            
if __name__ == "__main__":
    main()
