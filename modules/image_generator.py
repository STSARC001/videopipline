"""
Image Generator Module
Uses Stable Diffusion with ControlNet for consistent image generation
"""
import os
import logging
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import cv2
from tqdm import tqdm
import config

class ImageGenerator:
    """
    Generates images using Stable Diffusion with ControlNet to ensure consistency
    across scenes.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load model configuration
        self.model_id = config.MODELS["stable_diffusion"]["model_id"]
        self.controlnet_model = config.MODELS["stable_diffusion"]["controlnet_model"]
        self.height = config.MODELS["stable_diffusion"]["height"]
        self.width = config.MODELS["stable_diffusion"]["width"]
        self.steps = config.MODELS["stable_diffusion"]["num_inference_steps"]
        self.guidance_scale = config.MODELS["stable_diffusion"]["guidance_scale"]
        
        self.logger.info(f"Initializing ImageGenerator with model: {self.model_id}")
        self.logger.info(f"Using ControlNet model: {self.controlnet_model}")
    
    def _load_pipeline(self):
        """Load the Stable Diffusion pipeline with ControlNet"""
        self.logger.info("Loading Stable Diffusion pipeline with ControlNet")
        
        try:
            # Set device to CUDA if available, otherwise CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {device}")
            
            # Load ControlNet model
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            # Load StableDiffusion with ControlNet
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=controlnet,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None  # Disable safety checker for performance
            ).to(device)
            
            # Enable memory optimization
            pipeline.enable_xformers_memory_efficient_attention()
            
            return pipeline, device
            
        except Exception as e:
            self.logger.error(f"Error loading pipeline: {str(e)}")
            raise
    
    def _prepare_control_image(self, reference_image=None, control_mode="canny"):
        """
        Prepare a control image for ControlNet
        
        Args:
            reference_image: Optional reference image to use as base
            control_mode: Type of control to apply (canny, depth, pose, etc.)
            
        Returns:
            control_image: Processed control image
        """
        # If no reference image provided, create a blank one
        if reference_image is None:
            reference_image = Image.new("RGB", (self.width, self.height), color="white")
        
        # Convert PIL image to numpy array
        image_np = np.array(reference_image)
        
        if control_mode == "canny":
            # Apply Canny edge detection
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            image_np = cv2.Canny(image_np, 100, 200)
            image_np = cv2.dilate(image_np, np.ones((1, 1), np.uint8), iterations=1)
            image_np = cv2.GaussianBlur(image_np, (5, 5), 0)
            # Convert back to RGB
            image_np = np.stack([image_np, image_np, image_np], axis=2)
        
        # Convert numpy array back to PIL image
        control_image = Image.fromarray(image_np)
        return control_image
    
    def _enhance_prompt(self, description):
        """
        Enhance a scene description into a better prompt for image generation
        
        Args:
            description: Scene description text
            
        Returns:
            enhanced_prompt: Improved prompt for Stable Diffusion
        """
        # Add quality boosters and style directives to the prompt
        quality_boosters = [
            "highly detailed", 
            "8k resolution",
            "cinematic lighting",
            "professional photography"
        ]
        
        # Add style directive based on genre (could be made more sophisticated)
        style_directive = "digital art"
        
        # Combine elements into enhanced prompt
        boosters_text = ", ".join(quality_boosters)
        enhanced_prompt = f"{description}, {boosters_text}, {style_directive}"
        
        # Add negative prompt elements to avoid
        negative_prompt = "blurry, distorted, disfigured, ugly, bad anatomy, watermark, signature, text"
        
        return enhanced_prompt, negative_prompt
    
    def generate(self, script, scene_descriptions, output_dir):
        """
        Generate images based on script and scene descriptions
        
        Args:
            script (str): The complete script
            scene_descriptions (list): List of scene descriptions
            output_dir (Path): Directory to save generated images
            
        Returns:
            dict: Generated content including image paths
        """
        # Create images directory
        images_dir = output_dir / "images"
        os.makedirs(images_dir, exist_ok=True)
        
        self.logger.info(f"Generating images for {len(scene_descriptions)} scenes")
        
        # Load pipeline
        pipeline, device = self._load_pipeline()
        
        # Store generated images
        images = []
        image_paths = []
        
        # Create a reference image for consistency (initially None)
        reference_image = None
        
        # Generate images for each scene
        for i, description in enumerate(tqdm(scene_descriptions, desc="Generating images")):
            self.logger.info(f"Generating image for scene {i+1}/{len(scene_descriptions)}")
            
            # Create enhanced prompt for better image quality
            prompt, negative_prompt = self._enhance_prompt(description)
            self.logger.debug(f"Enhanced prompt: {prompt}")
            
            # Prepare control image from reference (if available)
            control_image = self._prepare_control_image(reference_image)
            
            # Generate image with ControlNet
            output = pipeline(
                prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                height=self.height,
                width=self.width
            )
            
            # Save generated image
            image = output.images[0]
            image_path = images_dir / f"scene_{i+1:02d}.png"
            image.save(image_path)
            
            # Update reference image for consistency in next generation
            reference_image = image
            
            # Store results
            images.append(image)
            image_paths.append(str(image_path))
            
            self.logger.info(f"Saved image to {image_path}")
        
        self.logger.info(f"Generated {len(images)} images")
        
        # Return generated content
        return {
            "images": images,
            "image_paths": image_paths
        }
