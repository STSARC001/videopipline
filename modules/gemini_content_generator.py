"""
Gemini Content Generator Module
Uses Google's Gemini API for generating both story content and images in one integrated process
"""
import os
import logging
import json
import re
import base64
import mimetypes
from pathlib import Path
from google import genai
from google.genai import types
import config

class GeminiContentGenerator:
    """
    Uses Google's Gemini API to generate both story content and corresponding images
    in a single integrated process, replacing the separate prompt and image generators.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load Gemini configuration
        self.api_key = config.MODELS["gemini"]["api_key"]
        self.model_name = config.MODELS["gemini"]["model_name"]
        self.image_model = config.MODELS["gemini"]["image_model"]
        
        # Set up API key
        os.environ["GEMINI_API_KEY"] = self.api_key
        
        # Initialize client
        self.client = genai.Client(api_key=self.api_key)
        
        self.logger.info(f"Initializing GeminiContentGenerator with model: {self.model_name}")
        
        # Define genre-specific templates
        self.genre_templates = {
            "SciFi": "Generate a science fiction story about {topic} in a future where {setting}.",
            "Fantasy": "Generate a fantasy tale where {topic} in a world where {setting}.",
            "Thriller": "Generate a suspenseful thriller involving {topic} where {setting}.",
            "Comedy": "Generate a humorous story about {topic} in a situation where {setting}.",
            "Drama": "Generate an emotional drama centered on {topic} where {setting}.",
            "Educational": "Explain {topic} in an engaging way, focusing on {setting}."
        }
        
        # Topics and settings for random generation (reused from original PromptGenerator)
        self.topics = {
            "SciFi": ["space exploration", "artificial intelligence", "time travel", "alien contact", "dystopian society"],
            "Fantasy": ["magical creatures", "a prophesied hero", "an ancient artifact", "warring kingdoms", "elemental powers"],
            "Thriller": ["a mysterious disappearance", "a perfect crime", "a deadly chase", "a psychological mind game", "a conspiracy"],
            "Comedy": ["a case of mistaken identity", "an awkward reunion", "a fish out of water", "a ridiculous bet", "a day gone wrong"],
            "Drama": ["a family secret", "an impossible choice", "a journey of redemption", "a life-changing event", "a forbidden relationship"],
            "Educational": ["quantum physics", "historical events", "biology concepts", "technological innovations", "psychology theories"]
        }
        
        self.settings = {
            "SciFi": ["humanity has colonized Mars", "AI has achieved consciousness", "time is no longer linear", "resources are scarce", "humans have evolved new abilities"],
            "Fantasy": ["magic is fading from existence", "dragons have returned", "gods walk among mortals", "prophecies always come true", "nature has awakened"],
            "Thriller": ["no one can be trusted", "time is running out", "the truth is hidden in plain sight", "the past refuses to stay buried", "everyone has something to hide"],
            "Comedy": ["everything keeps going hilariously wrong", "cultural misunderstandings abound", "the stakes keep escalating", "nobody knows what they're doing", "coincidences pile up"],
            "Drama": ["the past and present collide", "moral boundaries are tested", "love and duty conflict", "sacrifice seems inevitable", "hope persists against all odds"],
            "Educational": ["it connects to everyday life", "it changed the course of history", "misconceptions are common", "recent discoveries have changed our understanding", "it affects everyone"]
        }
        
        # Image styles for each genre
        self.image_styles = {
            "SciFi": "in a futuristic sci-fi style with high-tech elements",
            "Fantasy": "in a magical fantasy style with ethereal lighting",
            "Thriller": "in a dark, suspenseful style with dramatic shadows",
            "Comedy": "in a bright, colorful cartoon style with exaggerated features",
            "Drama": "in a realistic style with emotional lighting and muted colors",
            "Educational": "in a clear, detailed infographic style with vibrant colors"
        }
    
    def save_binary_file(self, file_path, data):
        """Save binary data to a file"""
        with open(file_path, "wb") as f:
            f.write(data)
        self.logger.debug(f"Saved binary file to: {file_path}")
    
    def _get_prompt_template(self, genre, content_type, visual_style="3d cartoon animation"):
        """Get a prompt template based on genre and content type"""
        base_template = self.genre_templates.get(genre, self.genre_templates["SciFi"])
        image_style = self.image_styles.get(genre, "")
        
        # Add content type-specific instructions and request for images
        if content_type == "story":
            return base_template + f" Write a short story with a clear beginning, middle, and end, divided into 5-7 distinct scenes. For each scene, generate an image {image_style} {visual_style}."
        elif content_type == "animation":
            return base_template + f" Write a visual script with 5-7 distinct scenes that could be animated. For each scene, generate an image {image_style} {visual_style}."
        elif content_type == "news":
            return base_template + f" Write this as a news report with facts, interviews, and analysis. For each section, generate an image {image_style} {visual_style}."
        elif content_type == "comedy":
            return base_template + f" Make this a comedy sketch with punchlines and humorous situations. For each scene, generate an image {image_style} {visual_style}."
        else:
            return base_template + f" Divide into 5-7 distinct scenes. For each scene, generate an image {image_style} {visual_style}."
    
    def _extract_scenes_and_images(self, response_text):
        """
        Extract scene descriptions, dialogs, and image generation prompts from the response
        
        The expected format from Gemini would have scenes marked with headers and potentially
        image descriptions/prompts.
        """
        # Parse the response to extract scenes
        scene_pattern = r"(?i)(?:scene|chapter)\s+\d+[:\.]?(.*?)(?=(?:scene|chapter)\s+\d+[:\.]?|$)"
        scenes = re.findall(scene_pattern, response_text, re.DOTALL)
        
        # If no scenes found using the pattern, split by paragraphs and group
        if not scenes:
            paragraphs = response_text.split("\n\n")
            # Filter out very short paragraphs
            paragraphs = [p for p in paragraphs if len(p.strip()) > 50]
            
            # Group paragraphs into rough scenes (2-3 paragraphs per scene)
            scenes = []
            for i in range(0, len(paragraphs), 2):
                scene = "\n\n".join(paragraphs[i:i+2])
                scenes.append(scene)
        
        scene_descriptions = []
        scene_dialogs = []
        scene_image_prompts = []
        
        # Process each scene
        for i, scene in enumerate(scenes):
            # Extract dialog (text in quotes)
            dialog_pattern = r'"([^"]*)"'
            dialogs = re.findall(dialog_pattern, scene)
            scene_dialog = " ".join(dialogs) if dialogs else ""
            
            # Extract image description if present (often after "Image:" or similar)
            image_prompt_pattern = r"(?i)(?:image|visual|picture)\s*:\s*(.*?)(?:\n|$)"
            image_prompts = re.findall(image_prompt_pattern, scene)
            
            if image_prompts:
                image_prompt = image_prompts[0].strip()
            else:
                # If no explicit image prompt, use the first 1-2 sentences as the image prompt
                sentences = scene.split('.')
                image_prompt = '. '.join(sentences[:2]) + '.'
            
            # Remove dialog and image prompts from scene description
            description = scene
            description = re.sub(dialog_pattern, "", description)
            description = re.sub(image_prompt_pattern, "", description)
            description = description.strip()
            
            scene_descriptions.append(description)
            scene_dialogs.append(scene_dialog)
            scene_image_prompts.append(image_prompt)
        
        return scene_descriptions, scene_dialogs, scene_image_prompts
    
    def _generate_story_with_gemini(self, prompt):
        """Generate a story using Gemini API"""
        self.logger.info(f"Generating story with prompt: {prompt}")
        
        try:
            # Create content for the story generation model
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]
            
            # Generate the story
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
            )
            
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating story with Gemini: {str(e)}")
            raise
    
    def _generate_image_with_gemini(self, prompt, output_path):
        """Generate an image using Gemini API"""
        self.logger.info(f"Generating image for prompt: '{prompt}'")
        
        try:
            # Configure the content and model for image generation
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["image", "text"],
                response_mime_type="text/plain",
            )
            
            # Stream the content generation
            for chunk in self.client.models.generate_content_stream(
                model=self.image_model,
                contents=contents,
                config=generate_content_config,
            ):
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    continue
                
                if chunk.candidates[0].content.parts[0].inline_data:
                    inline_data = chunk.candidates[0].content.parts[0].inline_data
                    file_extension = mimetypes.guess_extension(inline_data.mime_type) or ".jpg"
                    full_path = f"{output_path}{file_extension}"
                    self.save_binary_file(full_path, inline_data.data)
                    self.logger.info(f"Image saved to: {full_path}")
                    return full_path
                else:
                    self.logger.debug(f"Text response: {chunk.text}")
            
            self.logger.warning(f"Failed to generate image for: {prompt}")
            return None
        
        except Exception as e:
            self.logger.error(f"Error generating image with Gemini: {str(e)}")
            return None
    
    def generate(self, genre, content_type, output_dir, custom_prompt=None, visual_style="3d cartoon animation"):
        """
        Generate creative content including both story and images using Gemini
        
        Args:
            genre (str): Content genre (SciFi, Fantasy, etc.)
            content_type (str): Type of content (story, animation, etc.)
            output_dir (Path): Directory to save generated content
            custom_prompt (str, optional): Use a custom prompt instead of generating one
            visual_style (str, optional): Visual style for image generation
            
        Returns:
            dict: Generated content including script, scenes, and image paths
        """
        # Create necessary directories
        scripts_dir = output_dir / "scripts"
        images_dir = output_dir / "images"
        os.makedirs(scripts_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # Use custom prompt or generate based on genre/content type
        if custom_prompt:
            full_prompt = custom_prompt
        else:
            # Pick random topic and setting if genre is valid
            if genre in self.topics:
                import random
                topic = random.choice(self.topics[genre])
                setting = random.choice(self.settings[genre])
            else:
                topic = "an unexpected adventure"
                setting = "nothing is as it seems"
            
            # Get prompt template and fill in the placeholders
            prompt_template = self._get_prompt_template(genre, content_type, visual_style)
            full_prompt = prompt_template.format(topic=topic, setting=setting)
        
        self.logger.info(f"Generating content with Gemini prompt: {full_prompt}")
        
        # Generate the story content with Gemini
        story_text = self._generate_story_with_gemini(full_prompt)
        
        # Save the raw story text
        script_file = scripts_dir / "script.txt"
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(story_text)
        
        # Extract scenes, dialogs, and image prompts
        scene_descriptions, scene_dialogs, scene_image_prompts = self._extract_scenes_and_images(story_text)
        
        # Calculate scene timings
        total_duration = config.CONTENT["video_length"]
        scene_timings = [total_duration / len(scene_descriptions)] * len(scene_descriptions)
        
        # Generate images for each scene
        image_paths = []
        for i, image_prompt in enumerate(scene_image_prompts):
            enhanced_prompt = f"{image_prompt} {visual_style}"
            image_path = self._generate_image_with_gemini(
                enhanced_prompt, 
                str(images_dir / f"scene_{i+1}")
            )
            if image_path:
                image_paths.append(image_path)
        
        # Save scenes data
        scenes_file = scripts_dir / "scenes.json"
        scenes_data = {
            "descriptions": scene_descriptions,
            "dialogs": scene_dialogs,
            "timings": scene_timings,
            "image_prompts": scene_image_prompts,
            "image_paths": image_paths
        }
        with open(scenes_file, "w", encoding="utf-8") as f:
            json.dump(scenes_data, f, indent=2)
        
        self.logger.info(f"Generated script with {len(scene_descriptions)} scenes and {len(image_paths)} images")
        
        # Return generated content in the format expected by the pipeline
        return {
            "script": story_text,
            "script_file": str(script_file),
            "scene_descriptions": scene_descriptions,
            "scene_dialogs": scene_dialogs,
            "scene_timings": scene_timings,
            "scene_image_prompts": scene_image_prompts,
            "image_paths": image_paths,
            "base_prompt": full_prompt
        }
