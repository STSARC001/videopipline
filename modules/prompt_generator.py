"""
Prompt Generator Module
Uses GPT-Neo or GPT-J for generating creative prompts and scripts
"""
import os
import logging
import json
import re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import config

class PromptGenerator:
    """
    Uses open-source LLMs (GPT-Neo or GPT-J) to generate creative prompts
    and scripts for video content creation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Download NLTK resources if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt')
        
        # Load model configuration
        self.model_name = config.MODELS["llm"]["model_name"]
        self.max_length = config.MODELS["llm"]["max_length"]
        self.temperature = config.MODELS["llm"]["temperature"]
        self.top_p = config.MODELS["llm"]["top_p"]
        
        self.logger.info(f"Initializing PromptGenerator with model: {self.model_name}")
        
        # Define genre-specific templates
        self.genre_templates = {
            "SciFi": "Create a science fiction story about {topic} in a future where {setting}.",
            "Fantasy": "Write a fantasy tale where {topic} in a world where {setting}.",
            "Thriller": "Create a suspenseful thriller involving {topic} where {setting}.",
            "Comedy": "Write a humorous story about {topic} in a situation where {setting}.",
            "Drama": "Create an emotional drama centered on {topic} where {setting}.",
            "Educational": "Explain {topic} in an engaging way, focusing on {setting}."
        }
        
        # Topics and settings for random generation
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
    
    def _load_model(self):
        """Load the language model and tokenizer"""
        self.logger.info(f"Loading model: {self.model_name}")
        
        try:
            # Set device to CUDA if available, otherwise CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {device}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            return model, tokenizer, device
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _get_prompt_template(self, genre, content_type):
        """Get a prompt template based on genre and content type"""
        base_template = self.genre_templates.get(genre, self.genre_templates["SciFi"])
        
        # Add content type-specific instructions
        if content_type == "story":
            return base_template + " Write a short story with a clear beginning, middle, and end."
        elif content_type == "animation":
            return base_template + " Write a visual script with 5-7 distinct scenes that could be animated."
        elif content_type == "news":
            return base_template + " Write this as a news report with facts, interviews, and analysis."
        elif content_type == "comedy":
            return base_template + " Make this a comedy sketch with punchlines and humorous situations."
        else:
            return base_template
    
    def _generate_variations(self, base_prompt, model, tokenizer, device, num_variations=3):
        """Generate multiple variations of content based on the base prompt"""
        variations = []
        
        for i in range(num_variations):
            variation_prompt = f"{base_prompt}\nVariation {i+1}:"
            self.logger.debug(f"Generating variation with prompt: {variation_prompt}")
            
            inputs = tokenizer(variation_prompt, return_tensors="pt").to(device)
            
            # Generate text with some randomness
            outputs = model.generate(
                inputs.input_ids,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the newly generated part (after the prompt)
            generated_part = generated_text[len(variation_prompt):].strip()
            variations.append(generated_part)
            
        return variations
    
    def _extract_scenes(self, script):
        """Extract scene descriptions and dialogs from the script"""
        # Simple scene splitting - in a real implementation, this would be more sophisticated
        scene_pattern = r"(?i)(?:scene|chapter)\s+\d+[:\.]?(.*?)(?=(?:scene|chapter)\s+\d+[:\.]?|$)"
        scenes = re.findall(scene_pattern, script, re.DOTALL)
        
        # If no scenes found using the pattern, split by sentences and group
        if not scenes:
            sentences = sent_tokenize(script)
            # Group sentences into rough scenes (5-6 sentences per scene)
            scenes = []
            for i in range(0, len(sentences), 5):
                scene = " ".join(sentences[i:i+5])
                scenes.append(scene)
        
        scene_descriptions = []
        scene_dialogs = []
        scene_timings = []
        
        # Process each scene
        total_duration = config.CONTENT["video_length"]
        scene_duration = total_duration / len(scenes)
        
        for i, scene in enumerate(scenes):
            # Extract dialog (text in quotes)
            dialog_pattern = r'"([^"]*)"'
            dialogs = re.findall(dialog_pattern, scene)
            scene_dialog = " ".join(dialogs) if dialogs else ""
            
            # Remove dialog from description
            description = re.sub(dialog_pattern, "", scene).strip()
            
            scene_descriptions.append(description)
            scene_dialogs.append(scene_dialog)
            scene_timings.append(scene_duration)
        
        return scene_descriptions, scene_dialogs, scene_timings
    
    def generate(self, genre, content_type, output_dir):
        """
        Generate creative prompts and scripts based on the specified genre and content type
        
        Args:
            genre (str): Content genre (SciFi, Fantasy, etc.)
            content_type (str): Type of content (story, animation, etc.)
            output_dir (Path): Directory to save generated content
            
        Returns:
            dict: Generated content including script, scenes, and metadata
        """
        # Create scripts directory
        scripts_dir = output_dir / "scripts"
        os.makedirs(scripts_dir, exist_ok=True)
        
        # Pick random topic and setting if genre is valid
        if genre in self.topics:
            import random
            topic = random.choice(self.topics[genre])
            setting = random.choice(self.settings[genre])
        else:
            topic = "an unexpected adventure"
            setting = "nothing is as it seems"
        
        # Get prompt template and fill in the placeholders
        prompt_template = self._get_prompt_template(genre, content_type)
        base_prompt = prompt_template.format(topic=topic, setting=setting)
        
        self.logger.info(f"Generating content with base prompt: {base_prompt}")
        
        # Load model
        model, tokenizer, device = self._load_model()
        
        # Generate variations
        variations = self._generate_variations(base_prompt, model, tokenizer, device)
        
        # Select the best variation (in this case, just take the longest one)
        selected_script = max(variations, key=len)
        
        # Extract scenes
        scene_descriptions, scene_dialogs, scene_timings = self._extract_scenes(selected_script)
        
        # Save generated content
        script_file = scripts_dir / "script.txt"
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(selected_script)
        
        # Save variations for reference
        variations_file = scripts_dir / "variations.json"
        with open(variations_file, "w", encoding="utf-8") as f:
            json.dump(variations, f, indent=2)
        
        # Save scenes
        scenes_file = scripts_dir / "scenes.json"
        scenes_data = {
            "descriptions": scene_descriptions,
            "dialogs": scene_dialogs,
            "timings": scene_timings
        }
        with open(scenes_file, "w", encoding="utf-8") as f:
            json.dump(scenes_data, f, indent=2)
        
        self.logger.info(f"Generated script with {len(scene_descriptions)} scenes")
        
        # Return generated content
        return {
            "script": selected_script,
            "script_file": str(script_file),
            "scene_descriptions": scene_descriptions,
            "scene_dialogs": scene_dialogs,
            "scene_timings": scene_timings,
            "base_prompt": base_prompt,
            "variations": variations
        }
