"""
Voice Generator Module
Uses Coqui TTS or Bark AI for generating realistic voiceovers
"""
import os
import logging
import json
import re
import numpy as np
import torch
from pathlib import Path
from TTS.api import TTS
import librosa
import soundfile as sf
from tqdm import tqdm
import config

class VoiceGenerator:
    """
    Generates realistic voiceovers with emotion and dynamic effects
    using Coqui TTS or Bark AI.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load voice configuration
        self.model_name = config.MODELS["voice"]["model_name"]
        self.use_bark = config.MODELS["voice"]["use_bark"]
        
        if self.use_bark:
            self.logger.info("Using Bark AI for voice generation")
        else:
            self.logger.info(f"Using Coqui TTS model: {self.model_name}")
    
    def _load_tts_model(self):
        """Load the TTS model (Coqui TTS or Bark)"""
        self.logger.info("Loading TTS model")
        
        try:
            # Set device to CUDA if available, otherwise CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {device}")
            
            if self.use_bark:
                # Import Bark specific libraries only if needed
                try:
                    from bark import SAMPLE_RATE, generate_audio, preload_models
                    
                    # Load Bark models
                    self.logger.info("Preloading Bark models")
                    preload_models()
                    
                    # Return the model (dummy object with function)
                    class BarkWrapper:
                        @staticmethod
                        def tts(text, speaker=None, **kwargs):
                            # Generate audio with Bark
                            audio_array = generate_audio(text)
                            return audio_array, SAMPLE_RATE
                    
                    return BarkWrapper(), device
                    
                except ImportError:
                    self.logger.warning("Bark not installed, falling back to Coqui TTS")
                    self.use_bark = False
            
            # If we're not using Bark, or Bark failed to load, use Coqui TTS
            if not self.use_bark:
                # Initialize TTS model
                tts = TTS(model_name=self.model_name, progress_bar=False).to(device)
                return tts, device
                
        except Exception as e:
            self.logger.error(f"Error loading TTS model: {str(e)}")
            raise
    
    def _prepare_text(self, text):
        """
        Prepare text for TTS by adding annotations for emotion and emphasis
        
        Args:
            text: Raw text for TTS
            
        Returns:
            processed_text: Text with SSML or other annotations
        """
        # Simple processing - in a real implementation this would be more sophisticated
        # with NLP to detect emotional content and add appropriate SSML tags
        
        # Add periods where they might be missing
        text = re.sub(r'([a-zA-Z])(\s+[A-Z])', r'\1.\2', text)
        
        # Add pauses at commas and periods
        text = text.replace(',', ', <break time="300ms"/>')
        text = text.replace('.', '. <break time="500ms"/>')
        text = text.replace('!', '! <break time="500ms"/>')
        text = text.replace('?', '? <break time="500ms"/>')
        
        # Add emphasis to certain phrases
        text = re.sub(r'\b(important|critical|essential|crucial|significant)\b', r'<emphasis>\1</emphasis>', text, flags=re.IGNORECASE)
        
        # Add emotion hints
        text = re.sub(r'\b(happy|joyful|excited)\b', r'<prosody rate="fast" pitch="+10%">\1</prosody>', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(sad|depressed|unhappy)\b', r'<prosody rate="slow" pitch="-10%">\1</prosody>', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(angry|furious|enraged)\b', r'<prosody rate="fast" pitch="+15%" volume="+20%">\1</prosody>', text, flags=re.IGNORECASE)
        
        # If using Bark, remove SSML tags since it may not support them
        if self.use_bark:
            text = re.sub(r'<[^>]+>', '', text)
        
        return text
    
    def _add_audio_effects(self, audio, sr, effect_type=None):
        """
        Add audio effects to enhance the voiceover
        
        Args:
            audio: Audio data as numpy array
            sr: Sample rate
            effect_type: Type of effect to apply
            
        Returns:
            processed_audio: Audio with effects applied
        """
        if effect_type is None or effect_type == "none":
            return audio
        
        try:
            if effect_type == "reverb":
                # Simulate reverb effect with a simple convolution
                reverb_length = int(sr * 0.3)  # 300ms reverb
                reverb = np.exp(-np.arange(reverb_length) / (sr * 0.07))
                audio_with_reverb = np.convolve(audio, reverb, mode='full')
                # Normalize to avoid clipping
                audio_with_reverb = audio_with_reverb / (np.max(np.abs(audio_with_reverb)) + 1e-8)
                return audio_with_reverb[:len(audio)]
            
            elif effect_type == "telephone":
                # Simulate telephone effect with bandpass filter
                from scipy import signal
                b, a = signal.butter(3, [300/(sr/2), 3400/(sr/2)], btype='band')
                return signal.lfilter(b, a, audio)
            
            elif effect_type == "radio":
                # Simulate radio effect with compression and EQ
                from scipy import signal
                # Apply mild distortion
                audio = np.tanh(audio * 1.5) / 1.5
                # Apply bandpass filter
                b, a = signal.butter(3, [500/(sr/2), 5000/(sr/2)], btype='band')
                return signal.lfilter(b, a, audio)
            
            else:
                self.logger.warning(f"Unknown effect type: {effect_type}, returning original audio")
                return audio
        
        except Exception as e:
            self.logger.error(f"Error applying audio effect: {str(e)}")
            return audio
    
    def _generate_voice_for_text(self, text, output_path, speaker=None, effect=None):
        """
        Generate voice for a piece of text
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the output audio
            speaker: Optional speaker identifier
            effect: Audio effect to apply
            
        Returns:
            output_path: Path to the generated audio file
        """
        # Prepare text with annotations
        processed_text = self._prepare_text(text)
        self.logger.debug(f"Processed text: {processed_text}")
        
        # Load TTS model
        tts_model, device = self._load_tts_model()
        
        # Generate speech
        if self.use_bark:
            # For Bark AI
            audio, sr = tts_model.tts(processed_text)
        else:
            # For Coqui TTS
            speaker = speaker if speaker else None
            audio = tts_model.tts(processed_text, speaker=speaker)
            sr = tts_model.synthesizer.output_sample_rate
        
        # Apply audio effects if requested
        if effect:
            audio = self._add_audio_effects(audio, sr, effect)
        
        # Save to file
        sf.write(str(output_path), audio, sr)
        
        self.logger.info(f"Generated audio saved to {output_path}")
        return output_path
    
    def generate(self, script, scene_dialogs, output_dir):
        """
        Generate voiceover for script and scene dialogs
        
        Args:
            script: Full script text
            scene_dialogs: List of scene dialog texts
            output_dir: Directory to save generated audio
            
        Returns:
            dict: Generated content including audio file paths
        """
        # Create audio directory
        audio_dir = output_dir / "audio"
        os.makedirs(audio_dir, exist_ok=True)
        
        self.logger.info(f"Generating voiceover for {len(scene_dialogs)} scenes")
        
        # Store audio file paths
        audio_files = []
        full_audio_path = None
        
        # Process each scene dialog
        for i, dialog in enumerate(tqdm(scene_dialogs, desc="Generating audio")):
            if not dialog.strip():
                self.logger.warning(f"Empty dialog for scene {i+1}, skipping")
                # Create a silent audio file
                silent_duration = 3.0  # 3 seconds of silence
                sr = 22050  # Standard sample rate
                silence = np.zeros(int(silent_duration * sr))
                
                # Save silent audio
                audio_path = audio_dir / f"scene_{i+1:02d}_silence.wav"
                sf.write(str(audio_path), silence, sr)
                
                audio_files.append(str(audio_path))
                continue
            
            # Determine effect based on scene content (simple heuristic)
            effect = None
            if "phone" in dialog.lower() or "call" in dialog.lower():
                effect = "telephone"
            elif "radio" in dialog.lower() or "broadcast" in dialog.lower():
                effect = "radio"
            elif "cave" in dialog.lower() or "hall" in dialog.lower() or "church" in dialog.lower():
                effect = "reverb"
            
            # Generate audio for this scene dialog
            audio_path = audio_dir / f"scene_{i+1:02d}.wav"
            self._generate_voice_for_text(
                dialog, 
                audio_path, 
                speaker=None,  # Could be varied based on characters
                effect=effect
            )
            
            audio_files.append(str(audio_path))
        
        # Optionally generate audio for the entire script as a fallback
        if script:
            full_audio_path = audio_dir / "full_script.wav"
            self._generate_voice_for_text(script, full_audio_path)
        
        self.logger.info(f"Generated {len(audio_files)} audio files")
        
        # Return generated content
        return {
            "audio_files": audio_files,
            "full_audio": str(full_audio_path) if full_audio_path else None
        }
