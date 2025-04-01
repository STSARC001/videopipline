"""
Video Compiler Module
Uses FFmpeg and MoviePy for advanced video compilation
"""
import os
import logging
import subprocess
import json
from pathlib import Path
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip, ColorClip, VideoClip
from moviepy.video.fx import all as vfx
from tqdm import tqdm
import config

class VideoCompiler:
    """
    Compiles animations and audio into a final video with transitions and effects
    using FFmpeg and MoviePy.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load video configuration
        self.resolution = config.VIDEO["resolution"]
        self.bitrate = config.VIDEO["bitrate"]
        self.codec = config.VIDEO["codec"]
        self.audio_codec = config.VIDEO["audio_codec"]
        self.audio_bitrate = config.VIDEO["audio_bitrate"]
        
        self.logger.info(f"Initializing VideoCompiler with resolution: {self.resolution}")
    
    def _add_transition(self, clip1, clip2, transition_type="fade", duration=1.0):
        """
        Add transition between two video clips
        
        Args:
            clip1: First video clip
            clip2: Second video clip
            transition_type: Type of transition
            duration: Duration of transition in seconds
            
        Returns:
            composite_clip: Combined clip with transition
        """
        clip1_duration = clip1.duration
        
        if transition_type == "fade":
            # Crossfade between clips
            clip1 = clip1.crossfadeout(duration)
            clip2 = clip2.crossfadein(duration)
            
            # Overlap the clips
            clip1_end = clip1_duration - duration
            composite_clip = concatenate_videoclips([
                clip1.set_end(clip1_duration),
                clip2.set_start(clip1_end)
            ], method="compose")
            
            return composite_clip
            
        elif transition_type == "wipe":
            # Create a clip with mask for the second clip
            clip2_with_mask = clip2.copy()
            
            # Create a mask for the wipe
            mask_clip = ColorClip(clip2.size, col=[0, 0, 0], duration=duration)
            
            # Generate mask with animation
            def make_frame(t):
                """Create a moving wipe mask frame"""
                progress = t / duration
                frame = np.zeros((clip2.size[1], clip2.size[0]), dtype=np.uint8)
                width = clip2.size[0]
                mask_position = int(width * progress)
                frame[:, :mask_position] = 255
                return frame
            
            # Create an animated mask clip
            mask_clip = VideoClip(make_frame, duration=duration)
            mask_clip.ismask = True  # Explicitly set ismask attribute
            
            # Apply the mask
            clip2_with_mask = clip2_with_mask.set_mask(mask_clip)
            
            # Overlap the clips
            clip1_end = clip1_duration - duration
            composite_clip = CompositeVideoClip([
                clip1.set_end(clip1_duration),
                clip2_with_mask.set_start(clip1_end)
            ])
            
            return composite_clip
        
        else:
            # Default to simple concatenation
            return concatenate_videoclips([clip1, clip2])
    
    def _add_effects(self, clip, effect_type=None):
        """
        Add visual effects to a video clip
        
        Args:
            clip: Video clip to enhance
            effect_type: Type of effect to apply
            
        Returns:
            enhanced_clip: Clip with effects applied
        """
        if effect_type is None or effect_type == "none":
            return clip
        
        try:
            if effect_type == "zoom":
                # Add a slow zoom effect
                clip = clip.fx(vfx.resize, lambda t: 1 + 0.1 * t / clip.duration)
                return clip
            
            elif effect_type == "blur_in":
                # Start blurry and become clear
                def blur_factor(t):
                    """Calculate blur sigma based on time"""
                    # Linear decrease from 5 to 0 over 2 seconds
                    if t < 2:
                        return 5 * (1 - t / 2)
                    return 0
                
                # Using gblur instead of blur (which doesn't exist)
                if hasattr(vfx, 'gblur'):
                    clip = clip.fx(vfx.gblur, blur_factor)
                else:
                    # Fallback to no effect if blur isn't available
                    self.logger.warning("gblur effect not available in this version of MoviePy")
                return clip
            
            elif effect_type == "color_enhance":
                # Enhance colors
                clip = clip.fx(vfx.colorx, 1.2)  # Increase color intensity by 20%
                return clip
            
            else:
                self.logger.warning(f"Unknown effect type: {effect_type}, returning original clip")
                return clip
                
        except Exception as e:
            self.logger.error(f"Error applying video effect: {str(e)}")
            return clip
    
    def _create_thumbnail(self, video_file, output_dir):
        """
        Create a thumbnail from the video
        
        Args:
            video_file: Path to the video file
            output_dir: Directory to save the thumbnail
            
        Returns:
            thumbnail_path: Path to the generated thumbnail
        """
        self.logger.info("Creating thumbnail from video")
        
        try:
            # Open the video
            video = VideoFileClip(video_file)
            
            # Get frame from 20% into the video (usually a good spot)
            thumbnail_time = video.duration * 0.2
            thumbnail = video.get_frame(thumbnail_time)
            
            # Save the thumbnail
            from PIL import Image
            thumbnail_path = Path(output_dir) / "thumbnail.jpg"
            Image.fromarray(thumbnail).save(thumbnail_path)
            
            self.logger.info(f"Thumbnail saved to {thumbnail_path}")
            return str(thumbnail_path)
            
        except Exception as e:
            self.logger.error(f"Error creating thumbnail: {str(e)}")
            return None
    
    def _generate_metadata(self, script, output_dir):
        """
        Generate SEO-optimized metadata for the video
        
        Args:
            script: The script of the video
            output_dir: Directory to save the metadata
            
        Returns:
            metadata: Dictionary containing title, description, tags, etc.
        """
        self.logger.info("Generating video metadata")
        
        # Extract keywords using simple frequency analysis
        # In a real implementation, this would use more sophisticated NLP
        words = script.lower().split()
        # Remove common words and punctuation
        stop_words = {"the", "and", "a", "to", "of", "in", "is", "it", "that", "for", "you", "with", "on", "as", "are", "be", "this", "was", "an"}
        words = [word.strip(".,!?:;()[]{}\"'") for word in words if word not in stop_words]
        
        # Count word frequency
        from collections import Counter
        word_counts = Counter(words)
        
        # Get the most common words for tags
        common_words = [word for word, count in word_counts.most_common(10) if len(word) > 3]
        
        # Create a title from the first sentence or using keywords
        sentences = script.split('.')
        if sentences:
            title = sentences[0].strip()
            # Limit title length
            if len(title) > 60:
                title = title[:57] + "..."
        else:
            title = " ".join(common_words[:3]).title()
        
        # Create a description using the first few sentences
        description = ". ".join(sentences[:3]).strip()
        
        # Generate hashtags from common words
        hashtags = [f"#{word}" for word in common_words[:5]]
        
        # Combine into metadata
        metadata = {
            "title": title,
            "description": description,
            "tags": common_words,
            "hashtags": hashtags
        }
        
        # Save metadata to file
        metadata_path = Path(output_dir) / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved to {metadata_path}")
        return metadata
    
    def compile(self, animations, audio_files, script, output_dir):
        """
        Compile animations and audio into a final video
        
        Args:
            animations: List of animation file paths
            audio_files: List of audio file paths
            script: Full script text
            output_dir: Directory to save compiled video
            
        Returns:
            dict: Compiled content including video path and metadata
        """
        # Create videos directory
        videos_dir = output_dir / "videos"
        os.makedirs(videos_dir, exist_ok=True)
        
        self.logger.info(f"Compiling video from {len(animations)} animations and {len(audio_files)} audio files")
        
        # Final video path
        final_video_path = videos_dir / "final_video.mp4"
        
        try:
            # Load clips
            video_clips = []
            for i, anim_path in enumerate(tqdm(animations, desc="Loading animations")):
                video_clip = VideoFileClip(anim_path)
                
                # If audio file exists for this scene, add it
                if i < len(audio_files):
                    audio_clip = AudioFileClip(audio_files[i])
                    
                    # Adjust audio duration to match video if needed
                    if audio_clip.duration > video_clip.duration:
                        audio_clip = audio_clip.subclip(0, video_clip.duration)
                    elif audio_clip.duration < video_clip.duration:
                        # Loop or extend audio to match video duration
                        from moviepy.audio.fx.all import audio_loop
                        audio_clip = audio_loop(audio_clip, duration=video_clip.duration)
                    
                    # Add audio to video
                    video_clip = video_clip.set_audio(audio_clip)
                
                # Add random effects for variety
                import random
                effect_types = ["none", "zoom", "blur_in", "color_enhance"]
                effect = random.choice(effect_types)
                video_clip = self._add_effects(video_clip, effect)
                
                video_clips.append(video_clip)
            
            # Combine clips with transitions
            final_clip = None
            
            if len(video_clips) > 0:
                # Start with the first clip
                final_clip = video_clips[0]
                
                # Add subsequent clips with transitions
                for i in range(1, len(video_clips)):
                    transition_types = ["fade", "wipe"]
                    transition = transition_types[i % len(transition_types)]  # Alternate transitions
                    
                    # Combine with transition
                    combined = self._add_transition(
                        final_clip, 
                        video_clips[i], 
                        transition_type=transition,
                        duration=0.8  # Transition duration in seconds
                    )
                    
                    final_clip = combined
                
                # Resize to target resolution
                final_clip = final_clip.resize(width=self.resolution[0], height=self.resolution[1])
                
                # Write final video
                self.logger.info(f"Writing final video to {final_video_path}")
                final_clip.write_videofile(
                    str(final_video_path),
                    codec=self.codec,
                    audio_codec=self.audio_codec,
                    bitrate=self.bitrate,
                    audio_bitrate=self.audio_bitrate,
                    threads=4,
                    logger=None  # Disable moviepy's logger
                )
                
                # Clean up
                for clip in video_clips:
                    clip.close()
                if final_clip:
                    final_clip.close()
            else:
                self.logger.warning("No video clips to combine")
            
            # Generate thumbnail
            thumbnail_path = self._create_thumbnail(str(final_video_path), videos_dir)
            
            # Generate metadata
            metadata = self._generate_metadata(script, videos_dir)
            
            self.logger.info(f"Video compilation complete: {final_video_path}")
            
            # Return compiled content
            return {
                "final_video": str(final_video_path),
                "thumbnail": thumbnail_path,
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error compiling video: {str(e)}", exc_info=True)
            raise
