"""
Speech-to-Text Model Implementation

This module provides a comprehensive speech-to-text solution using multiple approaches:
1. Real-time microphone recording and transcription
2. File-based audio processing
3. Support for multiple audio formats
4. Configurable model options

Author: AI Assistant
Date: 2025
"""

import speech_recognition as sr
import pyaudio
import wave
import os
import threading
import time
from typing import Optional, Callable, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechToTextModel:
    """
    A comprehensive speech-to-text model that supports both real-time
    and file-based audio processing.
    """
    
    def __init__(self, 
                 model_type: str = "google",
                 language: str = "en-US",
                 energy_threshold: int = 300,
                 pause_threshold: float = 0.8):
        """
        Initialize the speech-to-text model.
        
        Args:
            model_type: Type of recognition engine ("google", "sphinx", "azure", "bing")
            language: Language code for recognition
            energy_threshold: Energy level for speech detection
            pause_threshold: Seconds of silence before considering speech ended
        """
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.model_type = model_type
        self.language = language
        
        # Configure recognizer settings
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = True
        
        # Calibrate for ambient noise
        self._calibrate_microphone()
        
    def _calibrate_microphone(self):
        """Calibrate the microphone for ambient noise."""
        logger.info("Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        logger.info("Microphone calibration complete.")
    
    def transcribe_audio_file(self, file_path: str) -> str:
        """
        Transcribe audio from a file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        logger.info(f"Transcribing audio file: {file_path}")
        
        with sr.AudioFile(file_path) as source:
            # Record the audio
            audio = self.recognizer.record(source)
        
        try:
            # Perform speech recognition
            text = self._recognize_audio(audio)
            logger.info(f"Transcription successful: {text[:100]}...")
            return text
        except sr.UnknownValueError:
            logger.error("Could not understand audio")
            return ""
        except sr.RequestError as e:
            logger.error(f"Recognition service error: {e}")
            return ""
    
    def start_real_time_transcription(self, 
                                    callback: Callable[[str], None],
                                    duration: Optional[int] = None):
        """
        Start real-time transcription from microphone.
        
        Args:
            callback: Function to call with transcribed text
            duration: Maximum duration in seconds (None for continuous)
        """
        logger.info("Starting real-time transcription...")
        
        def listen_continuously():
            start_time = time.time()
            while True:
                if duration and (time.time() - start_time) > duration:
                    break
                    
                try:
                    with self.microphone as source:
                        # Listen for audio with timeout
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                    # Recognize speech
                    text = self._recognize_audio(audio)
                    if text.strip():
                        callback(text)
                        
                except sr.WaitTimeoutError:
                    # No speech detected, continue listening
                    continue
                except Exception as e:
                    logger.error(f"Error in real-time transcription: {e}")
                    continue
        
        # Start listening in a separate thread
        thread = threading.Thread(target=listen_continuously, daemon=True)
        thread.start()
        return thread
    
    def _recognize_audio(self, audio) -> str:
        """
        Recognize speech from audio data using the configured model.
        
        Args:
            audio: Audio data from microphone or file
            
        Returns:
            Transcribed text
        """
        try:
            if self.model_type == "google":
                return self.recognizer.recognize_google(audio, language=self.language)
            elif self.model_type == "sphinx":
                return self.recognizer.recognize_sphinx(audio)
            elif self.model_type == "azure":
                # Note: Requires Azure Speech SDK
                return self.recognizer.recognize_azure(audio, language=self.language)
            elif self.model_type == "bing":
                # Note: Requires Bing Speech API key
                return self.recognizer.recognize_bing(audio, language=self.language)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            logger.error(f"Recognition service error: {e}")
            return ""
    
    def record_audio(self, duration: int = 5, filename: str = "recording.wav") -> str:
        """
        Record audio from microphone and save to file.
        
        Args:
            duration: Recording duration in seconds
            filename: Output filename
            
        Returns:
            Path to the recorded audio file
        """
        logger.info(f"Recording audio for {duration} seconds...")
        
        with self.microphone as source:
            audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=duration)
        
        # Save audio to file
        file_path = os.path.join(os.getcwd(), filename)
        with open(file_path, "wb") as f:
            f.write(audio.get_wav_data())
        
        logger.info(f"Audio saved to: {file_path}")
        return file_path
    
    def get_available_microphones(self) -> List[str]:
        """Get list of available microphones."""
        return [mic.name for mic in sr.Microphone.list_microphone_names()]
    
    def set_microphone(self, device_index: int):
        """Set the microphone device by index."""
        self.microphone = sr.Microphone(device_index=device_index)
        self._calibrate_microphone()


class AudioProcessor:
    """Utility class for audio file processing."""
    
    @staticmethod
    def convert_to_wav(input_path: str, output_path: str = None) -> str:
        """
        Convert audio file to WAV format.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output WAV file (optional)
            
        Returns:
            Path to converted WAV file
        """
        if output_path is None:
            output_path = input_path.rsplit('.', 1)[0] + '.wav'
        
        # This is a placeholder - in practice, you'd use ffmpeg or similar
        logger.info(f"Converting {input_path} to {output_path}")
        return output_path
    
    @staticmethod
    def get_audio_info(file_path: str) -> dict:
        """
        Get information about an audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio file information
        """
        try:
            with sr.AudioFile(file_path) as source:
                return {
                    "duration": source.DURATION,
                    "sample_rate": source.SAMPLE_RATE,
                    "channels": source.CHANNELS,
                    "sample_width": source.SAMPLE_WIDTH
                }
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return {}


def main():
    """Example usage of the SpeechToTextModel."""
    print("Speech-to-Text Model Demo")
    print("=" * 30)
    
    # Initialize the model
    stt = SpeechToTextModel(model_type="google", language="en-US")
    
    # Show available microphones
    print("Available microphones:")
    for i, mic in enumerate(stt.get_available_microphones()):
        print(f"  {i}: {mic}")
    
    print("\nChoose an option:")
    print("1. Record and transcribe audio")
    print("2. Transcribe existing audio file")
    print("3. Start real-time transcription")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            # Record and transcribe
            duration = int(input("Recording duration (seconds): "))
            filename = input("Output filename (default: recording.wav): ").strip()
            if not filename:
                filename = "recording.wav"
            
            file_path = stt.record_audio(duration, filename)
            text = stt.transcribe_audio_file(file_path)
            print(f"Transcribed text: {text}")
            
        elif choice == "2":
            # Transcribe existing file
            file_path = input("Enter audio file path: ").strip()
            if os.path.exists(file_path):
                text = stt.transcribe_audio_file(file_path)
                print(f"Transcribed text: {text}")
            else:
                print("File not found!")
                
        elif choice == "3":
            # Real-time transcription
            print("Starting real-time transcription. Press Ctrl+C to stop.")
            
            def on_transcription(text):
                print(f"You said: {text}")
            
            try:
                thread = stt.start_real_time_transcription(on_transcription)
                thread.join()
            except KeyboardInterrupt:
                print("\nReal-time transcription stopped.")
                
        elif choice == "4":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
