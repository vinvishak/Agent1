import os
import time
import datetime as dt

import sounddevice as sd
from scipy.io.wavfile import write
import whisper

SAMPLE_RATE = 16000        # 16kHz
CHUNK_DURATION = 10        # seconds per recording chunk
OUTPUT_DIR = "recordings"  # where we put WAV files
NOTES_FILE = "notes.txt"   # where we save all transcripts

def ensure_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def record_chunk(filename: str, duration: int = CHUNK_DURATION):
    print(f"\n[INFO] Recording {duration} seconds... Speak now.")
    audio = sd.rec(int(duration * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='int16')
    sd.wait()
    write(filename, SAMPLE_RATE, audio)
    print(f"[INFO] Saved audio to {filename}")

def append_to_notes(timestamp: str, text: str):
    line = f"[{timestamp}] {text.strip()}\n"
    with open(NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"[TRANSCRIPT] {line.strip()}")

def main():
    ensure_directories()

    print("[INFO] Loading Whisper model (first time can be slow)...")
    model = whisper.load_model("base")  # try 'tiny' if you want faster, 'small' or above for better accuracy

    print("[INFO] Voice notetaker started. Press Ctrl+C to stop.")
    try:
        while True:
            # Generate a timestamped filename
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_path = os.path.join(OUTPUT_DIR, f"chunk_{timestamp}.wav")

            # 1) Record audio to WAV
            record_chunk(wav_path, CHUNK_DURATION)

            # 2) Transcribe audio
            print("[INFO] Transcribing...")
            result = model.transcribe(wav_path, fp16=False)
            text = result.get("text", "").strip()

            if text:
                # 3) Save and print transcript
                human_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                append_to_notes(human_time, text)
            else:
                print("[INFO] No speech detected in this chunk.")

            # Optional: small pause before next chunk
            # time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user. Goodbye!")

if __name__ == "__main__":
    main()
