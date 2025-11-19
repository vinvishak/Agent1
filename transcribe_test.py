import whisper

def transcribe_file(path: str):
    print("Loading Whisper model (this can take a bit the first time)...")
    model = whisper.load_model("small")  # you can use "tiny", "base", "small", "medium", "large"
    result = model.transcribe(path, fp16=False)  # fp16=False to avoid GPU issues if you don't have one
    print("Transcription:")
    print(result["text"])

if __name__ == "__main__":
    transcribe_file("test.wav")
