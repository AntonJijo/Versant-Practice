from fastapi import FastAPI, UploadFile, File
import soundfile as sf
from riva.client import ASRService
import io
import os

app = FastAPI()

# Environment setup
RIVA_SERVER = "grpc.nvcf.nvidia.com:443"
RIVA_FUNCTION_ID = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"
RIVA_AUTH = os.getenv("RIVA_API_KEY", "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")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Read uploaded audio
    audio_bytes = await file.read()
    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="int16")

    # Connect to NVIDIA Riva
    asr_service = ASRService(
        RIVA_SERVER,
        use_ssl=True,
        metadata={
            "function-id": RIVA_FUNCTION_ID,
            "authorization": f"Bearer {RIVA_AUTH}"
        }
    )

    # Transcribe
    response = asr_service.offline_recognize(
        audio_data=audio_data,
        sample_rate_hz=sample_rate,
        language_code="en-US"
    )

    return {"transcript": response.transcripts[0].text}

