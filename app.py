from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
from riva.client import ASRService
from openai import OpenAI
import io
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

app = FastAPI(title="Versant Practice API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment setup
RIVA_SERVER = "grpc.nvcf.nvidia.com:443"
RIVA_FUNCTION_ID = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"
RIVA_AUTH = os.getenv("RIVA_API_KEY", "")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# Initialize LLM client
llm_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# Responses storage directory
RESPONSES_DIR = Path("responses")
RESPONSES_DIR.mkdir(exist_ok=True)

# Cache for generated questions
QUESTIONS_CACHE = {}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "Versant Practice API"}


@app.post("/generate-questions")
async def generate_questions(activity_type: str = "repeats"):
    """Generate realistic Versant questions using DeepSeek LLM"""
    cache_key = f"{activity_type}_{datetime.now().strftime('%Y-%m-%d')}"
    
    if cache_key in QUESTIONS_CACHE:
        return {"questions": QUESTIONS_CACHE[cache_key]}
    
    prompts = {
        "repeats": "Generate 3 professional English sentences for a Versant speaking test 'Read Aloud' section. Each sentence should be 12-20 words, contain advanced vocabulary, and focus on business/professional topics. Format as JSON array: [{\"id\": \"repeat-1\", \"text\": \"...\"}]",
        
        "conversation": "Generate 2 realistic business conversations with a follow-up question for each. Format as JSON: [{\"id\": \"conv-1\", \"exchange\": \"A: ...\\nB: ...\", \"question\": \"...\"}]",
        
        "jumbled": "Generate 3 complex sentences that will be presented as jumbled words for reconstruction. Format as JSON: [{\"id\": \"jumbled-1\", \"correct\": \"complete sentence here\"}]",
        
        "dictation": "Generate 3 professional sentences for dictation practice. Format as JSON: [{\"id\": \"dict-1\", \"text\": \"...\"}]",
        
        "fill": "Generate 3 sentences with blanks for filling in. Format as JSON: [{\"id\": \"fill-1\", \"sentence\": \"The company's ____ strategy...\", \"answer\": \"word\"}]",
        
        "passage": "Generate 1 comprehensive business passage (150-200 words) about professional development or organizational change. Format as JSON: {\"id\": \"passage-1\", \"text\": \"...\"}"
    }
    
    try:
        prompt = prompts.get(activity_type, prompts["repeats"])
        
        completion = llm_client.chat.completions.create(
            model="deepseek-ai/deepseek-r1-0528",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            top_p=0.7,
            max_tokens=2048,
            stream=False
        )
        
        response_text = completion.choices[0].message.content
        
        # Extract JSON from response
        start_idx = response_text.find('[') if activity_type != "passage" else response_text.find('{')
        end_idx = response_text.rfind(']') + 1 if activity_type != "passage" else response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            questions = json.loads(json_str)
            QUESTIONS_CACHE[cache_key] = questions
            return {"questions": questions}
        else:
            raise ValueError("Could not extract JSON from response")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")


@app.post("/evaluate-response")
async def evaluate_response(
    user_response: str,
    activity_type: str,
    reference_text: Optional[str] = None
):
    """Use LLM to evaluate user response for accuracy, fluency, and grammar"""
    
    try:
        evaluation_prompt = f"""
Evaluate this English response for a Versant speaking test.

Activity Type: {activity_type}
User Response: "{user_response}"
{"Reference/Correct Answer: " + reference_text if reference_text else ""}

Provide evaluation in JSON format:
{{
  "accuracy_score": 0-100,
  "fluency_score": 0-100,
  "grammar_score": 0-100,
  "pronunciation_clarity": 0-100,
  "feedback": "Brief feedback on strengths and areas to improve",
  "corrections": ["Any grammatical corrections needed"]
}}

Be strict but fair, using Versant standards.
"""
        
        completion = llm_client.chat.completions.create(
            model="deepseek-ai/deepseek-r1-0528",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.5,
            top_p=0.7,
            max_tokens=1024,
            stream=False
        )
        
        response_text = completion.choices[0].message.content
        
        # Extract JSON from response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            evaluation = json.loads(json_str)
            return evaluation
        else:
            raise ValueError("Could not extract JSON from response")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file using NVIDIA Riva service"""
    try:
        # Validate file type
        if not file.content_type or "audio" not in file.content_type:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
        
        # Read uploaded audio
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Parse audio data
        try:
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="int16")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse audio file: {str(e)}")

        # Connect to NVIDIA Riva
        try:
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

            transcript = response.transcripts[0].text if response.transcripts else ""
            return {"transcript": transcript, "success": True}
        
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Transcription service error: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/save-response")
async def save_response(data: dict):
    """Save user response for later analysis"""
    try:
        session_id = data.get("sessionId")
        if not session_id:
            raise HTTPException(status_code=400, detail="sessionId is required")
        
        response_file = RESPONSES_DIR / f"{session_id}.json"
        
        # Create response record with timestamp
        record = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Append to file or create new
        responses = []
        if response_file.exists():
            with open(response_file, 'r') as f:
                responses = json.load(f)
        
        responses.append(record)
        
        with open(response_file, 'w') as f:
            json.dump(responses, f, indent=2)
        
        return {"status": "saved", "sessionId": session_id}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save response: {str(e)}")


@app.get("/responses/{session_id}")
async def get_responses(session_id: str):
    """Retrieve saved responses for a session"""
    try:
        response_file = RESPONSES_DIR / f"{session_id}.json"
        
        if not response_file.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        
        with open(response_file, 'r') as f:
            responses = json.load(f)
        
        return {"sessionId": session_id, "responses": responses}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve responses: {str(e)}")

