


"""
Versant Practice Server - Main Server File
==========================================
This is the SINGLE server entry point (app.py) that coordinates all backend functionality.
It imports and uses other modules:
- Questions.py: Generates authentic Versant questions using DeepSeek LLM
- scoring_engine.py: Handles audio/text scoring and evaluation

All server endpoints are defined here. Other .py files are modules that provide
specific functionality but are not servers themselves.

To run: uvicorn app:app --reload
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import soundfile as sf
from openai import OpenAI
import io
import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import logging
from dotenv import load_dotenv

# from scoring_engine import ScoringEngine, QuestionResponse, TestResult
from Questions import VersantQuestionGenerator, generate_versant_questions

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# SERVER INITIALIZATION
# ============================================================================
# This is the main server file that coordinates all backend functionality
# It imports and uses other modules (Questions.py) for specific tasks

app = FastAPI(title="Versant Practice API", version="1.0.0")


# Enable CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    
allow_origins = [
    "https://antonjijo.github.io",  # Frontend
    "http://localhost:5500",        # Local dev (e.g., Live Server or similar)
],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
static_dir = Path(__file__).parent
if (static_dir / "index.html").exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Environment setup
RIVA_SERVER = os.getenv("RIVA_SERVER", "grpc.nvcf.nvidia.com:443")
RIVA_FUNCTION_ID = os.getenv("RIVA_FUNCTION_ID", "")
RIVA_AUTH = os.getenv("RIVA_API_KEY", "")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# Check for required API keys
if not NVIDIA_API_KEY:
    logger.warning("NVIDIA_API_KEY not found. Question generation will use fallback questions.")
    llm_client = None
else:
    # Initialize LLM client (DeepSeek via NVIDIA)
    llm_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY
    )

# Initialize question generator
try:
    question_generator = VersantQuestionGenerator(llm_client=llm_client)
    logger.info("Question generator initialized successfully")
except Exception as e:
    logger.warning(f"Question generator initialization failed: {e}. Will use fallback questions.")
    question_generator = None

# Initialize scoring engine - DISABLED
# scoring_engine = None

# Responses storage directory
RESPONSES_DIR = Path("responses")
RESPONSES_DIR.mkdir(exist_ok=True)

# Cache for generated questions
QUESTIONS_CACHE = {}
SCORING_RESULTS = {}


@app.get("/")
async def root():
    """Serve the main HTML file"""
    html_file = Path(__file__).parent / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    return {"message": "Versant Practice API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Versant Practice API",
        "scoring_engine": "disabled",
        "question_generator": "initialized" if question_generator else "failed"
    }


class EmailRequest(BaseModel):
    email: str


@app.post("/validate-email")
async def validate_email(request: EmailRequest):
    """Validate user email against allowed list"""
    try:
        allowed_emails_str = os.getenv("ALLOWED_EMAILS", "")
        if not allowed_emails_str:
            # If no emails are configured, block access
            raise HTTPException(status_code=403, detail="Access configuration missing. Contact administrator.")
            
        allowed_emails = [e.strip().lower() for e in allowed_emails_str.split(",") if e.strip()]
        
        if request.email.lower().strip() not in allowed_emails:
            raise HTTPException(status_code=403, detail="Email not authorized.")
            
        return {"status": "valid", "email": request.email}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Email validation error: {e}")
        raise HTTPException(status_code=500, detail="Validation service error")


@app.get("/generate-questions")
async def generate_questions_get(activity_type: str = "repeats", count: Optional[int] = None):
    """Generate questions via GET request (query parameters)"""
    return await _generate_questions_internal(activity_type, count)


@app.get("/generate-all-questions")
async def generate_all_questions():
    """Generate questions for all sections at once"""
    try:
        if not question_generator:
            raise ValueError("Question generator not initialized")
            
        # Check cache for full set
        cache_key = f"all_sections_{datetime.now().strftime('%Y-%m-%d')}"
        if cache_key in QUESTIONS_CACHE:
            logger.info("Returning cached questions for all sections")
            return QUESTIONS_CACHE[cache_key]
            
        all_questions = question_generator.generate_all_questions()
        
        # Transform questions to match frontend expectations
        transformed_all = {}
        
        for section, questions in all_questions.items():
            transformed_questions = []
            for q in questions:
                if section == "repeats":
                    transformed_questions.append({
                        "id": q["id"],
                        "text": q["text"],
                        "transcript": q["text"]
                    })
                elif section == "short_answer":
                    transformed_questions.append({
                        "id": q["id"],
                        "question": q["question"],
                        "exchange": q["question"]
                    })
                elif section == "sentence_builds":
                    transformed_questions.append({
                        "id": q["id"],
                        "correct": q["correct_sentence"],
                        "words": q.get("words", q["correct_sentence"].split())
                    })
                elif section == "story_retelling":
                    transformed_questions.append({
                        "id": q["id"],
                        "text": q["text"]
                    })
                else:
                    transformed_questions.append(q)
            
            transformed_all[section] = transformed_questions
            
        # Cache results
        QUESTIONS_CACHE[cache_key] = transformed_all
        return transformed_all
        
    except Exception as e:
        logger.error(f"Failed to generate all questions: {e}")
        # Return fallbacks for all sections
        return {
            "repeats": _get_fallback_questions("repeats", 16),
            "short_answer": _get_fallback_questions("short_answer", 24),
            "sentence_builds": _get_fallback_questions("sentence_builds", 10),
            "story_retelling": _get_fallback_questions("story_retelling", 3),
            "reading": _get_fallback_questions("reading", 8),
            "open_questions": _get_fallback_questions("open_questions", 2)
        }


@app.post("/generate-questions")
async def generate_questions_post(request: Request):
    """
    Generate authentic Versant questions using Questions.py module with DeepSeek LLM
    
    Accepts POST requests with JSON body containing:
    - activity_type: Section type (reading, repeats, short_answer, sentence_builds, story_retelling, open_questions)
                     Also supports legacy types: conversation, jumbled, dictation, fill, passage
    - count: Optional number of questions to generate (overrides default)
    
    Returns:
        JSON with questions array
    """
    try:
        # Try to get JSON body
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            activity_type = body.get("activity_type", "repeats")
            count = body.get("count", None)
        else:
            # Fallback to query params
            activity_type = request.query_params.get("activity_type", "repeats")
            count = request.query_params.get("count")
            if count:
                try:
                    count = int(count)
                except:
                    count = None
    except Exception as e:
        logger.warning(f"Error parsing request: {e}")
        # Fallback to query params if JSON parsing fails
        activity_type = request.query_params.get("activity_type", "repeats")
        count = request.query_params.get("count")
        if count:
            try:
                count = int(count)
            except:
                count = None
    
    return await _generate_questions_internal(activity_type, count)


async def _generate_questions_internal(activity_type: str, count: Optional[int] = None):
    """
    Internal function to generate questions
    """
    # Map legacy activity types to new Versant section names
    type_mapping = {
        "repeats": "repeats",
        "conversation": "short_answer",  # Map conversation to short answer
        "jumbled": "sentence_builds",
        "dictation": "reading",  # Map dictation to reading
        "fill": "short_answer",  # Map fill to short answer
        "passage": "story_retelling"
    }
    
    # Normalize activity type
    versant_section = type_mapping.get(activity_type, activity_type)
    
    cache_key = f"{versant_section}_{datetime.now().strftime('%Y-%m-%d')}"
    
    # Check cache
    if cache_key in QUESTIONS_CACHE:
        logger.info(f"Returning cached questions for {versant_section}")
        return {"questions": QUESTIONS_CACHE[cache_key]}
    
    try:
        if not question_generator:
            raise ValueError("Question generator not initialized")
        
        # Generate questions using Questions.py module
        if versant_section == "reading":
            questions = question_generator.generate_reading_questions(count or 8)
        elif versant_section == "repeats":
            questions = question_generator.generate_repeat_questions(count or 16)
        elif versant_section == "short_answer":
            questions = question_generator.generate_short_answer_questions(count or 24)
        elif versant_section == "sentence_builds":
            questions = question_generator.generate_sentence_builds(count or 10)
        elif versant_section == "story_retelling":
            questions = question_generator.generate_story_retelling(count or 3)
        elif versant_section == "open_questions":
            questions = question_generator.generate_open_questions(count or 2)
        else:
            raise ValueError(f"Unknown activity type: {activity_type}")
        
        # Transform questions to match frontend expectations
        transformed_questions = []
        for q in questions:
            if versant_section == "repeats":
                # Frontend expects: id, transcript, audio
                transformed_questions.append({
                    "id": q["id"],
                    "text": q["text"],
                    "transcript": q["text"]  # For compatibility
                })
            elif versant_section == "short_answer":
                # Frontend expects: id, question, exchange
                transformed_questions.append({
                    "id": q["id"],
                    "question": q["question"],
                    "exchange": q["question"]  # For compatibility
                })
            elif versant_section == "sentence_builds":
                # Frontend expects: id, correct, words
                transformed_questions.append({
                    "id": q["id"],
                    "correct": q["correct_sentence"],
                    "words": q.get("words", q["correct_sentence"].split())
                })
            elif versant_section == "story_retelling":
                # Frontend expects: id, text
                transformed_questions.append({
                    "id": q["id"],
                    "text": q["text"]
                })
            else:
                # Default: pass through
                transformed_questions.append(q)
        
        # Cache questions
        QUESTIONS_CACHE[cache_key] = transformed_questions
        logger.info(f"Generated {len(transformed_questions)} questions for {versant_section}")
        
        return {"questions": transformed_questions}
        
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        # Return fallback questions
        fallback = _get_fallback_questions(versant_section, count or 3)
        return {"questions": fallback, "warning": "Using fallback questions due to generation error"}


def _get_fallback_questions(section: str, count: int) -> List[Dict]:
    """Fallback questions if generation fails"""
    fallbacks = {
        "repeats": [
            {"id": "repeat-1", "text": "The meeting will start at three o'clock.", "transcript": "The meeting will start at three o'clock."},
            {"id": "repeat-2", "text": "She completed the project before the deadline.", "transcript": "She completed the project before the deadline."},
            {"id": "repeat-3", "text": "We need to discuss the budget for next year.", "transcript": "We need to discuss the budget for next year."},
        ],
        "short_answer": [
            {"id": "short-1", "question": "What time do you usually wake up?", "exchange": "What time do you usually wake up?"},
            {"id": "short-2", "question": "Where do you work?", "exchange": "Where do you work?"},
        ],
        "sentence_builds": [
            {"id": "build-1", "correct": "The meeting will start at three o'clock this afternoon.", "words": ["afternoon", "at", "meeting", "o'clock", "start", "The", "three", "this", "will"]},
        ],
        "story_retelling": [
            {"id": "story-1", "text": "Last Monday, Sarah arrived at her office at 8:30 AM. She had an important meeting with her manager at 9:00 AM to discuss a new project."},
        ],
        "reading": [
            {"id": "reading-1", "text": "The quarterly results exceeded expectations by twenty percent."},
            {"id": "reading-2", "text": "Successful teams require clear communication and shared goals."},
        ],
        "open_questions": [
            {"id": "open-1", "question": "What do you think makes a good leader?"},
            {"id": "open-2", "question": "Describe your ideal work environment."},
        ]
    }
    return fallbacks.get(section, [])[:count]


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
            # DISABLED - Riva client dependency removed for now
            # asr_service = ASRService(...)
            # response = asr_service.offline_recognize(...)
            # transcript = response.transcripts[0].text if response.transcripts else ""
            
            transcript = "Transcription service temporarily disabled."
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


@app.post("/score/test")
async def score_test(test_data: Dict):
    """
    Score a complete test with multiple questions
    """
    raise HTTPException(status_code=503, detail="Scoring engine disabled")


@app.post("/score/audio")
async def score_audio(
    audio_file: UploadFile = File(...),
    question_id: str = "audio-1",
    reference_text: Optional[str] = None
):
    """
    Score a single audio response
    """
    raise HTTPException(status_code=503, detail="Scoring engine disabled")


@app.post("/score/text")
async def score_text(
    response_text: str,
    question_id: str = "text-1",
    question_context: Optional[str] = None,
    rubric: Optional[Dict] = None
):
    """
    Score a single text response
    """
    raise HTTPException(status_code=503, detail="Scoring engine disabled")


@app.get("/score/results/{session_id}")
async def get_scoring_results(session_id: str):
    """Retrieve cached scoring results"""
    try:
        # Try cache first
        if session_id in SCORING_RESULTS:
            return {
                "session_id": session_id,
                "result": SCORING_RESULTS[session_id].to_dict()
            }

        # Try file
        result_file = RESPONSES_DIR / f"{session_id}_score.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                return {
                    "session_id": session_id,
                    "result": json.load(f)
                }

        raise HTTPException(status_code=404, detail="Results not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve results: {str(e)}")

