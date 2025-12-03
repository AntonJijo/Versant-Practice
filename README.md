
# Versant Practice Test

A FastAPI backend for the Versant English speaking assessment with audio scoring and LLM-based evaluation.

## Features

- **Audio Scoring Pipeline** - Speech recognition via NVIDIA Riva + linguistic feature extraction
- **Text Evaluation** - DeepSeek LLM scoring for accuracy, fluency, grammar, and comprehension
- **Linguistic Features** - Extracts speech rate, pauses, clarity, pitch variation, vocabulary diversity, and content accuracy
- **Multi-Question Support** - Process complete test with multiple audio/text questions
- **Session Management** - Caching and result retrieval per session
- **Comprehensive Scoring** - Weighted feature model with normalized 0-100 scale

## Requirements

- Python 3.8+
- NVIDIA Riva API (optional, falls back to mock mode)
- NVIDIA DeepSeek API key

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with API keys:
```
NVIDIA_API_KEY=your_key_here
RIVA_API_KEY=your_riva_key_here  # Optional
```

3. Start the server:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

4. Open your browser and navigate to:
```
http://localhost:8000
```

The server will serve both the API endpoints and the frontend HTML/CSS/JS files.

## Important Notes

### Puter.js Text-to-Speech
- **No API key required** - Puter.js TTS works without authentication for basic usage
- The 401 error from `api.puter.com/whoami` is normal and can be ignored - it's just Puter checking authentication status
- If Puter TTS is unavailable, the app will use fallback audio or skip audio generation

### API Endpoints
- `GET /` - Serves the main HTML page
- `POST /generate-questions` - Generate Versant questions (requires NVIDIA_API_KEY)
- `GET /health` - Health check endpoint
- All other endpoints are for scoring and evaluation

## Project Structure

```
├── app.py              # FastAPI backend with endpoints
├── scoring_engine.py   # Core scoring system
├── index.html          # Web interface
├── script.js           # Frontend logic
├── style.css           # Material Design 3 styling
├── requirements.txt    # Dependencies
└── .env                # API keys (create this)
```

## API Endpoints

- `POST /score/test` - Score complete test with multiple questions
- `POST /score/audio` - Score single audio response
- `POST /score/text` - Score single text response
- `GET /score/results/{session_id}` - Retrieve cached results
- `POST /generate-questions` - Generate questions via LLM
- `POST /evaluate-response` - Evaluate speech/text response
- `POST /transcribe` - Convert audio to text
- `GET /health` - Health check

## Audio Scoring

**Input Format:**
- 16-bit PCM WAV
- 16 kHz sample rate
- Mono channel

**Features Extracted:**
- Speech Rate (words per minute)
- Pause Detection (silence patterns)
- Clarity Score (signal quality)
- Pitch Variation (prosody/intonation)
- Vocabulary Diversity (Type-Token Ratio)
- Sentence Complexity (clause count)
- Content Accuracy (edit distance vs reference)

**Scoring Model:**
Weighted combination of normalized features (0-100):
- Speech Rate: 0.20
- Vocabulary Diversity: 0.35
- Clarity: 0.15
- Pause Pattern: 0.15
- Pitch Variation: 0.10
- Accuracy: 0.05

## Example Request

```bash
curl -X POST http://localhost:8000/score/audio \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "base64_encoded_audio",
    "sample_rate": 16000,
    "question_id": "q1",
    "reference_text": "The quick brown fox"
  }'
```

## Example Response

```json
{
  "id": "q1",
  "type": "audio",
  "transcript": "The quick brown fox",
  "score": 87.5,
  "features": {
    "speech_rate_wpm": 150,
    "pause_ratio": 0.12,
    "clarity_score": 0.89,
    "pitch_variation": 0.65,
    "vocabulary_diversity": 0.72,
    "sentence_complexity": 1.2,
    "accuracy_score": 0.95
  },
  "evaluation": "Excellent pronunciation and fluency..."
}
```

## Browser Support

| Browser | Support |
|---------|---------|
| Chrome  | ✅ Full |
| Firefox | ✅ Full |
| Safari  | ✅ Full |
| Edge    | ✅ Full |

## License

MIT License - See LICENSE file for details
