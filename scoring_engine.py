"""
Versant Scoring Backend Engine
Implements audio and text scoring pipelines with task queue orchestration
"""

import json
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from queue import Queue
from threading import Lock, Thread
import logging
from pathlib import Path
import soundfile as sf
import io
import re
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QuestionResponse:
    """Represents a single question response"""
    id: str
    type: str  # 'audio' or 'text'
    content: str  # transcript or written text
    max_score: float = 10.0
    score: float = 0.0
    features: Dict[str, float] = None
    evaluation: Dict[str, Any] = None
    raw_output: Dict[str, Any] = None

    def __post_init__(self):
        if self.features is None:
            self.features = {}
        if self.evaluation is None:
            self.evaluation = {}
        if self.raw_output is None:
            self.raw_output = {}


@dataclass
class TestResult:
    """Final test scoring result"""
    total_score: float = 0.0
    max_total: float = 0.0
    questions: List[QuestionResponse] = None
    timestamp: str = None
    detailed_breakdown: Dict[str, Any] = None

    def __post_init__(self):
        if self.questions is None:
            self.questions = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.detailed_breakdown is None:
            self.detailed_breakdown = {}

    def to_dict(self):
        """Convert to JSON-serializable dict"""
        return {
            "total_score": self.total_score,
            "max_total": self.max_total,
            "questions": [
                {
                    "id": q.id,
                    "type": q.type,
                    "score": q.score,
                    "max_score": q.max_score,
                    "features": q.features,
                    "evaluation": q.evaluation,
                    "raw_output": q.raw_output,
                    "content": q.content[:100] + "..." if len(q.content) > 100 else q.content
                }
                for q in self.questions
            ],
            "timestamp": self.timestamp,
            "detailed_breakdown": self.detailed_breakdown
        }


class AudioFeatureExtractor:
    """Extract linguistic and prosodic features from audio"""

    @staticmethod
    def extract_features(
        audio_data: np.ndarray,
        transcript: str,
        sample_rate: int = 16000,
        reference_text: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Extract audio features for scoring
        
        Features include:
        - Fluency: speech rate, pause patterns, utterance duration
        - Pronunciation: based on ASR confidence proxies
        - Language use: vocabulary diversity, grammar indicators
        - Content: accuracy vs reference
        """
        features = {}

        # 1. FLUENCY FEATURES
        duration = len(audio_data) / sample_rate
        features["utterance_duration"] = float(duration)

        # Speech rate (words per minute) - estimate from transcript
        word_count = len(transcript.split())
        speech_rate = (word_count / duration * 60) if duration > 0 else 0
        features["speech_rate_wpm"] = min(float(speech_rate), 200)  # Cap at reasonable max

        # Pause analysis (silence detection)
        pause_count, avg_pause_duration = AudioFeatureExtractor._detect_pauses(
            audio_data, sample_rate
        )
        features["pause_count"] = float(pause_count)
        features["avg_pause_duration"] = float(avg_pause_duration)

        # Articulation rate (words per minute excluding pauses)
        speech_duration = duration - (pause_count * avg_pause_duration) if avg_pause_duration else duration
        articulation_rate = (word_count / speech_duration * 60) if speech_duration > 0 else 0
        features["articulation_rate"] = float(min(articulation_rate, 200))

        # 2. PRONUNCIATION & CLARITY FEATURES
        # Energy-based clarity score
        clarity_score = AudioFeatureExtractor._compute_clarity(audio_data)
        features["clarity_score"] = float(clarity_score)

        # Pitch variation (prosody indicator)
        pitch_variation = AudioFeatureExtractor._estimate_pitch_variation(audio_data, sample_rate)
        features["pitch_variation"] = float(pitch_variation)

        # 3. LANGUAGE FEATURES
        # Vocabulary diversity (type-token ratio)
        tokens = transcript.lower().split()
        unique_tokens = len(set(tokens))
        ttr = unique_tokens / len(tokens) if tokens else 0
        features["type_token_ratio"] = float(ttr)

        # Sentence complexity (average words per sentence)
        sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
        avg_words_per_sentence = len(tokens) / len(sentences) if sentences else 0
        features["avg_words_per_sentence"] = float(avg_words_per_sentence)

        # 4. CONTENT ACCURACY
        if reference_text:
            accuracy = AudioFeatureExtractor._compute_accuracy(transcript, reference_text)
            features["content_accuracy"] = float(accuracy)
        else:
            features["content_accuracy"] = 0.5  # Neutral score if no reference

        return features

    @staticmethod
    def _detect_pauses(audio_data: np.ndarray, sample_rate: int, threshold_db: float = -40) -> tuple:
        """Detect pauses/silence in audio"""
        try:
            # Compute RMS energy
            frame_length = int(sample_rate * 0.02)  # 20ms frames
            hop_length = int(sample_rate * 0.01)   # 10ms hop

            if len(audio_data) < frame_length:
                return 0, 0.0

            rms_energy = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i : i + frame_length]
                rms = np.sqrt(np.mean(frame ** 2))
                rms_energy.append(rms)

            rms_energy = np.array(rms_energy)
            rms_db = 20 * np.log10(np.maximum(rms_energy, 1e-10))

            # Detect silence frames
            silence_frames = rms_db < threshold_db
            pause_duration = frame_length / sample_rate

            # Count pause segments
            pause_count = 0
            in_pause = False
            pause_durations = []
            consecutive_silent = 0

            for is_silent in silence_frames:
                if is_silent:
                    consecutive_silent += 1
                    if not in_pause:
                        pause_count += 1
                        in_pause = True
                else:
                    if in_pause:
                        pause_durations.append(consecutive_silent * pause_duration)
                    consecutive_silent = 0
                    in_pause = False

            avg_pause_duration = np.mean(pause_durations) if pause_durations else 0.0
            return pause_count, avg_pause_duration
        except Exception as e:
            logger.warning(f"Error detecting pauses: {e}")
            return 0, 0.0

    @staticmethod
    def _compute_clarity(audio_data: np.ndarray, sample_rate: int = 16000) -> float:
        """Compute clarity score based on signal-to-noise ratio proxy"""
        try:
            # Use energy distribution as proxy for clarity
            rms = np.sqrt(np.mean(audio_data ** 2))
            db_level = 20 * np.log10(np.maximum(rms, 1e-10))
            
            # Normalize to 0-100 range (assuming -40dB to -20dB is typical speech)
            clarity = max(0, min(100, (db_level + 40) * 2.5))
            return clarity
        except Exception as e:
            logger.warning(f"Error computing clarity: {e}")
            return 50.0

    @staticmethod
    def _estimate_pitch_variation(audio_data: np.ndarray, sample_rate: int) -> float:
        """Estimate pitch variation (prosody indicator)"""
        try:
            # Simplified: use zero-crossing rate variation
            zcr = np.abs(np.diff(np.sign(audio_data))).mean()
            # Normalize to 0-1
            variation = min(1.0, zcr / (sample_rate / 1000))
            return variation * 100
        except Exception as e:
            logger.warning(f"Error computing pitch variation: {e}")
            return 50.0

    @staticmethod
    def _compute_accuracy(transcript: str, reference: str) -> float:
        """Compute content accuracy using edit distance"""
        try:
            trans_tokens = transcript.lower().split()
            ref_tokens = reference.lower().split()

            if not ref_tokens:
                return 0.0

            # Simple token-level accuracy
            matches = sum(1 for t, r in zip(trans_tokens, ref_tokens) if t == r)
            accuracy = matches / max(len(trans_tokens), len(ref_tokens))
            return accuracy
        except Exception as e:
            logger.warning(f"Error computing accuracy: {e}")
            return 0.5


class AudioScoringPipeline:
    """Audio response scoring pipeline: transcribe + extract features + score"""

    def __init__(self, riva_client=None, llm_client=None):
        """
        Initialize audio pipeline
        
        Args:
            riva_client: Riva ASR client (optional, for testing)
            llm_client: LLM client for evaluation (optional)
        """
        self.riva_client = riva_client
        self.llm_client = llm_client
        self.feature_extractor = AudioFeatureExtractor()

    async def process(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        question_id: str,
        reference_text: Optional[str] = None,
        language: str = "en-US"
    ) -> QuestionResponse:
        """
        Process audio response end-to-end
        
        Returns:
            QuestionResponse with score and features
        """
        try:
            # Step 1: Transcribe
            transcript = await self._transcribe(audio_data, sample_rate, language)
            logger.info(f"Transcribed Q{question_id}: {transcript[:50]}...")

            # Step 2: Extract features
            features = self.feature_extractor.extract_features(
                audio_data, transcript, sample_rate, reference_text
            )
            logger.info(f"Extracted {len(features)} features for Q{question_id}")

            # Step 3: Compute score
            score = self._compute_score(features, reference_text is not None)

            # Step 4: LLM evaluation (optional)
            evaluation = {}
            if self.llm_client:
                evaluation = await self._llm_evaluate(transcript, reference_text)

            response = QuestionResponse(
                id=question_id,
                type="audio",
                content=transcript,
                score=score,
                features=features,
                evaluation=evaluation,
                raw_output={"transcript": transcript}
            )

            return response

        except Exception as e:
            logger.error(f"Error processing audio for Q{question_id}: {e}")
            return QuestionResponse(
                id=question_id,
                type="audio",
                content="",
                score=0.0,
                features={},
                evaluation={"error": str(e)}
            )

    async def _transcribe(
        self, audio_data: np.ndarray, sample_rate: int, language: str
    ) -> str:
        """Transcribe audio using Riva ASR"""
        try:
            if self.riva_client:
                # Real transcription with Riva
                response = self.riva_client.offline_recognize(
                    audio_data=audio_data,
                    sample_rate_hz=sample_rate,
                    language_code=language
                )
                return response.transcripts[0].text if response.transcripts else ""
            else:
                # Fallback for testing
                logger.warning("No Riva client available; using mock transcript")
                return "mock transcript for testing"
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise

    async def _llm_evaluate(self, transcript: str, reference: Optional[str]) -> Dict:
        """Get LLM evaluation of the response"""
        try:
            if not self.llm_client:
                return {}

            prompt = f"""
Evaluate this spoken English response:

Response: "{transcript}"
{f'Expected/Reference: "{reference}"' if reference else ''}

Provide scores (0-100) for:
- Fluency (pronunciation, naturalness)
- Grammar (accuracy, complexity)
- Content (if reference provided, accuracy; otherwise relevance)

Return JSON: {{"fluency": X, "grammar": Y, "content": Z}}
"""
            response = self.llm_client.chat.completions.create(
                model="deepseek-ai/deepseek-r1-0528",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256
            )

            text = response.choices[0].message.content
            # Extract JSON
            try:
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    return json.loads(text[json_start:json_end])
            except json.JSONDecodeError:
                pass

            return {}
        except Exception as e:
            logger.error(f"LLM evaluation error: {e}")
            return {}

    def _compute_score(self, features: Dict[str, float], has_reference: bool = False) -> float:
        """Compute final score from features using weighted model"""
        try:
            # Weighted feature combination
            # Normalize features to 0-100 scale where applicable

            # Speech Rate (ideal: 120-150 wpm)
            sr = features.get("speech_rate_wpm", 0)
            sr_score = 100 * (1 - abs(sr - 130) / 130) if sr > 0 else 0
            sr_score = max(0, min(100, sr_score))

            # Clarity
            clarity = features.get("clarity_score", 50)

            # Pause patterns (fewer/shorter pauses = better)
            pause_count = features.get("pause_count", 0)
            pause_score = 100 * (1 - min(pause_count / 10, 1))

            # Vocabulary diversity (TTR)
            ttr = features.get("type_token_ratio", 0.5)
            ttr_score = ttr * 100

            # Content accuracy
            content_acc = features.get("content_accuracy", 0.5)
            content_score = content_acc * 100 if has_reference else 50

            # Weighted average
            weights = {
                "speech_rate": 0.20,
                "clarity": 0.15,
                "pauses": 0.15,
                "vocabulary": 0.20,
                "content": 0.30 if has_reference else 0.0
            }

            # Normalize weights if no reference
            if not has_reference:
                weights["speech_rate"] = 0.25
                weights["clarity"] = 0.20
                weights["pauses"] = 0.20
                weights["vocabulary"] = 0.35

            score = (
                sr_score * weights["speech_rate"]
                + clarity * weights["clarity"]
                + pause_score * weights["pauses"]
                + ttr_score * weights["vocabulary"]
                + content_score * weights["content"]
            )

            return min(100, max(0, score))

        except Exception as e:
            logger.error(f"Error computing score: {e}")
            return 50.0


class TextScoringPipeline:
    """Text response scoring pipeline: evaluate with LLM + score"""

    def __init__(self, llm_client):
        """
        Initialize text pipeline
        
        Args:
            llm_client: OpenAI-compatible LLM client (DeepSeek)
        """
        self.llm_client = llm_client

    async def process(
        self,
        text_response: str,
        question_id: str,
        question_context: Optional[str] = None,
        rubric: Optional[Dict] = None
    ) -> QuestionResponse:
        """
        Process text response end-to-end
        
        Returns:
            QuestionResponse with score and evaluation
        """
        try:
            # Build evaluation prompt with rubric
            evaluation = await self._evaluate_with_rubric(
                text_response, question_context, rubric
            )

            # Extract numeric scores
            score = self._extract_score(evaluation)

            response = QuestionResponse(
                id=question_id,
                type="text",
                content=text_response,
                score=score,
                features={},
                evaluation=evaluation,
                raw_output=evaluation
            )

            return response

        except Exception as e:
            logger.error(f"Error processing text for Q{question_id}: {e}")
            return QuestionResponse(
                id=question_id,
                type="text",
                content=text_response,
                score=0.0,
                features={},
                evaluation={"error": str(e)}
            )

    async def _evaluate_with_rubric(
        self,
        text_response: str,
        question_context: Optional[str],
        rubric: Optional[Dict]
    ) -> Dict:
        """Evaluate text using LLM with detailed rubric"""
        try:
            # Build system prompt with rubric
            system_prompt = """You are an expert English proficiency evaluator using Versant criteria.
Evaluate responses on:
- Content Accuracy: How well the response answers the question
- Grammar: Sentence structure, tense, agreement accuracy
- Vocabulary: Word choice, appropriateness, diversity
- Fluency: Natural flow, coherence, proper pacing

Respond with JSON containing: {
  "content_score": 0-100,
  "grammar_score": 0-100,
  "vocabulary_score": 0-100,
  "fluency_score": 0-100,
  "overall_score": 0-100,
  "feedback": "brief feedback",
  "strengths": ["list"],
  "areas_to_improve": ["list"]
}"""

            user_message = f"Evaluate this response:\n\n"
            if question_context:
                user_message += f"Context/Question: {question_context}\n\n"
            user_message += f"Student Response: \"{text_response}\""

            if rubric:
                user_message += f"\n\nRubric: {json.dumps(rubric)}"

            response = self.llm_client.chat.completions.create(
                model="deepseek-ai/deepseek-r1-0528",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=512
            )

            response_text = response.choices[0].message.content

            # Parse JSON response
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    return json.loads(response_text[json_start:json_end])
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM JSON response")

            return {
                "raw_response": response_text,
                "parse_error": "Could not extract JSON"
            }

        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            raise

    def _extract_score(self, evaluation: Dict) -> float:
        """Extract numeric score from evaluation dict"""
        try:
            # Try to get overall_score first
            if "overall_score" in evaluation:
                return float(evaluation["overall_score"])

            # Fallback: average component scores
            scores = [
                evaluation.get("content_score", 50),
                evaluation.get("grammar_score", 50),
                evaluation.get("vocabulary_score", 50),
                evaluation.get("fluency_score", 50)
            ]
            return sum(scores) / len(scores)

        except Exception as e:
            logger.warning(f"Error extracting score: {e}")
            return 50.0


class TaskQueue:
    """Thread-safe task queue for sequential processing"""

    def __init__(self, max_workers: int = 1):
        """
        Initialize task queue
        
        Args:
            max_workers: Number of parallel workers (1 for sequential)
        """
        self.queue = Queue()
        self.lock = Lock()
        self.results = {}
        self.max_workers = max_workers
        self.is_running = False
        self.workers = []

    def enqueue(self, task_id: str, task_func, *args, **kwargs):
        """Enqueue a task"""
        self.queue.put((task_id, task_func, args, kwargs))

    async def process_all(self) -> Dict[str, Any]:
        """Process all enqueued tasks and return results"""
        results = {}
        while not self.queue.empty():
            try:
                task_id, task_func, args, kwargs = self.queue.get()
                logger.info(f"Processing task: {task_id}")

                # Execute task
                if asyncio.iscoroutinefunction(task_func):
                    result = await task_func(*args, **kwargs)
                else:
                    result = task_func(*args, **kwargs)

                with self.lock:
                    results[task_id] = result
                    self.results[task_id] = result

                self.queue.task_done()
            except Exception as e:
                logger.error(f"Task error: {e}")
                with self.lock:
                    results[task_id] = {"error": str(e)}

        return results

    def get_result(self, task_id: str) -> Optional[Any]:
        """Get result of a specific task"""
        with self.lock:
            return self.results.get(task_id)


class ScoringEngine:
    """Main scoring orchestrator"""

    def __init__(self, riva_client=None, llm_client=None):
        """
        Initialize scoring engine
        
        Args:
            riva_client: Riva ASR client
            llm_client: LLM client for text/evaluation
        """
        self.audio_pipeline = AudioScoringPipeline(riva_client, llm_client)
        self.text_pipeline = TextScoringPipeline(llm_client)
        self.task_queue = TaskQueue(max_workers=1)  # Sequential processing
        self.results_cache = {}

    async def score_test(
        self,
        questions: List[Dict],
        session_id: Optional[str] = None
    ) -> TestResult:
        """
        Score a complete test with multiple questions
        
        Args:
            questions: List of question dicts with type, content, reference, etc.
            session_id: Optional session identifier
            
        Returns:
            TestResult with all scores and details
        """
        try:
            logger.info(f"Starting test scoring with {len(questions)} questions")

            # Process each question
            responses = []
            for q in questions:
                if q["type"] == "audio":
                    response = await self._score_audio_question(q)
                elif q["type"] == "text":
                    response = await self._score_text_question(q)
                else:
                    logger.warning(f"Unknown question type: {q['type']}")
                    continue

                responses.append(response)

            # Aggregate results
            test_result = self._aggregate_results(responses, session_id)

            # Cache result
            if session_id:
                self.results_cache[session_id] = test_result

            logger.info(f"Test scoring complete. Total: {test_result.total_score}/{test_result.max_total}")
            return test_result

        except Exception as e:
            logger.error(f"Error scoring test: {e}")
            raise

    async def _score_audio_question(self, question: Dict) -> QuestionResponse:
        """Score a single audio question"""
        try:
            audio_data = question.get("audio_data")
            sample_rate = question.get("sample_rate", 16000)
            question_id = question.get("id", "unknown")
            reference = question.get("reference_text")

            # Parse audio if provided as bytes
            if isinstance(audio_data, bytes):
                audio_data, sample_rate = sf.read(io.BytesIO(audio_data))

            response = await self.audio_pipeline.process(
                audio_data=audio_data,
                sample_rate=sample_rate,
                question_id=question_id,
                reference_text=reference
            )

            return response

        except Exception as e:
            logger.error(f"Error scoring audio question: {e}")
            return QuestionResponse(
                id=question.get("id", "unknown"),
                type="audio",
                content="",
                score=0.0,
                evaluation={"error": str(e)}
            )

    async def _score_text_question(self, question: Dict) -> QuestionResponse:
        """Score a single text question"""
        try:
            text_response = question.get("text", "")
            question_id = question.get("id", "unknown")
            context = question.get("context")
            rubric = question.get("rubric")

            response = await self.text_pipeline.process(
                text_response=text_response,
                question_id=question_id,
                question_context=context,
                rubric=rubric
            )

            return response

        except Exception as e:
            logger.error(f"Error scoring text question: {e}")
            return QuestionResponse(
                id=question.get("id", "unknown"),
                type="text",
                content=question.get("text", ""),
                score=0.0,
                evaluation={"error": str(e)}
            )

    def _aggregate_results(
        self,
        responses: List[QuestionResponse],
        session_id: Optional[str]
    ) -> TestResult:
        """Aggregate individual question scores into final result"""
        try:
            total_score = 0.0
            max_total = 0.0

            for response in responses:
                total_score += response.score
                max_total += response.max_score

            # Compute breakdown by type
            audio_responses = [r for r in responses if r.type == "audio"]
            text_responses = [r for r in responses if r.type == "text"]

            breakdown = {
                "by_type": {
                    "audio": {
                        "count": len(audio_responses),
                        "avg_score": sum(r.score for r in audio_responses) / len(audio_responses) if audio_responses else 0
                    },
                    "text": {
                        "count": len(text_responses),
                        "avg_score": sum(r.score for r in text_responses) / len(text_responses) if text_responses else 0
                    }
                },
                "average_score": total_score / len(responses) if responses else 0
            }

            result = TestResult(
                total_score=total_score,
                max_total=max_total,
                questions=responses,
                detailed_breakdown=breakdown
            )

            return result

        except Exception as e:
            logger.error(f"Error aggregating results: {e}")
            return TestResult(questions=responses)

    def get_result(self, session_id: str) -> Optional[TestResult]:
        """Retrieve cached test result"""
        return self.results_cache.get(session_id)


# Module initialization
if __name__ == "__main__":
    logger.info("Scoring engine module loaded successfully")
