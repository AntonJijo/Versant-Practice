"""
Versant Question Generator using DeepSeek LLM
Generates authentic Versant-style questions for all test sections
"""

import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Versant test structure constants
VERSANT_SECTIONS = {
    "reading": {
        "name": "Reading",
        "description": "Read the sentence aloud",
        "count": 8,
        "word_range": (12, 20),
        "difficulty": "intermediate"
    },
    "repeats": {
        "name": "Repeats",
        "description": "Listen and repeat the sentence exactly as you hear it",
        "count": 16,
        "word_range": (8, 18),
        "difficulty": "intermediate"
    },
    "short_answer": {
        "name": "Short Answer Questions",
        "description": "Answer the question with a single word or short phrase",
        "count": 24,
        "word_range": (5, 12),
        "difficulty": "intermediate"
    },
    "sentence_builds": {
        "name": "Sentence Builds",
        "description": "Listen to the words and form a complete sentence",
        "count": 10,
        "word_range": (8, 15),
        "difficulty": "intermediate"
    },
    "story_retelling": {
        "name": "Story Retelling",
        "description": "Listen to the story and retell it in your own words",
        "count": 3,
        "word_range": (80, 120),
        "difficulty": "advanced"
    },
    "open_questions": {
        "name": "Open Questions",
        "description": "Answer the question with your opinion",
        "count": 2,
        "word_range": (8, 15),
        "difficulty": "advanced"
    }
}


class VersantQuestionGenerator:
    """Generate authentic Versant-style questions using DeepSeek LLM"""
    
    def __init__(self, llm_client: Optional[OpenAI] = None):
        """
        Initialize question generator
        
        Args:
            llm_client: OpenAI-compatible client (DeepSeek via NVIDIA)
        """
        if llm_client is None:
            nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")
            if not nvidia_api_key:
                raise ValueError("NVIDIA_API_KEY environment variable is required")
            
            self.llm_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nvidia_api_key
            )
        else:
            self.llm_client = llm_client
        
        self.cache = {}
    
    def generate_reading_questions(self, count: int = 8) -> List[Dict]:
        """
        Generate Reading section questions (sentences to read aloud)
        
        Returns:
            List of question dicts with id, text, and audio_url
        """
        prompt = f"""Generate {count} professional English sentences for a Versant Reading test section. 

Requirements:
- Each sentence should be 12-20 words long
- Use business, professional, or academic vocabulary
- Sentences should be grammatically correct and natural
- Vary sentence structures (simple, compound, complex)
- Topics: business, technology, education, workplace, communication

Format as JSON array:
[
  {{"id": "reading-1", "text": "Complete sentence here..."}},
  {{"id": "reading-2", "text": "Another sentence..."}}
]

Return ONLY the JSON array, no additional text."""
        
        try:
            response = self._call_llm(prompt)
            questions = self._parse_json_response(response)
            
            # Validate and format
            formatted_questions = []
            for i, q in enumerate(questions[:count], 1):
                if isinstance(q, dict) and "text" in q:
                    formatted_questions.append({
                        "id": q.get("id", f"reading-{i}"),
                        "text": q["text"],
                        "type": "reading",
                        "section": "reading"
                    })
            
            logger.info(f"Generated {len(formatted_questions)} reading questions")
            return formatted_questions[:count]
            
        except Exception as e:
            logger.error(f"Error generating reading questions: {e}")
            return self._get_fallback_reading_questions(count)
    
    def generate_repeat_questions(self, count: int = 16) -> List[Dict]:
        """
        Generate Repeats section questions (sentences to repeat verbatim)
        
        Returns:
            List of question dicts with id, text, and audio_url
        """
        prompt = f"""Generate {count} English sentences for a Versant Repeats test section.

Requirements:
- Each sentence should be 8-18 words long
- Use natural, conversational English
- Include common business and everyday vocabulary
- Sentences should be clear and easy to understand when spoken
- Vary complexity: some simple, some with clauses
- Topics: workplace situations, daily activities, professional communication

Format as JSON array:
[
  {{"id": "repeat-1", "text": "Complete sentence here..."}},
  {{"id": "repeat-2", "text": "Another sentence..."}}
]

Return ONLY the JSON array, no additional text."""
        
        try:
            response = self._call_llm(prompt)
            questions = self._parse_json_response(response)
            
            formatted_questions = []
            for i, q in enumerate(questions[:count], 1):
                if isinstance(q, dict) and "text" in q:
                    formatted_questions.append({
                        "id": q.get("id", f"repeat-{i}"),
                        "text": q["text"],
                        "type": "repeats",
                        "section": "repeats"
                    })
            
            logger.info(f"Generated {len(formatted_questions)} repeat questions")
            return formatted_questions[:count]
            
        except Exception as e:
            logger.error(f"Error generating repeat questions: {e}")
            return self._get_fallback_repeat_questions(count)
    
    def generate_short_answer_questions(self, count: int = 24) -> List[Dict]:
        """
        Generate Short Answer Questions section
        
        Returns:
            List of question dicts with id, question, and expected_answer
        """
        prompt = f"""Generate {count} short answer questions for a Versant test.

Requirements:
- Questions should be answerable with 1-3 words or a short phrase
- Use common, everyday topics: work, hobbies, preferences, experiences
- Questions should be clear and direct
- Expected answers should be obvious from the question
- Mix question types: What, Where, When, Who, How, Do/Does, Are/Is

Format as JSON array:
[
  {{
    "id": "short-1",
    "question": "What time do you usually wake up?",
    "expected_answer": "7 AM" or "early morning"
  }},
  {{
    "id": "short-2",
    "question": "Where do you work?",
    "expected_answer": "office" or "at home"
  }}
]

Return ONLY the JSON array, no additional text."""
        
        try:
            response = self._call_llm(prompt)
            questions = self._parse_json_response(response)
            
            formatted_questions = []
            for i, q in enumerate(questions[:count], 1):
                if isinstance(q, dict) and "question" in q:
                    formatted_questions.append({
                        "id": q.get("id", f"short-{i}"),
                        "question": q["question"],
                        "expected_answer": q.get("expected_answer", ""),
                        "type": "short_answer",
                        "section": "short_answer"
                    })
            
            logger.info(f"Generated {len(formatted_questions)} short answer questions")
            return formatted_questions[:count]
            
        except Exception as e:
            logger.error(f"Error generating short answer questions: {e}")
            return self._get_fallback_short_answer_questions(count)
    
    def generate_sentence_builds(self, count: int = 10) -> List[Dict]:
        """
        Generate Sentence Builds section (jumbled words to form sentences)
        
        Returns:
            List of question dicts with id, words (jumbled), and correct_sentence
        """
        prompt = f"""Generate {count} sentence building exercises for a Versant test.

Requirements:
- Each should have 8-15 words when complete
- Provide the correct sentence, then list words in jumbled order
- Sentences should be grammatically correct and natural
- Use business or everyday vocabulary
- Vary sentence structures

Format as JSON array:
[
  {{
    "id": "build-1",
    "correct_sentence": "The meeting will start at three o'clock this afternoon.",
    "words": ["afternoon", "at", "meeting", "o'clock", "start", "The", "three", "this", "will"]
  }},
  {{
    "id": "build-2",
    "correct_sentence": "She completed the project before the deadline.",
    "words": ["before", "completed", "deadline", "project", "She", "the", "the"]
  }}
]

Return ONLY the JSON array, no additional text."""
        
        try:
            response = self._call_llm(prompt)
            questions = self._parse_json_response(response)
            
            formatted_questions = []
            for i, q in enumerate(questions[:count], 1):
                if isinstance(q, dict) and "correct_sentence" in q:
                    formatted_questions.append({
                        "id": q.get("id", f"build-{i}"),
                        "correct_sentence": q["correct_sentence"],
                        "words": q.get("words", q["correct_sentence"].split()),
                        "type": "sentence_builds",
                        "section": "sentence_builds"
                    })
            
            logger.info(f"Generated {len(formatted_questions)} sentence build questions")
            return formatted_questions[:count]
            
        except Exception as e:
            logger.error(f"Error generating sentence builds: {e}")
            return self._get_fallback_sentence_builds(count)
    
    def generate_story_retelling(self, count: int = 3) -> List[Dict]:
        """
        Generate Story Retelling section (stories to listen to and retell)
        
        Returns:
            List of story dicts with id, text, and audio_url
        """
        prompt = f"""Generate {count} short stories for a Versant Story Retelling test section.

Requirements:
- Each story should be 80-120 words long
- Stories should have a clear beginning, middle, and end
- Use simple, clear language suitable for intermediate English learners
- Topics: workplace situations, daily life events, problem-solving scenarios
- Stories should be interesting but not too complex
- Include specific details (names, places, times) that test-takers should remember

Format as JSON array:
[
  {{
    "id": "story-1",
    "text": "Complete story text here...",
    "title": "Brief title"
  }},
  {{
    "id": "story-2",
    "text": "Another story...",
    "title": "Brief title"
  }}
]

Return ONLY the JSON array, no additional text."""
        
        try:
            response = self._call_llm(prompt)
            stories = self._parse_json_response(response)
            
            formatted_stories = []
            for i, s in enumerate(stories[:count], 1):
                if isinstance(s, dict) and "text" in s:
                    formatted_stories.append({
                        "id": s.get("id", f"story-{i}"),
                        "text": s["text"],
                        "title": s.get("title", f"Story {i}"),
                        "type": "story_retelling",
                        "section": "story_retelling"
                    })
            
            logger.info(f"Generated {len(formatted_stories)} story retelling prompts")
            return formatted_stories[:count]
            
        except Exception as e:
            logger.error(f"Error generating story retelling: {e}")
            return self._get_fallback_story_retelling(count)
    
    def generate_open_questions(self, count: int = 2) -> List[Dict]:
        """
        Generate Open Questions section (opinion-based questions)
        
        Returns:
            List of question dicts with id and question
        """
        prompt = f"""Generate {count} open-ended opinion questions for a Versant test.

Requirements:
- Questions should require thoughtful responses (2-3 sentences minimum)
- Topics: work, lifestyle, preferences, experiences, opinions
- Questions should be clear and easy to understand
- Should encourage detailed responses, not yes/no answers

Format as JSON array:
[
  {{
    "id": "open-1",
    "question": "What do you think makes a good leader?"
  }},
  {{
    "id": "open-2",
    "question": "Describe your ideal work environment."
  }}
]

Return ONLY the JSON array, no additional text."""
        
        try:
            response = self._call_llm(prompt)
            questions = self._parse_json_response(response)
            
            formatted_questions = []
            for i, q in enumerate(questions[:count], 1):
                if isinstance(q, dict) and "question" in q:
                    formatted_questions.append({
                        "id": q.get("id", f"open-{i}"),
                        "question": q["question"],
                        "type": "open_questions",
                        "section": "open_questions"
                    })
            
            logger.info(f"Generated {len(formatted_questions)} open questions")
            return formatted_questions[:count]
            
        except Exception as e:
            logger.error(f"Error generating open questions: {e}")
            return self._get_fallback_open_questions(count)
    
    def generate_all_questions(self) -> Dict[str, List[Dict]]:
        """
        Generate questions for all Versant sections
        
        Returns:
            Dict mapping section names to question lists
        """
        logger.info("Generating questions for all Versant sections...")
        
        all_questions = {
            "reading": self.generate_reading_questions(VERSANT_SECTIONS["reading"]["count"]),
            "repeats": self.generate_repeat_questions(VERSANT_SECTIONS["repeats"]["count"]),
            "short_answer": self.generate_short_answer_questions(VERSANT_SECTIONS["short_answer"]["count"]),
            "sentence_builds": self.generate_sentence_builds(VERSANT_SECTIONS["sentence_builds"]["count"]),
            "story_retelling": self.generate_story_retelling(VERSANT_SECTIONS["story_retelling"]["count"]),
            "open_questions": self.generate_open_questions(VERSANT_SECTIONS["open_questions"]["count"])
        }
        
        total_count = sum(len(questions) for questions in all_questions.values())
        logger.info(f"Generated {total_count} total questions across all sections")
        
        return all_questions
    
    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call DeepSeek LLM with the given prompt"""
        try:
            completion = self.llm_client.chat.completions.create(
                model="deepseek-ai/deepseek-r1-0528",
                messages=[
                    {"role": "system", "content": "You are an expert English language test creator specializing in Versant-style assessments. Generate authentic, professional questions that match the Versant test format and difficulty."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                top_p=0.8,
                max_tokens=2048,
                stream=False
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _parse_json_response(self, response_text: str) -> List[Dict]:
        """Extract JSON array from LLM response"""
        try:
            # Try to find JSON array in response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Try parsing entire response
                return json.loads(response_text)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return []
    
    # Fallback questions if LLM fails
    def _get_fallback_reading_questions(self, count: int) -> List[Dict]:
        fallbacks = [
            {"id": "reading-1", "text": "The quarterly results exceeded expectations by twenty percent.", "type": "reading", "section": "reading"},
            {"id": "reading-2", "text": "Successful teams require clear communication and shared goals.", "type": "reading", "section": "reading"},
            {"id": "reading-3", "text": "Digital transformation requires both technology and cultural change.", "type": "reading", "section": "reading"},
            {"id": "reading-4", "text": "Effective leadership involves listening to team members and making informed decisions.", "type": "reading", "section": "reading"},
            {"id": "reading-5", "text": "The company announced a new initiative to improve workplace diversity.", "type": "reading", "section": "reading"},
        ]
        return fallbacks[:count]
    
    def _get_fallback_repeat_questions(self, count: int) -> List[Dict]:
        fallbacks = [
            {"id": "repeat-1", "text": "The meeting will start at three o'clock.", "type": "repeats", "section": "repeats"},
            {"id": "repeat-2", "text": "She completed the project before the deadline.", "type": "repeats", "section": "repeats"},
            {"id": "repeat-3", "text": "We need to discuss the budget for next year.", "type": "repeats", "section": "repeats"},
            {"id": "repeat-4", "text": "The presentation was very informative and well-organized.", "type": "repeats", "section": "repeats"},
            {"id": "repeat-5", "text": "Please send me the report by Friday afternoon.", "type": "repeats", "section": "repeats"},
        ]
        return fallbacks[:count]
    
    def _get_fallback_short_answer_questions(self, count: int) -> List[Dict]:
        fallbacks = [
            {"id": "short-1", "question": "What time do you usually wake up?", "expected_answer": "7 AM", "type": "short_answer", "section": "short_answer"},
            {"id": "short-2", "question": "Where do you work?", "expected_answer": "office", "type": "short_answer", "section": "short_answer"},
            {"id": "short-3", "question": "How do you get to work?", "expected_answer": "by car", "type": "short_answer", "section": "short_answer"},
            {"id": "short-4", "question": "What is your favorite hobby?", "expected_answer": "reading", "type": "short_answer", "section": "short_answer"},
        ]
        return fallbacks[:count]
    
    def _get_fallback_sentence_builds(self, count: int) -> List[Dict]:
        fallbacks = [
            {"id": "build-1", "correct_sentence": "The meeting will start at three o'clock this afternoon.", "words": ["afternoon", "at", "meeting", "o'clock", "start", "The", "three", "this", "will"], "type": "sentence_builds", "section": "sentence_builds"},
            {"id": "build-2", "correct_sentence": "She completed the project before the deadline.", "words": ["before", "completed", "deadline", "project", "She", "the", "the"], "type": "sentence_builds", "section": "sentence_builds"},
        ]
        return fallbacks[:count]
    
    def _get_fallback_story_retelling(self, count: int) -> List[Dict]:
        fallbacks = [
            {
                "id": "story-1",
                "text": "Last Monday, Sarah arrived at her office at 8:30 AM. She had an important meeting with her manager at 9:00 AM to discuss a new project. However, when she checked her email, she discovered that the meeting had been rescheduled to 2:00 PM. Sarah used the extra time to prepare her presentation and review the project details. When the meeting finally took place, she was well-prepared and the discussion went smoothly.",
                "title": "The Rescheduled Meeting",
                "type": "story_retelling",
                "section": "story_retelling"
            }
        ]
        return fallbacks[:count]
    
    def _get_fallback_open_questions(self, count: int) -> List[Dict]:
        fallbacks = [
            {"id": "open-1", "question": "What do you think makes a good leader?", "type": "open_questions", "section": "open_questions"},
            {"id": "open-2", "question": "Describe your ideal work environment.", "type": "open_questions", "section": "open_questions"},
        ]
        return fallbacks[:count]


# Convenience function for easy import
def generate_versant_questions(section: str = None, count: int = None) -> Dict:
    """
    Generate Versant questions for specified section(s)
    
    Args:
        section: Section name (reading, repeats, short_answer, etc.) or None for all
        count: Number of questions to generate (overrides default count)
    
    Returns:
        Dict of questions or list of questions for single section
    """
    generator = VersantQuestionGenerator()
    
    if section is None:
        return generator.generate_all_questions()
    
    section_map = {
        "reading": generator.generate_reading_questions,
        "repeats": generator.generate_repeat_questions,
        "short_answer": generator.generate_short_answer_questions,
        "sentence_builds": generator.generate_sentence_builds,
        "story_retelling": generator.generate_story_retelling,
        "open_questions": generator.generate_open_questions
    }
    
    if section not in section_map:
        raise ValueError(f"Unknown section: {section}. Available: {list(section_map.keys())}")
    
    func = section_map[section]
    if count:
        return func(count)
    else:
        return func(VERSANT_SECTIONS[section]["count"])

