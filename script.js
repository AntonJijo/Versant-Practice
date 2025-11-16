/**
 * Versant Practice Application - Frontend
 * Handles UI orchestration, audio recording, LLM-based scoring integration
 */

const STORAGE_KEY = 'geminiPracticeSession';

// API Configuration - Backend server URL
// If frontend is on a different port (e.g., Live Server on 5500), set this to your backend URL
const API_BASE_URL = window.location.port === '5500' || window.location.port === '3000' 
  ? 'http://localhost:8000'  // Backend API is on port 8000
  : window.location.origin;   // Same origin if served from backend

// Log API configuration for debugging
console.log('API Base URL:', API_BASE_URL);
console.log('Frontend URL:', window.location.origin);

const audioBank = {
  repeat1: 'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQUZABp6AAD//+wbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFJwAaegAA//+sGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=',
  repeat2: 'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQU5ABp6AAD//9AbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFNwAaegAA//9AbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=',
  repeat3: 'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQUZABp6AAD//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFNwAaegAA//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=',
  story: 'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQUZABp6AAD//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFNwAaegAA//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA='
};

const sections = [
  { id: 'repeats', title: 'Read Aloud', description: 'Listen to each sentence and repeat it clearly.', timerProfile: { phases: [{ id: 'listen', label: 'Listen', duration: 5 }, { id: 'speak', label: 'Speak', duration: 5 }] } },
  { id: 'conversation', title: 'Answer Question', description: 'Listen to the conversation and answer the question.', timerProfile: { phases: [{ id: 'listen', label: 'Listen', duration: 5 }, { id: 'speak', label: 'Speak', duration: 5 }] } },
  { id: 'jumbled', title: 'Sentence Assembly', description: 'Listen to the words and form a complete sentence.', timerSeconds: 10 },
  { id: 'dictation', title: 'Dictation', description: 'Listen and type exactly what you hear.', timerSeconds: 10 },
  { id: 'fill', title: 'Fill in the Blank', description: 'Complete the sentence with the correct word.', timerSeconds: 8 },
  { id: 'passage', title: 'Retell', description: 'Read the passage, then retell it in your own words.', timerProfile: { phases: [{ id: 'read', label: 'Read', duration: 8 }, { id: 'retell', label: 'Retell', duration: 12 }] } },
];

// Default fallback questions if LLM fails
let repeatPrompts = [
  { id: 'repeat-1', transcript: 'The quarterly results exceeded expectations by twenty percent.', audio: 'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQUZABp6AAD//+wbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFJwAaegAA//+sGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=' },
  { id: 'repeat-2', transcript: 'Successful teams require clear communication and shared goals.', audio: 'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQU5ABp6AAD//9AbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFNwAaegAA//9AbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=' },
  { id: 'repeat-3', transcript: 'Digital transformation requires both technology and cultural change.', audio: 'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQUZABp6AAD//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFNwAaegAA//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=' }
];

let conversationPrompts = [
  { id: 'conv-1', audio: 'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQUZABp6AAD//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFNwAaegAA//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=', exchange: 'How do you manage stress?', question: 'How do you manage stress?' },
  { id: 'conv-2', audio: 'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQU5ABp6AAD//9AbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFNwAaegAA//9AbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=', exchange: 'What are your career goals?', question: 'What are your career goals?' },
  { id: 'conv-3', audio: 'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQUZABp6AAD//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFNwAaegAA//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=', exchange: 'Describe your ideal work environment.', question: 'Describe your ideal work environment.' }
];

let dictationPrompts = [
  { id: 'dict-1', transcript: 'Innovation drives competitive advantage in modern markets.' },
  { id: 'dict-2', transcript: 'Effective leadership requires empathy and strategic thinking.' },
  { id: 'dict-3', transcript: 'Sustainable practices benefit both business and environment.' }
];

let passagePrompt = {
  id: 'passage-1',
  text: 'The digital transformation of business operations has accelerated dramatically over the past decade. Companies across all sectors are investing heavily in cloud infrastructure, artificial intelligence, and automation technologies. This shift has profound implications for workforce development and organizational culture. While technology enables efficiency gains and cost reduction, success ultimately depends on human factors such as change management, employee training, and customer engagement. Organizations that balance technological innovation with people-centered approaches are most likely to thrive in this rapidly evolving landscape.'
};

let jumbledPrompts = [
  { id: 'jumbled-1', audio: 'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQUZABp6AAD//+wbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFJwAaegAA//+sGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=', correct: 'Strategic planning requires market analysis and resource allocation' },
  { id: 'jumbled-2', audio: 'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQU5ABp6AAD//9AbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFNwAaegAA//9AbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=', correct: 'Customer satisfaction depends on quality and responsive service' },
  { id: 'jumbled-3', audio: 'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQUZABp6AAD//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFNwAaegAA//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=', correct: 'Data analytics provides insights for informed business decisions' }
];

let fillPrompts = [
  { id: 'fill-1', sentence: 'Effective communication is ______ for team success.', answer: 'essential' },
  { id: 'fill-2', sentence: 'The company decided to ______ its operations in new markets.', answer: 'expand' },
  { id: 'fill-3', sentence: 'Project management ______ help organizations meet deadlines.', answer: 'tools' }
];



const initialState = () => ({
  sectionIndex: 0,
  itemIndex: 0,
  stepsCompleted: 0,
  responses: {
    repeats: repeatPrompts.map(p => ({ id: p.id, transcript: p.transcript, recognized: '', manual: '', attempts: 0 })),
    conversation: conversationPrompts.map(p => ({ id: p.id, recognized: '', manual: '', attempts: 0 })),
    jumbled: jumbledPrompts.map(p => ({ id: p.id, recognized: '', correct: null })),
    dictation: dictationPrompts.map(p => ({ id: p.id, transcript: p.transcript, typed: '', attempts: 0 })),
    fill: fillPrompts.map(p => ({ id: p.id, response: '', correct: null })),
    passage: { text: passagePrompt.text, typed: '', timestamp: null }
  },
  log: [],
  scored: false,
  scores: { fluency: 0, accuracy: 0, comprehension: 0, completeness: 0 }
});

let appState = loadState();
let recognitionSupported = 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window;
let activeRecognition = null;
let timerInterval = null;

const dom = {
  startScreen: document.getElementById('startScreen'),
  testContainer: document.getElementById('testContainer'),
  summaryScreen: document.getElementById('summaryScreen'),
  sectionContent: document.getElementById('sectionContent'),
  progressIndicator: document.getElementById('progressIndicator'),
  timerDisplay: document.getElementById('timerDisplay'),
  restartBtn: document.getElementById('restartBtn'),
  summaryMetrics: document.getElementById('summaryMetrics'),
  summaryLog: document.getElementById('summaryLog'),
  instructionsModal: document.getElementById('instructionsModal'),
  startInstructions: document.getElementById('startInstructions'),
  modalClose: document.getElementById('modalClose'),
  modalStartTest: document.getElementById('modalStartTest'),
  closeInstructions: document.getElementById('closeInstructions'),
  // New phase elements
  listeningPhase: document.getElementById('listeningPhase'),
  speakingPhase: document.getElementById('speakingPhase'),
  navigationButtons: document.getElementById('navigationButtons'),
  nextButton: document.getElementById('nextButton'),
  speakingLabel: document.getElementById('speakingLabel'),
  micIcon: document.getElementById('micIcon'),
  recordingTimer: document.getElementById('recordingTimer'),
  timerRing: document.querySelector('.ring-progress'),
  timerText: document.querySelector('.timer-text'),
  beepFlash: document.getElementById('beepFlash')
};

const totalSteps = sections.reduce((acc, section) => acc + getItemCountForSection(section.id), 0);

document.addEventListener('DOMContentLoaded', () => {
  bootstrap();
});

function bootstrap() {
  loadQuestionsFromLLM();
  attachGlobalEvents();
  updateUI();
}

async function loadQuestionsFromLLM() {
  console.log('Loading questions from LLM...');
  try {
    const activityTypes = ['repeats', 'conversation', 'jumbled', 'dictation', 'fill', 'passage'];
    
    for (const type of activityTypes) {
      try {
        console.log(`Fetching ${type} questions from ${API_BASE_URL}/generate-questions`);
        const response = await fetch(`${API_BASE_URL}/generate-questions`, {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({ activity_type: type })
        });
        
        if (!response.ok) {
          console.error(`Failed to fetch ${type} questions: ${response.status} ${response.statusText}`);
          console.error(`Response URL: ${response.url}`);
          // Continue with fallback questions
          continue;
        }
        
        if (response.ok) {
          const data = await response.json();
          const questions = data.questions || [];
          console.log(`Successfully loaded ${questions.length} ${type} questions`);
          
          // Map questions to global arrays
          if (type === 'repeats') {
            console.log(`Generating audio for ${questions.length} repeat questions using Puter.js...`);
            repeatPrompts = await Promise.all(questions.map(async (q, i) => {
              const text = q.text || q.transcript;
              console.log(`Generating audio for: "${text.substring(0, 50)}..."`);
              const audio = await generateTextToSpeech(text);
              if (!audio) {
                console.warn(`Failed to generate audio for question ${q.id}, using fallback`);
              }
              return {
                id: q.id,
                transcript: text,
                audio: audio || audioBank[`repeat${(i % 3) + 1}`] || audioBank.repeat1
              };
            }));
            console.log(`Completed audio generation for repeats`);
          } else if (type === 'conversation') {
            conversationPrompts = await Promise.all(questions.map(async (q) => {
              const audio = await generateTextToSpeech(q.question);
              return {
                id: q.id,
                audio: audio,
                exchange: q.exchange || q.question,
                question: q.question
              };
            }));
          } else if (type === 'jumbled') {
            jumbledPrompts = await Promise.all(questions.map(async (q) => {
              const audio = await generateTextToSpeech(q.text || q.correct);
              return {
                id: q.id,
                audio: audio,
                correct: q.correct || q.text
              };
            }));
          } else if (type === 'dictation') {
            dictationPrompts = questions.map(q => ({
              id: q.id,
              transcript: q.text || q.transcript
            }));
          } else if (type === 'fill') {
            fillPrompts = questions.map(q => ({
              id: q.id,
              sentence: q.sentence,
              type: 'text',
              answer: q.answer
            }));
          } else if (type === 'passage') {
            passagePrompt = {
              id: questions[0]?.id || 'passage-1',
              text: questions[0]?.text || questions[0] || passagePrompt.text
            };
          }
        }
      } catch (err) {
        console.error(`Failed to load ${type} questions:`, err);
        console.error('Error details:', err.message, err.stack);
      }
    }
    console.log('Question loading complete. Using:', {
      repeats: repeatPrompts.length,
      conversation: conversationPrompts.length,
      jumbled: jumbledPrompts.length,
      dictation: dictationPrompts.length,
      fill: fillPrompts.length,
      passage: passagePrompt ? 1 : 0
    });
  } catch (err) {
    console.error('Failed to load questions from LLM:', err);
    console.error('Error details:', err.message, err.stack);
  }
}

// Play audio - handles both Puter audio objects and data URLs
async function playAudio(audioSource, onEnded = null) {
  try {
    if (!audioSource) {
      console.warn('No audio source provided');
      return null;
    }
    
    let audioElement = null;
    
    // Check if it's a Puter audio object
    if (audioSource && typeof audioSource === 'object' && audioSource.play) {
      audioElement = audioSource;
      if (onEnded) {
        audioElement.addEventListener('ended', onEnded);
      }
      await audioElement.play();
    } else if (typeof audioSource === 'string') {
      // It's a data URL or URL string
      audioElement = new Audio(audioSource);
      if (onEnded) {
        audioElement.addEventListener('ended', onEnded);
      }
      await audioElement.play();
    }
    
    return audioElement;
  } catch (err) {
    console.error('Audio playback failed:', err);
    return null;
  }
}

// Generate audio from text using Puter AI Text-to-Speech
async function generateTextToSpeech(text) {
  try {
    if (!text) {
      console.warn('No text provided for TTS');
      return null;
    }
    
    // Check if Puter is available
    if (typeof puter === 'undefined') {
      console.warn('Puter.js not loaded yet, waiting...');
      // Wait a bit for Puter to load
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    if (typeof puter !== 'undefined' && puter.ai && puter.ai.txt2speech) {
      console.log('Using Puter.js TTS for:', text.substring(0, 50));
      const audio = await puter.ai.txt2speech(text, {
        voice: 'Joanna',
        engine: 'neural',
        language: 'en-US'
      });
      console.log('Puter.js TTS generated successfully');
      return audio;
    } else {
      console.warn('Puter AI not available:', {
        puter: typeof puter,
        puter_ai: typeof puter !== 'undefined' ? typeof puter.ai : 'undefined',
        txt2speech: typeof puter !== 'undefined' && puter.ai ? typeof puter.ai.txt2speech : 'undefined'
      });
      return null;
    }
  } catch (err) {
    console.error('Text-to-speech generation failed:', err);
    return null;
  }
}

function persistState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(appState));
}

function loadState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return initialState();
  try {
    return JSON.parse(raw);
  } catch (error) {
    console.warn('Failed to parse stored session. Resetting.', error);
    return initialState();
  }
}

function attachGlobalEvents() {
  document.getElementById('startTest').addEventListener('click', beginTestFlow);
  dom.startInstructions.addEventListener('click', () => toggleModal(true));
  dom.modalClose.addEventListener('click', () => toggleModal(false));
  dom.closeInstructions.addEventListener('click', () => toggleModal(false));
  dom.instructionsModal.addEventListener('click', (e) => e.target === dom.instructionsModal && toggleModal(false));
  dom.modalStartTest.addEventListener('click', () => {
    toggleModal(false);
    beginTestFlow();
  });
  dom.restartBtn.addEventListener('click', resetPractice);
  
  // Next button navigation
  if (dom.nextButton) {
    dom.nextButton.addEventListener('click', () => advanceStep(1));
  }

  // Download results button
  if (document.getElementById('downloadResultsBtn')) {
    document.getElementById('downloadResultsBtn').addEventListener('click', downloadResults);
  }
}

function toggleModal(visible) {
  dom.instructionsModal.classList.toggle('hidden', !visible);
  document.body.classList.toggle('no-scroll', visible);
}

function updateUI() {
  const isStart = appState.stepsCompleted === 0;
  const isEnd = appState.sectionIndex >= sections.length;

  dom.startScreen.classList.toggle('hidden', !isStart);
  dom.testContainer.classList.toggle('hidden', isStart || isEnd);
  dom.summaryScreen.classList.toggle('hidden', !isEnd);

  if (isEnd) {
    renderSummary();
    dom.timerDisplay.classList.add('hidden');
    if (dom.navigationButtons) {
      dom.navigationButtons.classList.add('hidden');
    }
  } else if (!isStart) {
    renderCurrentSection();
    // Update progress indicator in header
    dom.progressIndicator.textContent = `Question ${appState.stepsCompleted} of ${totalSteps}`;
    // Show navigation buttons during test
    if (dom.navigationButtons) {
      dom.navigationButtons.classList.remove('hidden');
    }
  }
}

function beginTestFlow() {
  appState.stepsCompleted = 1;
  updateUI();
}

function advanceStep(direction = 1) {
  stopRecognition();
  stopTimer();
  hideAllPhases();

  if (direction === -1) {
    // Previous button disabled - no backward navigation allowed
    console.log('Previous navigation disabled');
    return;
  }
  
  // Go forward (next button or auto-advance on completion)
  const currentSection = sections[appState.sectionIndex];
  if (!currentSection) {
    console.error('No section found at index:', appState.sectionIndex);
    return;
  }
  
  const itemsInSection = getItemCountForSection(currentSection.id);
  if (appState.itemIndex < itemsInSection - 1) {
    appState.itemIndex++;
  } else {
    appState.sectionIndex++;
    appState.itemIndex = 0;
  }
  appState.stepsCompleted++;
  
  persistState();
  updateUI();
}

function showPhase(phaseId) {
  const phases = [dom.listeningPhase, dom.speakingPhase];
  phases.forEach(p => p.classList.add('hidden'));
  
  if (phaseId === 'listening') {
    dom.listeningPhase.classList.remove('hidden');
  } else if (phaseId === 'speaking') {
    dom.speakingPhase.classList.remove('hidden');
  }
}

function hideAllPhases() {
  dom.listeningPhase.classList.add('hidden');
  dom.speakingPhase.classList.add('hidden');
}

function updateMicIcon(isActive) {
  dom.micIcon.classList.toggle('active', isActive);
  dom.micIcon.classList.toggle('inactive', !isActive);
}

function triggerBeepFlash() {
  dom.beepFlash.classList.remove('hidden');
  setTimeout(() => {
    dom.beepFlash.classList.add('hidden');
  }, 500);
}

function updateRecordingTimer(elapsedSeconds, totalSeconds) {
  const ratio = Math.min(elapsedSeconds / totalSeconds, 1);
  const circumference = 2 * Math.PI * 45;
  const offset = circumference * (1 - ratio);
  
  if (dom.timerRing) {
    dom.timerRing.style.strokeDashoffset = offset;
  }
  
  if (dom.timerText) {
    dom.timerText.textContent = Math.ceil(totalSeconds - elapsedSeconds) + 's';
  }
}

function getItemCountForSection(sectionId) {
  switch (sectionId) {
    case 'repeats': return repeatPrompts.length;
    case 'conversation': return conversationPrompts.length;
    case 'jumbled': return jumbledPrompts.length;
    case 'dictation': return dictationPrompts.length;
    case 'fill': return fillPrompts.length;
    case 'passage':
      return 1;
    default: return 1;
  }
}

function createSectionHeader(title, description) {
  const header = document.createElement('div');
  header.className = 'section-header';
  header.innerHTML = `
    <h2>${title}</h2>
    <p>${description}</p>
  `;
  return header;
}

function renderCurrentSection() {
  const section = sections[appState.sectionIndex];
  if (!section) {
    console.warn('No section at index', appState.sectionIndex);
    return;
  }

  // Don't try to render if arrays are empty or undefined
  const itemsInSection = getItemCountForSection(section.id);
  if (itemsInSection === 0) {
    console.warn('No items in section:', section.id);
    dom.sectionContent.innerHTML = '<div style="text-align: center; color: var(--on-surface-variant-color); padding: 2rem;"><p>Loading activity...</p></div>';
    return;
  }

  let renderResult;
  try {
    switch (section.id) {
      case 'repeats': renderResult = renderRepeatsSection(appState.itemIndex); break;
      case 'conversation': renderResult = renderConversationSection(appState.itemIndex); break;
      case 'jumbled': renderResult = renderJumbledSection(appState.itemIndex); break;
      case 'dictation': renderResult = renderDictationSection(appState.itemIndex); break;
      case 'fill': renderResult = renderFillSection(appState.itemIndex); break;
      case 'passage': renderResult = renderPassageSection(); break;
      default:
        const content = document.createElement('div');
        content.className = 'section-header';
        content.innerHTML = `<h2>${section.title}</h2><p>${section.description}</p>`;
        renderResult = { node: content };
    }
  } catch (error) {
    console.error('Error rendering section:', error);
    dom.sectionContent.innerHTML = `<div style="color: var(--error-color); padding: 2rem;">Error loading activity</div>`;
    return;
  }

  dom.sectionContent.innerHTML = '';
  if (renderResult && renderResult.node) {
    dom.sectionContent.appendChild(renderResult.node);
  }

  // Start timer with hooks if available
  const timerConfig = section.timerProfile || (section.timerSeconds ? { phases: [{ id: 'listen', duration: section.timerSeconds }] } : null);
  if (timerConfig && renderResult && renderResult.hooks) {
    startTimer(timerConfig, renderResult.hooks);
  } else if (timerConfig) {
    startTimer(timerConfig);
  } else {
    stopTimer();
  }
}

function createAudioVisualization() {
  const visualization = document.createElement('div');
  visualization.className = 'audio-visualization';
  for (let i = 0; i < 20; i++) {
    const bar = document.createElement('div');
    bar.className = 'audio-bar';
    visualization.appendChild(bar);
  }
  return visualization;
}

function animateAudioBars(visualization, isPlaying) {
  const bars = visualization.querySelectorAll('.audio-bar');
  if (isPlaying) {
    bars.forEach((bar, i) => {
      bar.style.animation = `waveform ${Math.random() * 0.5 + 0.5}s ease-in-out infinite alternate ${i * 0.05}s`;
    });
  } else {
    bars.forEach(bar => {
      bar.style.animation = 'none';
      bar.style.height = '4px';
    });
  }
}

function renderMicrophoneIcon() {
  const micContainer = document.createElement('div');
  micContainer.className = 'microphone-icon-container';
  micContainer.innerHTML = `
    <lord-icon
      src="https://cdn.lordicon.com/tiawvGsq.json"
      trigger="loop"
      colors="primary:#ffffff,secondary:#89b1f7"
      style="width:100px;height:100px">
    </lord-icon>
    <p class="status-text">Recording... Speak now</p>
  `;
  return micContainer;
}

function renderRepeatsSection(index) {
  const section = sections[appState.sectionIndex];
  const prompt = repeatPrompts[index];
  const response = appState.responses.repeats[index];
  const wrapper = document.createElement('div');
  wrapper.style.display = 'flex';
  wrapper.style.flexDirection = 'column';
  wrapper.style.gap = '1.5rem';
  wrapper.style.alignItems = 'center';

  const header = createSectionHeader(section.title, section.description);
  wrapper.appendChild(header);

  const repeatItem = document.createElement('div');
  repeatItem.className = 'repeat-item';
  repeatItem.innerHTML = `<div class="repeat-text">${prompt.transcript}</div>`;
  wrapper.appendChild(repeatItem);

  const audioSource = prompt.audio || audioBank.repeat1;

  const hooks = {
    onPhaseStart: (phase) => {
      if (phase.id === 'listen') {
        showPhase('listening');
        updateMicIcon(false);
        playAudio(audioSource);
      } else if (phase.id === 'speak') {
        showPhase('speaking');
        triggerBeepFlash();
        dom.speakingLabel.textContent = 'Recording... Repeat now';
        updateMicIcon(true);
        
        startRecognition({
          onResult: (text) => {
            response.recognized = text;
            persistState();
            appState.log.push({
              section: 'Read Aloud',
              promptId: prompt.id,
              text: text,
              timestamp: new Date().toISOString()
            });
            
            evaluateResponse(text, 'repeats', prompt.transcript).then(evaluation => {
              const logEntry = appState.log.find(e => e.promptId === prompt.id);
              if (logEntry) {
                logEntry.evaluation = evaluation;
                persistState();
              }
            });
            
            updateMicIcon(false);
            setTimeout(() => advanceStep(1), 1000);
          },
          onEnd: () => {
            updateMicIcon(false);
            if (!response.recognized) {
              setTimeout(() => advanceStep(1), 1000);
            }
          }
        });
      }
    },
    onTimerComplete: () => {
      hideAllPhases();
      stopRecognition();
      advanceStep(1);
    }
  };

  return { node: wrapper, hooks, timerProfile: section.timerProfile };
}

function renderConversationSection(index) {
  const section = sections[appState.sectionIndex];
  const prompt = conversationPrompts[index];
  const response = appState.responses.conversation[index];
  const wrapper = document.createElement('div');

  const header = createSectionHeader(section.title, section.description);
  wrapper.appendChild(header);

  const conversationSection = document.createElement('div');
  conversationSection.className = 'conversation-section';
  
  // Audio visualization
  const audioContainer = document.createElement('div');
  audioContainer.className = 'p-4 mb-4 gemini-border';
  const audioVisualization = createAudioVisualization();
  audioContainer.appendChild(audioVisualization);

  const statusText = document.createElement('p');
  statusText.className = 'status-text';
  statusText.textContent = 'Listen to the question.';
  audioContainer.appendChild(statusText);

  conversationSection.appendChild(audioContainer);

  // Show the question text (reference)
  const questionBubble = document.createElement('div');
  questionBubble.className = 'chat-bubble question';
  questionBubble.innerHTML = `<strong>Question:</strong> ${prompt.question}`;
  conversationSection.appendChild(questionBubble);

  // Optional tip
  const tipBubble = document.createElement('div');
  tipBubble.className = 'chat-bubble tip';
  tipBubble.innerHTML = '<em>A short, clear answer is best</em>';
  conversationSection.appendChild(tipBubble);

  wrapper.appendChild(conversationSection);

  const audioSource = prompt.audio || audioBank.repeat1;
  const micContainer = renderMicrophoneIcon();
  micContainer.classList.add('hidden');
  wrapper.appendChild(micContainer);

  let currentAudio = null;
  
  const hooks = {
    onPhaseStart: (phase) => {
      if (phase.id === 'listen') {
        showPhase('listening');
        updateMicIcon(false);
        statusText.textContent = 'Listen to the question.';
        animateAudioBars(audioVisualization, true);
        playAudio(audioSource, () => {
          animateAudioBars(audioVisualization, false);
          statusText.textContent = 'Question complete. Ready to answer.';
        }).then(audio => {
          currentAudio = audio;
        });
      } else if (phase.id === 'speak') {
        showPhase('speaking');
        triggerBeepFlash();
        dom.speakingLabel.textContent = 'Recording... Answer the question';
        updateMicIcon(true);
        statusText.textContent = 'Now speak your answer.';
        
        startRecognition({
          onResult: (text) => {
            response.recognized = text;
            appState.log.push({
              section: 'Answer Question',
              promptId: prompt.id,
              text: text,
              timestamp: new Date().toISOString()
            });
            persistState();
            
            evaluateResponse(text, 'conversation').then(evaluation => {
              const logEntry = appState.log.find(e => e.promptId === prompt.id);
              if (logEntry) {
                logEntry.evaluation = evaluation;
                persistState();
              }
            });
            
            updateMicIcon(false);
            setTimeout(() => advanceStep(1), 1000);
          },
          onEnd: () => {
            updateMicIcon(false);
            if (!response.recognized) {
              setTimeout(() => advanceStep(1), 1000);
            }
          }
        });
      }
    },
    onTimerComplete: () => {
      hideAllPhases();
      stopRecognition();
      advanceStep(1);
    }
  };

  return { node: wrapper, hooks, timerProfile: section.timerProfile };
}

function renderJumbledSection(index) {
  const section = sections[appState.sectionIndex];
  const prompt = jumbledPrompts[index];
  const response = appState.responses.jumbled[index];

  const wrapper = document.createElement('div');
  wrapper.className = 'space-y-4';

  const header = createSectionHeader(section.title, section.description);
  wrapper.appendChild(header);

  const audioContainer = document.createElement('div');
  audioContainer.className = 'p-4 mb-4 gemini-border';
  const audioVisualization = createAudioVisualization();
  audioContainer.appendChild(audioVisualization);

  const statusText = document.createElement('p');
  statusText.className = 'status-text';
  statusText.textContent = 'Listen to the jumbled sentence.';
  audioContainer.appendChild(statusText);

  const audioSource = prompt.audio || audioBank.repeat1;

  const micContainer = renderMicrophoneIcon();
  micContainer.classList.add('hidden'); 

  wrapper.appendChild(audioContainer);
  wrapper.appendChild(micContainer);

  const hooks = {
    onPhaseStart: (phase) => {
      if (phase.id === 'listen' || !phase.id) {
        statusText.textContent = 'Listen to the jumbled sentence.';
        animateAudioBars(audioVisualization, true);
        playAudio(audioSource, () => {
          animateAudioBars(audioVisualization, false);
          statusText.textContent = 'Now, form a sentence by speaking.';
          micContainer.classList.remove('hidden');
          startRecognition({
      onResult: (text) => {
        const isCorrect = text.trim().toLowerCase() === prompt.correct.trim().toLowerCase();
        response.recognized = text;
        response.correct = isCorrect;
        persistState();
        appState.log.push({
          section: 'Sentence Assembly',
          promptId: prompt.id,
          text: text,
          correct: isCorrect,
          timestamp: new Date().toISOString()
        });
        
        // Evaluate response asynchronously
        evaluateResponse(text, 'jumbled', prompt.correct).then(evaluation => {
          const logEntry = appState.log.find(e => e.promptId === prompt.id);
          if (logEntry) {
            logEntry.evaluation = evaluation;
            persistState();
          }
        });
        
        micContainer.querySelector('.status-text').textContent = `Captured: "${text}"`;
        setTimeout(() => advanceStep(1), 2000); 
      },
      onEnd: () => {
        if (!response.recognized) {
          micContainer.querySelector('.status-text').textContent = 'No speech detected.';
          setTimeout(() => advanceStep(1), 2000);
        }
      }
    });
        }).then(audio => {
          // Audio playback started
        });
      }
    },
    onTimerComplete: () => {
      stopRecognition();
      advanceStep(1);
    }
  };

  return { node: wrapper, hooks };
}



function renderDictationSection(index) {
  const section = sections[appState.sectionIndex];
  const prompt = dictationPrompts[index];
  const response = appState.responses.dictation[index];
  const wrapper = document.createElement('div');
  wrapper.className = 'space-y-4';

  const header = createSectionHeader(section.title, section.description);
  wrapper.appendChild(header);

  const audioContainer = document.createElement('div');
  audioContainer.className = 'p-4 mb-4 gemini-border';
  const audioVisualization = createAudioVisualization();
  audioContainer.appendChild(audioVisualization);

  const statusText = document.createElement('p');
  statusText.className = 'status-text';
  statusText.textContent = 'Listen to the sentence.';
  audioContainer.appendChild(statusText);

  const audioSource = audioBank.repeat1; // Placeholder audio for dictation

  const hooks = {
    onPhaseStart: (phase) => {
      if (phase.id === 'default' || !phase.id) { // Assuming a single phase for dictation
        statusText.textContent = 'Listening to sentence...';
        animateAudioBars(audioVisualization, true);
        playAudio(audioSource, () => {
          animateAudioBars(audioVisualization, false);
          audioContainer.innerHTML = ''; // Clear audio visualization
          
          const keyboardIcon = document.createElement('div');
          keyboardIcon.className = 'keyboard-icon';
          keyboardIcon.innerHTML = '⌨️';
          wrapper.appendChild(keyboardIcon);

          const textBox = document.createElement('textarea');
          textBox.className = 'response-area';
          textBox.placeholder = 'Type exactly what you heard...';
          textBox.value = response.typed;
          wrapper.appendChild(textBox);
          textBox.focus();

          let timeLeft = sections[appState.sectionIndex].timerSeconds;
          const timerDisplay = document.createElement('p');
          timerDisplay.textContent = `Time left: ${timeLeft}s`;
          wrapper.appendChild(timerDisplay);

          const countdown = setInterval(() => {
            timeLeft--;
            timerDisplay.textContent = `Time left: ${timeLeft}s`;
            if (timeLeft <= 0) {
              clearInterval(countdown);
              response.typed = textBox.value;
              persistState();
              advanceStep(1); // Automatic advancement
            }
          }, 1000);

          textBox.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              clearInterval(countdown);
              response.typed = textBox.value;
              persistState();
              advanceStep(1); // Automatic advancement
            }
          });
          textBox.addEventListener('input', () => {
            response.typed = textBox.value;
            persistState();
          });
        });
      }
    },
    onTimerComplete: () => {
      // Timer for dictation is handled internally after audio ends
    }
  };

  wrapper.appendChild(audioContainer);
  return { node: wrapper, hooks, timerProfile: sections[appState.sectionIndex].timerProfile };
}

function renderFillSection(index) {
  const section = sections[appState.sectionIndex];
  const prompt = fillPrompts[index];
  const response = appState.responses.fill[index];

  const wrapper = document.createElement('div');
  wrapper.className = 'space-y-4 gemini-card';

  const header = createSectionHeader(section.title, section.description);
  wrapper.appendChild(header);

  const sentenceEl = document.createElement('p');
  sentenceEl.className = 'text-base';
  
  const inputEl = document.createElement('span');
  inputEl.className = 'inline-blank-input';
  inputEl.contentEditable = 'true';
  inputEl.textContent = response.response;

  const sentenceParts = prompt.sentence.split('____');
  sentenceEl.textContent = sentenceParts[0];
  sentenceEl.appendChild(inputEl);
  sentenceEl.append(sentenceParts[1]);
  
  wrapper.appendChild(sentenceEl);

  const feedback = document.createElement('p');
  feedback.className = 'text-sm';
  wrapper.appendChild(feedback);

  inputEl.addEventListener('blur', () => {
    evaluateFillResponse(inputEl.textContent);
    advanceStep(1);
  });

  inputEl.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      inputEl.blur();
    }
  });

  function evaluateFillResponse(value) {
    const normalized = value.trim().toLowerCase();
    const isCorrect = normalized === prompt.answer.trim().toLowerCase();
    response.response = value;
    response.correct = isCorrect;
    feedback.textContent = isCorrect ? '✅ Correct!' : '⚠️ Incorrect.';
    appState.log.push({
      section: 'Fill in the Blank',
      promptId: prompt.id,
      text: value,
      correct: isCorrect,
      timestamp: new Date().toISOString()
    });
    persistState();
  }

  const hooks = {
    onTimerComplete: () => {
      if (response.response === '') { // If no response, mark as incorrect
        evaluateFillResponse('');
      }
      advanceStep(1); // Automatic advancement
    }
  };

  return { node: wrapper, hooks, timerProfile: sections[appState.sectionIndex].timerProfile };
}

function renderPassageSection() {
  const section = sections[appState.sectionIndex];
  const response = appState.responses.passage;
  const wrapper = document.createElement('div');
  wrapper.className = 'space-y-4';

  const header = createSectionHeader(section.title, section.description);
  wrapper.appendChild(header);

  const passageContainer = document.createElement('div');
  passageContainer.className = 'p-6 mb-4 gemini-border';
  wrapper.appendChild(passageContainer);

  const passageText = document.createElement('p');
  passageText.className = 'text-lg leading-relaxed';
  passageText.textContent = passagePrompt.text;

  const timerDisplay = document.createElement('div');
  timerDisplay.className = 'passage-timer';
  const timerBar = document.createElement('div');
  timerBar.className = 'passage-timer-bar';
  timerDisplay.appendChild(timerBar);

  passageContainer.appendChild(passageText);
  passageContainer.appendChild(timerDisplay);

  const hooks = {
    onPhaseStart: (phase) => {
      if (phase.id === 'view') {
        passageContainer.classList.remove('hidden');
        timerBar.style.transition = `width ${phase.duration}s linear`;
        setTimeout(() => {
          timerBar.style.width = '100%';
        }, 50);
      }
      if (phase.id === 'write') {
        passageContainer.classList.add('hidden');
        const textBox = document.createElement('textarea');
        textBox.className = 'response-area passage-textarea';
        textBox.placeholder = 'Type what you remember from the passage...';
        textBox.value = response.typed;
        wrapper.appendChild(textBox);
        textBox.focus();

        textBox.addEventListener('input', () => {
          response.typed = textBox.value;
          persistState();
        });
      }
    },
    onTimerComplete: () => {
      response.timestamp = new Date().toISOString();
      persistState();
      advanceStep(1); // Automatic advancement
    }
  };

  return { node: wrapper, hooks, timerProfile: sections[appState.sectionIndex].timerProfile };
}



function startRecognition({ onResult, onEnd, onError }) {
  if (!recognitionSupported) return;
  stopRecognition();

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  activeRecognition = new SpeechRecognition();
  activeRecognition.lang = 'en-US';
  activeRecognition.interimResults = true;

  activeRecognition.onresult = (event) => {
    let transcript = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      if (event.results[i].isFinal) {
        transcript += event.results[i][0].transcript;
      }
    }
    if (transcript) onResult(transcript.trim());
  };

  activeRecognition.onend = () => {
    activeRecognition = null;
    onEnd?.();
  };
  activeRecognition.onerror = (event) => {
    console.error('Speech recognition error', event.error);
    onError?.(event.error);
  };

  activeRecognition.start();
}

function stopRecognition() {
  if (activeRecognition) {
    activeRecognition.stop();
    activeRecognition = null;
  }
}

function startTimer(config, hooks = {}) {
  stopTimer();
  dom.timerDisplay.classList.remove('hidden');

  let currentPhaseIndex = -1;
  const phases = config.phases;

  const advancePhase = () => {
    currentPhaseIndex++;
    if (currentPhaseIndex >= phases.length) {
      stopTimer(true);
      hooks.onTimerComplete?.();
      return;
    }

    const phase = phases[currentPhaseIndex];
    let remaining = phase.duration;
    let elapsed = 0;
    
    // Update display
    dom.timerDisplay.textContent = formatTime(remaining);
    dom.timerDisplay.classList.remove('hidden');
    
    hooks.onPhaseStart?.(phase);

    timerInterval = setInterval(() => {
      remaining--;
      elapsed++;
      dom.timerDisplay.textContent = formatTime(remaining);
      
      // Update recording timer visual if speaking phase
      if (phase.id === 'speak' || phase.id === 'retell') {
        updateRecordingTimer(elapsed, phase.duration);
      }
      
      if (remaining <= 0) {
        clearInterval(timerInterval);
        advancePhase();
      }
    }, 1000);
  };

  advancePhase();
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function stopTimer(expired = false) {
  clearInterval(timerInterval);
  timerInterval = null;
  if (expired) {
    dom.timerDisplay.textContent = '00:00';
    dom.timerDisplay.classList.add('hidden');
  } else {
    dom.timerDisplay.classList.add('hidden');
  }
}

function renderSummary() {
  calculateScores();
  dom.summaryMetrics.innerHTML = Object.entries(appState.scores).map(([label, value]) => `
    <div class="summary-card">
      <div class="score">${value}</div>
      <div class="label">${label}</div>
    </div>
  `).join('');

  dom.summaryLog.innerHTML = appState.log.map(entry => `
    <div class="summary-log-item">
      <strong>${entry.section}</strong>: ${entry.text || '(no response)'}
    </div>
  `).join('');
}

async function evaluateResponse(userResponse, activityType, referenceText = null) {
  try {
    const response = await fetch(`${API_BASE_URL}/evaluate-response`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        user_response: userResponse,
        activity_type: activityType,
        reference_text: referenceText
      })
    });
    
    if (response.ok) {
      return await response.json();
    } else {
      console.error('Evaluation failed:', response.statusText);
      return null;
    }
  } catch (err) {
    console.error('Error evaluating response:', err);
    return null;
  }
}
function calculateScores() {
  if (appState.scored) return;
  
  // Scores calculated from LLM evaluations
  let fluency = 0;
  let accuracy = 0;
  let comprehension = 0;
  let completeness = 0;
  let evaluationCount = 0;
  
  // Collect all evaluations
  appState.log.forEach(entry => {
    if (entry.evaluation) {
      fluency += entry.evaluation.fluency_score || 0;
      accuracy += entry.evaluation.accuracy_score || 0;
      comprehension += entry.evaluation.grammar_score || 0;
      completeness += entry.evaluation.pronunciation_clarity || 0;
      evaluationCount++;
    }
  });
  
  // Calculate averages
  if (evaluationCount > 0) {
    fluency = Math.round(fluency / evaluationCount);
    accuracy = Math.round(accuracy / evaluationCount);
    comprehension = Math.round(comprehension / evaluationCount);
    completeness = Math.round(completeness / evaluationCount);
  } else {
    // Fallback to basic calculation if no evaluations
    const respondedCount = appState.log.filter(r => r.text).length;
    const totalCount = appState.log.length;
    const responseRate = totalCount > 0 ? (respondedCount / totalCount) * 100 : 0;
    fluency = Math.round(responseRate * 0.8 + 20);
    accuracy = Math.round(responseRate * 0.8 + 20);
    comprehension = Math.round(responseRate * 0.8 + 20);
    completeness = Math.round(responseRate * 0.8 + 20);
  }
  
  appState.scores = { 
    fluency: Math.min(fluency, 100),
    accuracy: Math.min(accuracy, 100),
    comprehension: Math.min(comprehension, 100),
    completeness: Math.min(completeness, 100)
  };
  appState.scored = true;
  persistState();
}

function resetPractice() {
  appState = initialState();
  persistState();
  updateUI();
}

function shuffleArray(array) {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

function downloadResults() {
  // Download test results as JSON
  try {
    const results = {
      timestamp: new Date().toISOString(),
      scores: appState.scores,
      log: appState.log,
      totalResponses: appState.log.length,
      responseRate: appState.log.filter(r => r.text).length / appState.log.length
    };

    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `versant-results-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error('Download error:', err);
    alert('Failed to download results');
  }
}