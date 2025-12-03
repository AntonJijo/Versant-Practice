
/**
 * Versant Practice Application - Frontend
 * Handles UI orchestration, audio recording, LLM-based scoring integration
 */

const STORAGE_KEY = 'geminiPracticeSession';

// API Configuration - Backend server URL
// If frontend is on a different port (e.g., Live Server on 5500), set this to your backend URL

const API_BASE_URL =
  (window.location.port === '5500' || window.location.port === '3000')
    ? 'http://localhost:8000'                       // dev: FastAPI on 8000
    : 'https://versant-practice.onrender.com'; // prod: Render backend


// Log API configuration for debugging
console.log('API Base URL:', API_BASE_URL);
console.log('Frontend URL:', window.location.origin);

// Audio bank removed - strictly using dynamic generation


const sections = [
  {
    id: 'repeats',
    title: 'Read Aloud',
    description: 'Listen to each sentence and repeat it clearly.',
    instructions: 'Please repeat each sentence exactly as you hear it. You will hear the sentence only once.',
    example: 'If you hear "The meeting is at 5 PM", say "The meeting is at 5 PM".',
    timerProfile: { phases: [{ id: 'listen', label: 'Listen', duration: 5 }, { id: 'speak', label: 'Speak', duration: 5 }] }
  },
  {
    id: 'conversation',
    title: 'Answer Question',
    description: 'Listen to the conversation and answer the question.',
    instructions: 'You will hear a short question. Give a simple, direct answer.',
    example: 'Question: "What time do you have lunch?" Answer: "At 12:30 PM" or "Around noon".',
    timerProfile: { phases: [{ id: 'listen', label: 'Listen', duration: 5 }, { id: 'speak', label: 'Speak', duration: 5 }] }
  },
  {
    id: 'jumbled',
    title: 'Sentence Assembly',
    description: 'Listen to the words and form a complete sentence.',
    instructions: 'You will hear a list of words. Rearrange them to form a correct English sentence.',
    example: 'Words: "is", "sky", "blue", "the". Answer: "The sky is blue".',
    timerSeconds: 10
  },
  {
    id: 'dictation',
    title: 'Dictation',
    description: 'Listen and type exactly what you hear.',
    instructions: 'Type the sentence exactly as spoken. Pay attention to spelling and punctuation.',
    example: 'You hear: "Please close the door." You type: "Please close the door."',
    timerSeconds: 10
  },
  {
    id: 'fill',
    title: 'Fill in the Blank',
    description: 'Complete the sentence with the correct word.',
    instructions: 'Read the sentence and fill in the missing word that makes the most sense.',
    example: 'Sentence: "She ____ to the store." Answer: "went" or "walked".',
    timerSeconds: 8
  },
  {
    id: 'passage',
    title: 'Retell',
    description: 'Read the passage, then retell it in your own words.',
    instructions: 'Read the story silently. Then, retell the story in your own words, including as many details as possible.',
    example: 'Read a short story about a lost dog. Then tell what happened to the dog.',
    timerProfile: { phases: [{ id: 'read', label: 'Read', duration: 8 }, { id: 'retell', label: 'Retell', duration: 12 }] }
  },
];

// Initial empty prompts - will be populated by loadAllQuestions
let repeatPrompts = [];
let conversationPrompts = [];
let jumbledPrompts = [];
let dictationPrompts = [];
let fillPrompts = [];
let passagePrompt = { text: '' };

const initialState = () => ({
  sectionIndex: 0,
  itemIndex: 0,
  stepsCompleted: 0,
  showingInstructions: true, // Start with instructions for the first section
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
  beepFlash: document.getElementById('beepFlash'),
  loadingScreen: document.getElementById('loadingScreen'),
  loadingStatus: document.getElementById('loadingStatus'),
  // Email verification elements
  emailModal: document.getElementById('emailModal'),
  emailInput: document.getElementById('emailInput'),
  verifyEmailBtn: document.getElementById('verifyEmailBtn'),
  emailError: document.getElementById('emailError')
};

function getTotalSteps() {
  return sections.reduce((acc, section) => acc + getItemCountForSection(section.id), 0);
}

document.addEventListener('DOMContentLoaded', () => {
  bootstrap();
});

function bootstrap() {
  // Enforce email verification
  if (dom.emailModal) {
    dom.emailModal.classList.remove('hidden');

    // Prevent closing by clicking outside or escape (modal-overlay usually covers this, but let's be sure)
    // The HTML structure doesn't have a close button for this modal, which is good.

    dom.verifyEmailBtn.addEventListener('click', handleEmailVerification);
    dom.emailInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') handleEmailVerification();
    });
  }

  attachGlobalEvents();
  // Don't update UI yet, wait for verification
  // updateUI(); 
}

async function handleEmailVerification() {
  const email = dom.emailInput.value.trim();
  const errorDiv = dom.emailError;
  const btn = dom.verifyEmailBtn;

  if (!email) {
    errorDiv.textContent = "Please enter an email address.";
    return;
  }

  // Basic format check
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    errorDiv.textContent = "Please enter a valid email format.";
    return;
  }

  try {
    btn.disabled = true;
    btn.textContent = "Verifying...";
    errorDiv.textContent = "";

    const response = await fetch(`${API_BASE_URL}/validate-email`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email })
    });

    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || "Verification failed");
    }

    const data = await response.json();
    if (data.status === 'valid') {
      // Success
      dom.emailModal.classList.add('hidden');
      updateUI(); // Now we can show the UI
    } else {
      throw new Error("Email not authorized");
    }

  } catch (err) {
    console.error("Email verification error:", err);
    errorDiv.textContent = err.message || "Connection failed. Please check backend.";
    btn.disabled = false;
    btn.textContent = "Verify & Proceed";
  }
}

async function loadAllQuestions() {
  console.log('Loading all questions from LLM...');
  dom.loadingStatus.textContent = "Connecting to question generator...";

  try {
    const response = await fetch(`${API_BASE_URL}/generate-all-questions`);

    if (!response.ok) {
      throw new Error(`Failed to fetch questions: ${response.status}`);
    }

    const data = await response.json();
    console.log('Received questions data:', data);

    // Process Repeats
    if (data.repeats && data.repeats.length > 0) {
      dom.loadingStatus.textContent = "Preparing audio for Repeats section...";
      repeatPrompts = await Promise.all(data.repeats.map(async (q, i) => {
        const text = q.text || q.transcript;
        const audio = await generateTextToSpeech(text);
        return {
          id: q.id,
          transcript: text,
          audio: audio
        };
      }));
    }

    // Process Short Answer (Conversation)
    if (data.short_answer && data.short_answer.length > 0) {
      dom.loadingStatus.textContent = "Preparing audio for Short Answer section...";
      conversationPrompts = await Promise.all(data.short_answer.map(async (q) => {
        const audio = await generateTextToSpeech(q.question);
        return {
          id: q.id,
          audio: audio,
          exchange: q.exchange || q.question,
          question: q.question
        };
      }));
    }

    // Process Sentence Builds (Jumbled)
    if (data.sentence_builds && data.sentence_builds.length > 0) {
      dom.loadingStatus.textContent = "Preparing audio for Sentence Builds section...";
      jumbledPrompts = await Promise.all(data.sentence_builds.map(async (q) => {
        const audio = await generateTextToSpeech(q.correct);
        return {
          id: q.id,
          audio: audio,
          correct: q.correct
        };
      }));
    }

    // Process Story Retelling (Passage)
    if (data.story_retelling && data.story_retelling.length > 0) {
      const story = data.story_retelling[0];
      passagePrompt = {
        id: story.id,
        text: story.text
      };
    }

    // Process Open Questions (mapped to dictation/reading)
    if (data.reading && data.reading.length > 0) {
      dictationPrompts = data.reading.map(q => ({
        id: q.id,
        transcript: q.text
      }));
    }

    dom.loadingStatus.textContent = "Ready!";

    // Re-initialize state to ensure responses structure matches the newly loaded questions
    appState = initialState();
    persistState();

    return true;

  } catch (err) {
    console.error('Failed to load all questions:', err);

    // Show technical error screen
    dom.loadingScreen.innerHTML = `
      <div class="gemini-card" style="text-align: center; max-width: 500px; margin: 0 auto; border-color: var(--error-color);">
        <div style="font-size: 3rem; color: var(--error-color); margin-bottom: 1rem;">
          <i class="fas fa-exclamation-triangle"></i>
        </div>
        <h2 style="color: var(--error-color); margin-bottom: 1rem;">Technical Issue</h2>
        <p style="margin-bottom: 1.5rem;">We could not connect to the question server. Please contact the administrator.</p>
        <div style="padding: 1rem; background: rgba(244, 67, 54, 0.1); border-radius: 0.5rem; font-family: monospace; font-size: 0.8rem; color: var(--on-surface-variant-color);">
          ${err.message || 'Unknown network error'}
        </div>
        <button onclick="location.reload()" class="gemini-button secondary" style="margin-top: 1.5rem;">Try Again</button>
      </div>
    `;

    // Return false to stop the test from starting
    return false;
  }
}

// Play audio - handles both Puter audio objects and data URLs
let currentAudioElement = null;

// Play audio - handles both Puter audio objects and data URLs
async function playAudio(audioSource, onEnded = null) {
  try {
    // Stop any currently playing audio first
    stopAudio();

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

    currentAudioElement = audioElement;
    return audioElement;
  } catch (err) {
    console.error('Audio playback failed:', err);
    return null;
  }
}

function stopAudio() {
  if (currentAudioElement) {
    try {
      currentAudioElement.pause();
      if (currentAudioElement.currentTime) {
        currentAudioElement.currentTime = 0;
      }
    } catch (e) {
      console.warn('Error stopping audio:', e);
    }
    currentAudioElement = null;
  }
}

// Generate audio from text using Puter AI Text-to-Speech
async function generateTextToSpeech(text, preferredVoice = null) {
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
      // List of American English voices typically available in AWS Polly (which Puter likely uses)
      const americanVoices = ['Joanna', 'Salli', 'Matthew', 'Ivy', 'Justin', 'Kendra', 'Kimberly', 'Joey'];

      let selectedVoice;
      if (preferredVoice === 'male') {
        selectedVoice = 'Matthew'; // Fixed male voice for instructions
      } else if (preferredVoice && americanVoices.includes(preferredVoice)) {
        selectedVoice = preferredVoice;
      } else {
        selectedVoice = americanVoices[Math.floor(Math.random() * americanVoices.length)];
      }

      console.log(`Using Puter.js TTS with voice ${selectedVoice} for:`, text.substring(0, 50));
      const audio = await puter.ai.txt2speech(text, {
        voice: selectedVoice,
        engine: 'neural',
        language: 'en-US'
      });
      console.log('Puter.js TTS generated successfully');
      return audio;
    } else {
      console.warn('Puter.js TTS not available or failed to load.');
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
  // Always start fresh to ensure questions match the session
  // Since questions are generated dynamically and stored in memory, 
  // we cannot resume a session after reload without re-fetching/re-generating.
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (e) {
    console.warn('Failed to clear storage', e);
  }
  return initialState();
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

  // If we are loading, we might want to hide everything else, but updateUI is usually called after state changes.
  // We'll handle loading screen visibility manually in beginTestFlow.

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
    dom.progressIndicator.textContent = `Question ${appState.stepsCompleted} of ${getTotalSteps()}`;
    // Show navigation buttons during test, but NOT during instructions
    if (dom.navigationButtons) {
      if (appState.showingInstructions) {
        dom.navigationButtons.classList.add('hidden');
      } else {
        dom.navigationButtons.classList.remove('hidden');
      }
    }
  }
}

async function beginTestFlow() {
  // Show loading screen
  dom.startScreen.classList.add('hidden');
  dom.loadingScreen.classList.remove('hidden');

  // Load questions
  const success = await loadAllQuestions();

  if (!success) {
    // Error screen is already rendered by loadAllQuestions
    return;
  }

  // Hide loading screen
  dom.loadingScreen.classList.add('hidden');

  appState.stepsCompleted = 1;
  updateUI();
}

function advanceStep(direction = 1) {
  stopRecognition();
  stopTimer();
  stopAudio(); // Ensure any playing audio is stopped
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

  // Disable next button temporarily to prevent double clicks
  if (dom.nextButton) {
    dom.nextButton.disabled = true;
    setTimeout(() => {
      if (dom.nextButton) dom.nextButton.disabled = false;
    }, 1000);
  }

  const itemsInSection = getItemCountForSection(currentSection.id);
  if (appState.itemIndex < itemsInSection - 1) {
    appState.itemIndex++;
  } else {
    appState.sectionIndex++;
    appState.itemIndex = 0;
    appState.showingInstructions = true; // Show instructions for the new section
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

function renderInstructionScreen(section) {
  const wrapper = document.createElement('div');
  wrapper.className = 'instruction-screen';

  // Determine if this is an audio-heavy section where we should offer spoken instructions
  const audioSections = ['repeats', 'conversation', 'jumbled', 'dictation'];
  const isAudioSection = audioSections.includes(section.id);

  // Auto-play instructions for audio sections
  if (isAudioSection) {
    setTimeout(async () => {
      // Check if we are still on the instruction screen to avoid playing if user clicked start quickly
      if (!appState.showingInstructions) return;

      const fullText = `${section.instructions}. For example: ${section.example}`;
      try {
        const audio = await generateTextToSpeech(fullText, 'male');
        if (audio && appState.showingInstructions) {
          await playAudio(audio);
        }
      } catch (e) {
        console.error("Failed to auto-play instructions:", e);
      }
    }, 500);
  }

  wrapper.innerHTML = `
    <div class="instruction-content">
      <div class="start-screen-icon">
        <i class="fas fa-info-circle"></i>
      </div>
      <h2>${section.title}</h2>
      <p>${section.instructions}</p>
      
      <div class="example-box">
        <h4>Example:</h4>
        <p>${section.example}</p>
      </div>
      
      <button id="startSectionBtn" class="gemini-button">Start Section</button>
    </div>
  `;


  return wrapper;
}


function renderCurrentSection() {
  const section = sections[appState.sectionIndex];
  if (!section) {
    console.warn('No section at index', appState.sectionIndex);
    return;
  }
  // Check if we should show instructions first
  if (appState.showingInstructions) {
    const instructionNode = renderInstructionScreen(section);
    dom.sectionContent.innerHTML = '';
    dom.sectionContent.appendChild(instructionNode);

    document.getElementById('startSectionBtn').addEventListener('click', () => {
      stopAudio(); // Stop instruction audio if playing
      appState.showingInstructions = false;
      persistState();
      updateUI();
    });

    // Hide timer and nav buttons during instructions
    dom.timerDisplay.classList.add('hidden');
    if (dom.navigationButtons) {
      dom.navigationButtons.classList.add('hidden');
    }
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

  if (!prompt) {
    console.warn(`No prompt found for repeats index ${index}`);
    return { node: document.createTextNode('Loading question...') };
  }

  const response = appState.responses.repeats[index] || { id: prompt.id, transcript: prompt.transcript, recognized: '', manual: '', attempts: 0 };
  // Ensure response exists in state if it was missing (e.g. due to count mismatch)
  if (!appState.responses.repeats[index]) {
    appState.responses.repeats[index] = response;
  }

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

        // Pause timer while audio plays to prevent premature advance
        pauseTimer();

        playAudio(audioSource, () => {
          // Audio ended - immediately advance to speaking phase
          advancePhase();
        });
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

  if (!prompt) {
    console.warn(`No prompt found for conversation index ${index}`);
    return { node: document.createTextNode('Loading question...') };
  }

  const response = appState.responses.conversation[index] || { id: prompt.id, recognized: '', manual: '', attempts: 0 };
  if (!appState.responses.conversation[index]) {
    appState.responses.conversation[index] = response;
  }

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

        pauseTimer(); // Pause timer for audio

        playAudio(audioSource, () => {
          animateAudioBars(audioVisualization, false);
          statusText.textContent = 'Question complete. Ready to answer.';
          advancePhase(); // Immediately advance to speaking phase
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

  if (!prompt) {
    console.warn(`No prompt found for jumbled index ${index}`);
    return { node: document.createTextNode('Loading question...') };
  }

  const response = appState.responses.jumbled[index] || { id: prompt.id, recognized: '', correct: null };
  if (!appState.responses.jumbled[index]) {
    appState.responses.jumbled[index] = response;
  }

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

        pauseTimer(); // Pause timer during audio

        playAudio(audioSource, () => {
          animateAudioBars(audioVisualization, false);
          statusText.textContent = 'Now, form a sentence by speaking.';
          micContainer.classList.remove('hidden');

          resumeTimer(); // Resume timer for answering

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

  // Ensure audio exists, otherwise generate it (fallback for dictation)
  let audioSource = prompt.audio;

  const hooks = {
    onPhaseStart: async (phase) => {
      if (phase.id === 'default' || !phase.id) {
        statusText.textContent = 'Listening to sentence...';
        animateAudioBars(audioVisualization, true);

        pauseTimer(); // Pause global timer

        if (!audioSource) {
          // Generate audio on the fly if missing
          audioSource = await generateTextToSpeech(prompt.transcript);
          // Cache it
          prompt.audio = audioSource;
        }

        playAudio(audioSource, () => {
          animateAudioBars(audioVisualization, false);
          audioContainer.innerHTML = ''; // Clear audio visualization

          resumeTimer(); // Resume global timer

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

          // Removed manual timer as global timer handles it
          // Just update UI if needed, but global timer updates dom.timerDisplay

          textBox.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
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
      // Handle timer completion for dictation
      const textBox = wrapper.querySelector('.response-area');
      if (textBox) {
        response.typed = textBox.value;
        persistState();
      }
      advanceStep(1);
    }
  };

  return { node: wrapper, hooks, timerProfile: section.timerProfile };
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

let activeTimer = {
  interval: null,
  paused: false,
  remaining: 0,
  elapsed: 0,
  phase: null,
  phases: [],
  phaseIndex: 0,
  hooks: {},
  config: null
};

function startTimer(config, hooks = {}) {
  stopTimer();
  dom.timerDisplay.classList.remove('hidden');

  activeTimer.config = config;
  activeTimer.phases = config.phases;
  activeTimer.hooks = hooks;
  activeTimer.phaseIndex = -1;
  activeTimer.paused = false;

  advancePhase();
}

function advancePhase() {
  activeTimer.paused = false; // Ensure timer is unpaused when advancing
  activeTimer.phaseIndex++;
  if (activeTimer.phaseIndex >= activeTimer.phases.length) {
    stopTimer(true);
    activeTimer.hooks.onTimerComplete?.();
    return;
  }

  activeTimer.phase = activeTimer.phases[activeTimer.phaseIndex];
  activeTimer.remaining = activeTimer.phase.duration;
  activeTimer.elapsed = 0;

  // Update display
  dom.timerDisplay.textContent = formatTime(activeTimer.remaining);
  dom.timerDisplay.classList.remove('hidden');

  activeTimer.hooks.onPhaseStart?.(activeTimer.phase);

  startInterval();
}

function startInterval() {
  if (activeTimer.interval) clearInterval(activeTimer.interval);

  activeTimer.interval = setInterval(() => {
    if (activeTimer.paused) return;

    activeTimer.remaining--;
    activeTimer.elapsed++;
    dom.timerDisplay.textContent = formatTime(activeTimer.remaining);

    // Update recording timer visual if speaking phase
    if (activeTimer.phase.id === 'speak' || activeTimer.phase.id === 'retell') {
      updateRecordingTimer(activeTimer.elapsed, activeTimer.phase.duration);
    }

    if (activeTimer.remaining <= 0) {
      clearInterval(activeTimer.interval);
      advancePhase();
    }
  }, 1000);
}

function pauseTimer() {
  activeTimer.paused = true;
  // We don't clear interval, just set flag to skip ticks. 
  // Or better, clear interval to stop CPU usage, but flag is easier to resume.
  // Actually, if we just set flag, the 'remaining' won't decrement.
  // But the interval keeps firing.
  // Let's clear interval to be clean.
  if (activeTimer.interval) {
    clearInterval(activeTimer.interval);
    activeTimer.interval = null;
  }
}

function resumeTimer() {
  if (activeTimer.paused) {
    activeTimer.paused = false;
    startInterval();
  }
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function stopTimer(expired = false) {
  if (activeTimer.interval) {
    clearInterval(activeTimer.interval);
    activeTimer.interval = null;
  }
  activeTimer.paused = false;

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
