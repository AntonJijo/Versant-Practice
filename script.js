/**
 * Versant-style English Test Practice
 * Client-side SPA using Tailwind CSS and Web APIs only.
 * -----------------------------------------------
 * Sections:
 * 1. Repeat short sentences after listening.
 * 2. Arrange jumbled sentence fragments.
 * 3. Fill in missing words.
 * 4. Voice to text warm-up.
 * 5. Retell a short story.
 *
 * Speech recognition uses the Web Speech API where available,
 * falling back to manual text inputs otherwise.
 *
 * State is persisted in localStorage for the duration of the session.
 */

const STORAGE_KEY = 'versantPracticeSession';

// Placeholder audio clips encoded as data URIs (mp3).
// Each clip is ~1 second long with simple tones to avoid copyrighted material.
const audioBank = {
  repeat1:
    'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQUZABp6AAD//+wbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFJwAaegAA//+sGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=',
  repeat2:
    'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQU5ABp6AAD//9AbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFNwAaegAA//9AbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=',
  repeat3:
    'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQUZABp6AAD//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFNwAaegAA//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=',
  story:
    'data:audio/mp3;base64,SUQzAwAAAAAAF1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMQAAAAAAAAAAAAAA//uQxAADBQUZABp6AAD//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/7kMQAAwUFNwAaegAA//9gbAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA='
};

const sections = [
  {
    id: 'repeats',
    title: 'Repeats',
    description: 'Listen carefully to each clip, then repeat what you heard using the microphone.',
    timerProfile: {
      phases: [
        { id: 'listen', label: 'Listen', duration: 4 },
        { id: 'speak', label: 'Speak', duration: 3 }
      ]
    }
  },
  {
    id: 'conversation',
    title: 'Short Answer',
    description: 'Listen to the conversation and question, then provide a brief answer.',
    timerProfile: {
      phases: [
        { id: 'listen', label: 'Listen', duration: 8 },
        { id: 'speak', label: 'Speak', duration: 4 }
      ]
    }
  },
  {
    id: 'jumbled',
    title: 'Jumbled Sentence',
    description: 'Drag the fragments into the correct order to form a complete sentence.',
    timerSeconds: 60
  },
  {
    id: 'dictation',
    title: 'Dictation',
    description: 'Listen to the sentence, then type exactly what you heard.',
    timerSeconds: 25
  },
  {
    id: 'fill',
    title: 'Fill in the Blank',
    description: 'Complete each sentence by choosing the best option or typing the missing word.',
    timerSeconds: 90
  },
  {
    id: 'passage',
    title: 'Passage Reconstruction',
    description: 'Read the paragraph, then type what you remember after it disappears.',
    timerProfile: {
      phases: [
        { id: 'view', label: 'View', duration: 30 },
        { id: 'write', label: 'Write', duration: 90 }
      ]
    }
  },
  {
    id: 'voice',
    title: 'Voice to Text',
    description: 'Speak freely about the prompt. Your words will appear on the screen in real time.',
    timerProfile: {
      phases: [
        { id: 'prep', label: 'Prepare', duration: 5 },
        { id: 'speak', label: 'Speak', duration: 45 }
      ]
    }
  },
  {
    id: 'story',
    title: 'Story Retelling',
    description: 'Listen to the short story. When it finishes, retell it in your own words.',
    timerProfile: {
      phases: [
        { id: 'listen', label: 'Listen', duration: 12 },
        { id: 'prep', label: 'Prepare', duration: 3 },
        { id: 'speak', label: 'Speak', duration: 40 }
      ]
    }
  }
];

const repeatPrompts = [
  {
    id: 'repeat-1',
    transcript: 'The weather will clear up by this evening.',
    audio: audioBank.repeat1
  },
  {
    id: 'repeat-2',
    transcript: 'She always takes the early train to work.',
    audio: audioBank.repeat2
  },
  {
    id: 'repeat-3',
    transcript: 'Let’s schedule a follow-up meeting next week.',
    audio: audioBank.repeat3
  }
];

const conversationPrompts = [
  {
    id: 'conv-1',
    exchange: 'A: Did you finish the report? B: Yes, I sent it yesterday.',
    question: 'When was the report sent?',
    answer: 'yesterday'
  },
  {
    id: 'conv-2',
    exchange: 'A: Where is the meeting? B: In the conference room on the third floor.',
    question: 'Where is the meeting?',
    answer: 'conference room'
  }
];

const dictationPrompts = [
  {
    id: 'dict-1',
    transcript: 'The meeting will begin at three o\'clock in the afternoon.'
  },
  {
    id: 'dict-2',
    transcript: 'Please send the documents to the main office by Friday.'
  },
  {
    id: 'dict-3',
    transcript: 'She completed the project ahead of schedule.'
  }
];

const passagePrompt = {
  id: 'passage-1',
  text: 'A small community garden faced a drought, threatening the vegetables that volunteers tended each weekend. One student suggested collecting rainwater from nearby roofs. Working together, the neighbors installed barrels, and when the next rain finally arrived, the tanks filled. The garden recovered, and the project became a model collaboration for the town.'
};

const jumbledPrompts = [
  {
    id: 'jumbled-1',
    fragments: ['completed', 'has', 'She', 'assignment', 'her'],
    correct: 'She has completed her assignment'
  },
  {
    id: 'jumbled-2',
    fragments: ['started', 'meeting', 'The', 'promptly'],
    correct: 'The meeting started promptly'
  },
  {
    id: 'jumbled-3',
    fragments: ['around', 'people', 'world', 'travel', 'the'],
    correct: 'People travel around the world'
  }
];

const fillPrompts = [
  {
    id: 'fill-1',
    sentence: 'The quick brown ____ jumped over the lazy dog.',
    type: 'choice',
    options: ['fox', 'cat', 'wolf'],
    answer: 'fox'
  },
  {
    id: 'fill-2',
    sentence: 'Please ____ the meeting notes by tomorrow morning.',
    type: 'choice',
    options: ['review', 'paint', 'boil'],
    answer: 'review'
  },
  {
    id: 'fill-3',
    sentence: 'I have ____ the presentation slides already.',
    type: 'text',
    answer: 'prepared'
  }
];

const voicePrompt =
  'Talk about a recent challenge you overcame and what you learned from the experience.';

const storyPrompt = {
  audio: audioBank.story,
  transcript:
    'A small community garden faced a drought, threatening the vegetables that volunteers tended each weekend. One student suggested collecting rainwater from nearby roofs. Working together, the neighbors installed barrels, and when the next rain finally arrived, the tanks filled. The garden recovered, and the project became a model collaboration for the town.'
};

const initialState = () => ({
  sectionIndex: 0,
  itemIndex: 0,
  stepsCompleted: 0,
  responses: {
    repeats: repeatPrompts.map((prompt) => ({
      id: prompt.id,
      transcript: prompt.transcript,
      recognized: '',
      manual: '',
      attempts: 0
    })),
    conversation: conversationPrompts.map((prompt) => ({
      id: prompt.id,
      recognized: '',
      manual: '',
      attempts: 0
    })),
    jumbled: jumbledPrompts.map((prompt) => ({
      id: prompt.id,
      arrangement: [],
      correct: null
    })),
    dictation: dictationPrompts.map((prompt) => ({
      id: prompt.id,
      transcript: prompt.transcript,
      typed: '',
      attempts: 0
    })),
    fill: fillPrompts.map((prompt) => ({
      id: prompt.id,
      response: '',
      correct: null
    })),
    passage: {
      text: passagePrompt.text,
      typed: '',
      timestamp: null
    },
    voice: {
      transcript: '',
      timestamp: null
    },
    story: {
      transcript: '',
      timestamp: null
    }
  },
  log: [],
  scored: false,
  scores: {
    fluency: 0,
    accuracy: 0,
    comprehension: 0,
    completeness: 0
  }
});

let appState = loadState();
let recognitionSupported = 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window;
let activeRecognition = null;
let recognitionMode = null;
let timerInterval = null;
let currentTimerPhases = [];
let currentPhaseIndex = -1;
let currentTimerHooks = null;

const dom = {
  startScreen: document.getElementById('startScreen'),
  testContainer: document.getElementById('testContainer'),
  summaryScreen: document.getElementById('summaryScreen'),
  sectionContent: document.getElementById('sectionContent'),
  progressBar: document.getElementById('progressBar'),
  progressLabel: document.getElementById('progressLabel'),
  timerBadge: document.getElementById('timerBadge'),
  nextBtn: document.getElementById('nextBtn'),
  prevBtn: document.getElementById('prevBtn'),
  restartBtn: document.getElementById('restartBtn'),
  summaryMetrics: document.getElementById('summaryMetrics'),
  summaryLog: document.getElementById('summaryLog'),
  instructionsModal: document.getElementById('instructionsModal'),
  openInstructions: document.getElementById('openInstructions'),
  startInstructions: document.getElementById('startInstructions'),
  modalClose: document.getElementById('modalClose'),
  modalStartTest: document.getElementById('modalStartTest'),
  closeInstructions: document.getElementById('closeInstructions')
};

const totalSteps =
  repeatPrompts.length +
  conversationPrompts.length +
  jumbledPrompts.length +
  dictationPrompts.length +
  fillPrompts.length +
  1 + // passage
  1 + // voice
  1; // story

bootstrap();

/**
 * Initialise application events and render first view.
 */
function bootstrap() {
  attachGlobalEvents();
  updateProgressUI();

  if (appState.sectionIndex === 0 && appState.stepsCompleted === 0) {
    showStartScreen();
  } else if (appState.sectionIndex >= sections.length) {
    renderSummary();
  } else {
    showTestContainer();
    renderCurrentSection();
  }
}

/**
 * Persist state to localStorage.
 */
function persistState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(appState));
}

/**
 * Load state from localStorage or return defaults.
 */
function loadState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return initialState();
  try {
    const parsed = JSON.parse(raw);
    return mergeWithInitial(parsed);
  } catch (error) {
    console.warn('Failed to parse stored session. Resetting.', error);
    return initialState();
  }
}

/**
 * Ensures newly added fields exist when upgrading the schema.
 */
function mergeWithInitial(stored) {
  const fresh = initialState();
  return {
    ...fresh,
    ...stored,
    responses: {
      ...fresh.responses,
      ...stored.responses
    },
    scores: {
      ...fresh.scores,
      ...stored.scores
    }
  };
}

/**
 * Global event listeners.
 */
function attachGlobalEvents() {
  document.getElementById('startTest').addEventListener('click', beginTestFlow);
  dom.startInstructions.addEventListener('click', () => toggleModal(true));
  dom.openInstructions.addEventListener('click', () => toggleModal(true));
  dom.modalClose.addEventListener('click', () => toggleModal(false));
  dom.closeInstructions.addEventListener('click', () => toggleModal(false));
  dom.instructionsModal.addEventListener('click', (event) => {
    if (event.target === dom.instructionsModal) toggleModal(false);
  });
  dom.modalStartTest.addEventListener('click', () => {
    toggleModal(false);
    beginTestFlow();
  });
  dom.nextBtn.addEventListener('click', () => advanceStep(1));
  dom.prevBtn.addEventListener('click', () => advanceStep(-1));
  dom.restartBtn.addEventListener('click', resetPractice);
}

function toggleModal(visible) {
  dom.instructionsModal.classList.toggle('hidden', !visible);
  if (visible) {
    dom.instructionsModal.classList.add('flex');
  } else {
    dom.instructionsModal.classList.remove('flex');
  }
}

function showStartScreen() {
  dom.startScreen.classList.remove('hidden');
  dom.testContainer.classList.add('hidden');
  dom.summaryScreen.classList.add('hidden');
}

function showTestContainer() {
  dom.startScreen.classList.add('hidden');
  dom.summaryScreen.classList.add('hidden');
  dom.testContainer.classList.remove('hidden');
}

function beginTestFlow() {
  showTestContainer();
  renderCurrentSection(true);
}

function advanceStep(direction) {
  stopRecognition();
  stopTimer();

  const { sectionIndex, itemIndex } = appState;
  const currentSection = sections[sectionIndex];
  const itemsInSection = getItemCountForSection(currentSection.id);

  if (direction === 1) {
    if (itemIndex < itemsInSection - 1) {
      appState.itemIndex += 1;
      appState.stepsCompleted += 1;
    } else {
      appState.sectionIndex += 1;
      appState.itemIndex = 0;
      appState.stepsCompleted += 1;
    }
  } else if (direction === -1) {
    if (sectionIndex === 0 && itemIndex === 0) {
      return;
    }
    if (itemIndex > 0) {
      appState.itemIndex -= 1;
      appState.stepsCompleted = Math.max(appState.stepsCompleted - 1, 0);
    } else {
      const prevSectionId = sections[sectionIndex - 1].id;
      const prevItems = getItemCountForSection(prevSectionId);
      appState.sectionIndex -= 1;
      appState.itemIndex = prevItems - 1;
      appState.stepsCompleted = Math.max(appState.stepsCompleted - 1, 0);
    }
  }

  persistState();

  if (appState.sectionIndex >= sections.length) {
    renderSummary();
    return;
  }

  updateProgressUI();
  renderCurrentSection(true);
}

function getItemCountForSection(sectionId) {
  switch (sectionId) {
    case 'repeats':
      return repeatPrompts.length;
    case 'conversation':
      return conversationPrompts.length;
    case 'jumbled':
      return jumbledPrompts.length;
    case 'dictation':
      return dictationPrompts.length;
    case 'fill':
      return fillPrompts.length;
    case 'passage':
    case 'voice':
    case 'story':
      return 1;
    default:
      return 1;
  }
}

function updateProgressUI() {
  const progressRatio = Math.min(appState.stepsCompleted / totalSteps, 1);
  dom.progressBar.style.width = `${progressRatio * 100}%`;
  dom.progressLabel.textContent = `${Math.round(progressRatio * 100)}%`;

  const prevDisabled = appState.sectionIndex === 0 && appState.itemIndex === 0;
  dom.prevBtn.disabled = prevDisabled;

  const nextLabel =
    appState.sectionIndex === sections.length - 1 && appState.itemIndex === 0
      ? 'Finish'
      : 'Next';
  dom.nextBtn.textContent = nextLabel;
}

function renderCurrentSection(withAnimation = false) {
  const section = sections[appState.sectionIndex];
  if (!section) return;

  dom.sectionContent.innerHTML = '';
  dom.nextBtn.disabled = false;

  const container = document.createElement('div');
  container.className =
    'card-fade space-y-6 rounded-3xl border border-slate-800 bg-slate-900/70 p-6 shadow-xl';

  const header = document.createElement('div');
  header.className = 'flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between';
  header.innerHTML = `
    <div>
      <p class="text-xs uppercase tracking-[0.3em] text-slate-500">Activity ${
        appState.sectionIndex + 1
      } of ${sections.length}</p>
      <h2 class="text-2xl font-semibold text-slate-100">${section.title}</h2>
    </div>
    <span class="text-sm text-slate-400">Item ${appState.itemIndex + 1} of ${getItemCountForSection(
    section.id
  )}</span>
  `;

  const description = document.createElement('p');
  description.className = 'text-slate-300 leading-relaxed text-sm sm:text-base';
  description.textContent = section.description;

  container.appendChild(header);
  container.appendChild(description);

  const contentWrapper = document.createElement('div');
  contentWrapper.className = 'space-y-5';

  let renderResult = null;

  switch (section.id) {
    case 'repeats':
      renderResult = renderRepeatsSection(appState.itemIndex);
      break;
    case 'conversation':
      renderResult = renderConversationSection(appState.itemIndex);
      break;
    case 'jumbled':
      renderResult = renderJumbledSection(appState.itemIndex);
      break;
    case 'dictation':
      renderResult = renderDictationSection(appState.itemIndex);
      break;
    case 'fill':
      renderResult = renderFillSection(appState.itemIndex);
      break;
    case 'passage':
      renderResult = renderPassageSection();
      break;
    case 'voice':
      renderResult = renderVoiceSection();
      break;
    case 'story':
      renderResult = renderStorySection();
      break;
    default:
      renderResult = null;
      break;
  }

  const normalized = normalizeSectionRender(renderResult);
  if (normalized.node) {
    contentWrapper.appendChild(normalized.node);
  }

  container.appendChild(contentWrapper);
  dom.sectionContent.appendChild(container);

  if (withAnimation) {
    container.classList.add('card-fade');
  }

  const timerConfig =
    normalized.timerProfile ||
    section.timerProfile ||
    (section.timerSeconds
      ? { phases: [{ id: 'default', label: 'Time', duration: section.timerSeconds }] }
      : null);

  const shouldLockNext =
    normalized.lockNextUntilTimer || Boolean(section.lockNextDuringTimer);

  if (shouldLockNext) {
    dom.nextBtn.disabled = true;
  }

  if (timerConfig) {
    startTimer(timerConfig, {
      onPhaseStart: (phase, index) => {
        normalized.onPhaseStart?.(phase, index);
      },
      onTimerComplete: () => {
        normalized.onTimerComplete?.();
        if (shouldLockNext) {
          dom.nextBtn.disabled = false;
        }
      }
    });
  } else {
    stopTimer();
  }
}

function normalizeSectionRender(output) {
  if (!output) {
    return { node: null };
  }

  const isDomNode =
    typeof Node !== 'undefined' &&
    (output instanceof Node || (output.nodeType && typeof output.appendChild === 'function'));

  if (isDomNode) {
    return { node: output };
  }

  return {
    node: output.node ?? null,
    timerProfile: output.timerProfile,
    onPhaseStart: output.onPhaseStart,
    onTimerComplete: output.onTimerComplete,
    lockNextUntilTimer: output.lockNextUntilTimer
  };
}

/**
 * Repeat Section UI.
 */
function renderRepeatsSection(index) {
  const prompt = repeatPrompts[index];
  const response = appState.responses.repeats[index];

  const wrapper = document.createElement('div');
  wrapper.className = 'space-y-4';

  const audioContainer = document.createElement('div');
  audioContainer.className =
    'flex flex-wrap items-center gap-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-4';
  audioContainer.innerHTML = `
    <button
      class="play-audio inline-flex h-12 w-12 items-center justify-center rounded-full bg-emerald-500 text-slate-900 font-semibold shadow-lg hover:bg-emerald-400 transition"
      aria-label="Play audio clip"
    >
      ▶
    </button>
    <div class="text-sm text-slate-300 max-w-md">
      <p>The prompt plays automatically. Use the button if you need to hear it again.</p>
    </div>
  `;

  const playButton = audioContainer.querySelector('.play-audio');
  const audio = new Audio(prompt.audio);
  audio.preload = 'auto';

  // Create audio visualization
  const audioVisualization = createAudioVisualization(audioContainer);

  const safePlay = () => {
    audio.currentTime = 0;
    audio.play().catch((error) => {
      console.error('Audio playback failed', error);
    });
    // Start animation when audio plays
    animateAudioBars(audioVisualization, true);
  };

  // Stop animation when audio ends
  audio.addEventListener('ended', () => {
    animateAudioBars(audioVisualization, false);
  });

  playButton.addEventListener('click', safePlay);

  const transcriptHint = document.createElement('details');
  transcriptHint.className =
    'rounded-2xl border border-slate-800 bg-slate-900/40 p-4 text-sm text-slate-400';
  transcriptHint.innerHTML = `
    <summary class="cursor-pointer text-slate-300 font-medium">Need a hint?</summary>
    <p class="mt-2 text-xs text-slate-500">Original prompt: "${prompt.transcript}"</p>
  `;

  const recorder = renderSpeechRecorder({
    captureKey: `repeat-${index}`,
    placeholderText: 'Microphone ready. Wait for the tone before speaking.',
    savedTranscript: response.recognized,
    savedManual: response.manual,
    initialStatus: recognitionSupported
      ? 'Waiting for the prompt...'
      : 'Speech recognition unavailable - type your response.',
    autoRecord: recognitionSupported,
    locked: true,
    autoStartDelay: 150,
    onResult: (recognizedText, isManual = false) => {
      const target = appState.responses.repeats[index];
      if (isManual) {
        target.manual = recognizedText;
      } else {
        target.recognized = recognizedText;
        target.manual = '';
      }
      target.attempts += 1;
      appState.log.push({
        section: 'Repeats',
        promptId: prompt.id,
        text: recognizedText,
        timestamp: new Date().toISOString()
      });
      persistState();
    }
  });

  wrapper.appendChild(audioContainer);
  wrapper.appendChild(transcriptHint);
  wrapper.appendChild(recorder);

  const controller = recorder.__controller;

  return {
    node: wrapper,
    onPhaseStart: (phase) => {
      if (!phase) return;
      if (phase.id === 'listen') {
        controller?.lock?.();
        controller?.setStatus?.('Listening to the prompt...');
        safePlay();
      }
      if (phase.id === 'speak') {
        controller?.unlock?.();
        controller?.setStatus?.('Respond now');
        controller?.startAutoRecording?.(0);
      }
    },
    onTimerComplete: () => {
      controller?.setStatus?.('Time up – response saved');
      controller?.stopRecording?.();
      const initialSectionIndex = appState.sectionIndex;
      const initialItemIndex = appState.itemIndex;
      setTimeout(() => {
        if (
          appState.sectionIndex === initialSectionIndex &&
          appState.itemIndex === initialItemIndex
        ) {
          advanceStep(1);
        }
      }, 600);
    },
    timerProfile: null
  };
}

/**
 * Drag & drop jumbled sentences.
 */
function renderJumbledSection(index) {
  const prompt = jumbledPrompts[index];
  const response = appState.responses.jumbled[index];

  // Ensure arrangement is stored as fragment indexes for consistent rendering.
  let arrangementIndexes = Array.isArray(response.arrangement) ? response.arrangement : [];
  if (arrangementIndexes.some((item) => typeof item === 'string')) {
    arrangementIndexes = response.arrangement
      .map((text) => prompt.fragments.indexOf(text))
      .filter((idx) => idx !== -1);
    response.arrangement = arrangementIndexes;
  }

  const wrapper = document.createElement('div');
  wrapper.className = 'space-y-4';

  // Add audio visualization container
  const audioContainer = document.createElement('div');
  audioContainer.className = 'audio-container rounded-2xl border border-slate-800 bg-slate-900/60 p-4 mb-4';
  const audioVisualization = createAudioVisualization(audioContainer);
  
  // Simulate playing phrases
  const playPhrases = () => {
    animateAudioBars(audioVisualization, true);
    setTimeout(() => {
      animateAudioBars(audioVisualization, false);
      // Show beep and microphone after phrases
      setTimeout(() => {
        showMicrophone();
      }, 500);
    }, 3000);
  };
  
  const showMicrophone = () => {
    // Remove audio visualization
    audioContainer.remove();
    
    // Show microphone icon
    const micContainer = document.createElement('div');
    micContainer.className = 'flex flex-col items-center justify-center my-6';
    micContainer.innerHTML = `
      <div class="w-16 h-16 rounded-full bg-red-500 flex items-center justify-center mb-4">
        <span class="text-2xl">🎤</span>
      </div>
      <p class="text-sm text-slate-400">Recording... Speak now</p>
    `;
    wrapper.insertBefore(micContainer, wrapper.firstChild);
    
    // Auto advance after recording time
    setTimeout(() => {
      micContainer.remove();
      advanceStep(1);
    }, 3000);
  };
  
  wrapper.appendChild(audioContainer);
  
  // Start playing phrases automatically
  setTimeout(() => {
    playPhrases();
  }, 1000);

  const fragmentsContainer = document.createElement('div');
  fragmentsContainer.className =
    'flex flex-wrap gap-3 rounded-2xl border border-slate-800 bg-slate-900/60 p-4';

  const availableIndexes = prompt.fragments
    .map((_, idx) => idx)
    .filter((idx) => !arrangementIndexes.includes(idx));
  const pool = availableIndexes.length
    ? availableIndexes
    : prompt.fragments.map((_, idx) => idx);
  const fragmentOrder = shuffleArray(pool);

  fragmentOrder.forEach((fragmentIndex) => {
    const fragment = prompt.fragments[fragmentIndex];
    const button = document.createElement('button');
    button.className =
      'drag-fragment select-none rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-sm font-medium text-slate-200 hover:border-emerald-400 hover:text-emerald-300';
    button.draggable = true;
    button.textContent = fragment;
    button.dataset.fragmentIndex = fragmentIndex;

    button.addEventListener('dragstart', handleDragStart);
    button.addEventListener('dragend', handleDragEnd);
    fragmentsContainer.appendChild(button);
  });

  const dropZone = document.createElement('div');
  dropZone.className =
    'drop-target frosted-scroll flex min-h-[4rem] flex-wrap gap-2 rounded-2xl border border-dashed border-slate-700 bg-slate-900/40 p-4 text-sm text-slate-400';
  dropZone.textContent = arrangementIndexes.length
    ? ''
    : 'Drag the fragments here in the correct order';

  if (arrangementIndexes.length) {
    arrangementIndexes.forEach((fragmentIndex) => {
      const fragmentText = prompt.fragments[fragmentIndex];
      dropZone.appendChild(createPlacedPill(fragmentText, fragmentIndex));
    });
  }

  dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropZone.classList.add('active');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('active'));
  dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('active');
    const fragmentIndex = parseInt(event.dataTransfer.getData('fragment-index'), 10);
    if (Number.isNaN(fragmentIndex)) return;
    const fragmentText = prompt.fragments[fragmentIndex];

    if (dropZone.textContent.includes('Drag the fragments')) {
      dropZone.textContent = '';
    }

    const existing = dropZone.querySelector(`[data-fragment-index="${fragmentIndex}"]`);
    if (existing) return; // prevent duplicates

    dropZone.appendChild(createPlacedPill(fragmentText, fragmentIndex, true));

    const sourceButton = fragmentsContainer.querySelector(
      `[data-fragment-index="${fragmentIndex}"]`
    );
    if (sourceButton) {
      sourceButton.remove();
    }
  });

  const actions = document.createElement('div');
  actions.className = 'flex flex-wrap items-center gap-3';

  const submitBtn = document.createElement('button');
  submitBtn.className =
    'inline-flex items-center gap-2 rounded-full bg-indigo-500 px-4 py-2 text-sm font-semibold hover:bg-indigo-400 transition';
  submitBtn.textContent = 'Check Sentence';

  const resetBtn = document.createElement('button');
  resetBtn.className =
    'inline-flex items-center gap-2 rounded-full border border-slate-700 px-4 py-2 text-sm font-semibold hover:border-emerald-400 hover:text-emerald-300 transition';
  resetBtn.textContent = 'Reset';

  const feedback = document.createElement('div');
  feedback.className = 'text-sm text-slate-300';

  submitBtn.addEventListener('click', () => {
    const orderedIndexes = Array.from(dropZone.children)
      .map((node) => parseInt(node.dataset.fragmentIndex, 10))
      .filter((idx) => !Number.isNaN(idx));
    const orderedFragments = orderedIndexes.map((idx) => prompt.fragments[idx]);
    const sentence = orderedFragments.join(' ');
    const isCorrect = sentence.trim().toLowerCase() === prompt.correct.trim().toLowerCase();

    response.arrangement = orderedIndexes;
    response.correct = isCorrect;

    feedback.textContent = isCorrect
      ? '✅ Great! That matches the expected sentence.'
      : `⚠️ Not quite. Expected: "${prompt.correct}".`;
    feedback.className = `text-sm ${
      isCorrect ? 'text-emerald-300' : 'text-amber-300'
    }`;

    appState.log.push({
      section: 'Jumbled Sentence',
      promptId: prompt.id,
      text: sentence,
      correct: isCorrect,
      timestamp: new Date().toISOString()
    });
    persistState();
  });

  resetBtn.addEventListener('click', () => {
    response.arrangement = [];
    response.correct = null;
    persistState();
    renderCurrentSection();
  });

  actions.appendChild(submitBtn);
  actions.appendChild(resetBtn);
  actions.appendChild(feedback);

  wrapper.appendChild(fragmentsContainer);
  wrapper.appendChild(dropZone);
  wrapper.appendChild(actions);
  return { node: wrapper };

  function createPlacedPill(text, fragmentIndex, shouldPersist = false) {
    const pill = document.createElement('span');
    pill.className =
      'inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-sm text-slate-200';
    pill.dataset.fragmentIndex = fragmentIndex;
    pill.innerHTML = `<span>${text}</span>`;

    const removeBtn = document.createElement('button');
    removeBtn.className =
      'rounded-full bg-slate-800 px-2 text-xs text-slate-400 hover:text-rose-400';
    removeBtn.textContent = '✕';
    removeBtn.addEventListener('click', () => {
      pill.remove();
      response.arrangement = response.arrangement.filter((idx) => idx !== fragmentIndex);
      const restored = document.createElement('button');
      restored.className =
        'drag-fragment select-none rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-sm font-medium text-slate-200 hover:border-emerald-400 hover:text-emerald-300';
      restored.draggable = true;
      restored.textContent = text;
      restored.dataset.fragmentIndex = fragmentIndex;
      restored.addEventListener('dragstart', handleDragStart);
      restored.addEventListener('dragend', handleDragEnd);
      fragmentsContainer.appendChild(restored);
      if (!dropZone.children.length) {
        dropZone.textContent = 'Drag the fragments here in the correct order';
      }
      persistState();
    });

    pill.appendChild(removeBtn);

    if (shouldPersist && !response.arrangement.includes(fragmentIndex)) {
      response.arrangement.push(fragmentIndex);
      persistState();
    }

    return pill;
  }
}

/**
 * Render conversation section with short answer
 */
function renderConversationSection(index) {
  const prompt = conversationPrompts[index];
  const response = appState.responses.conversation[index];

  const wrapper = document.createElement('div');
  wrapper.className = 'space-y-4';

  // Add audio visualization container
  const audioContainer = document.createElement('div');
  audioContainer.className = 'audio-container rounded-2xl border border-slate-800 bg-slate-900/60 p-4 mb-4';
  const audioVisualization = createAudioVisualization(audioContainer);
  
  // Simulate playing conversation and question
  const playConversation = () => {
    animateAudioBars(audioVisualization, true);
    setTimeout(() => {
      animateAudioBars(audioVisualization, false);
      // Show beep and microphone after conversation
      setTimeout(() => {
        showMicrophone();
      }, 500);
    }, 5000);
  };
  
  const showMicrophone = () => {
    // Remove audio visualization
    audioContainer.remove();
    
    // Show microphone icon
    const micContainer = document.createElement('div');
    micContainer.className = 'flex flex-col items-center justify-center my-6';
    micContainer.innerHTML = `
      <div class="w-16 h-16 rounded-full bg-red-500 flex items-center justify-center mb-4">
        <span class="text-2xl">🎤</span>
      </div>
      <p class="text-sm text-slate-400">Recording... Speak now</p>
    `;
    wrapper.insertBefore(micContainer, wrapper.firstChild);
    
    // Auto advance after recording time
    setTimeout(() => {
      micContainer.remove();
      advanceStep(1);
    }, 3000);
  };
  
  wrapper.appendChild(audioContainer);
  
  // Start playing conversation automatically
  setTimeout(() => {
    playConversation();
  }, 1000);
  
  // Show the question
  const questionContainer = document.createElement('div');
  questionContainer.className = 'text-center py-4';
  questionContainer.innerHTML = `
    <p class="text-lg font-medium text-slate-200">${prompt.question}</p>
    <p class="text-sm text-slate-400 mt-2">Provide a brief answer</p>
  `;
  wrapper.appendChild(questionContainer);

  return { node: wrapper };
}

/**
 * Render dictation section
 */
function renderDictationSection(index) {
  const prompt = dictationPrompts[index];
  const response = appState.responses.dictation[index];

  const wrapper = document.createElement('div');
  wrapper.className = 'space-y-4';

  // Add audio visualization container
  const audioContainer = document.createElement('div');
  audioContainer.className = 'audio-container rounded-2xl border border-slate-800 bg-slate-900/60 p-4 mb-4';
  const audioVisualization = createAudioVisualization(audioContainer);
  
  // Simulate playing sentence
  const playSentence = () => {
    animateAudioBars(audioVisualization, true);
    setTimeout(() => {
      animateAudioBars(audioVisualization, false);
      // Show text box after sentence
      setTimeout(() => {
        showTextBox();
      }, 500);
    }, 3000);
  };
  
  const showTextBox = () => {
    // Remove audio visualization
    audioContainer.remove();
    
    // Show text box
    const textBoxContainer = document.createElement('div');
    textBoxContainer.className = 'flex flex-col items-center justify-center my-6';
    
    const textArea = document.createElement('textarea');
    textArea.className = 'w-full max-w-2xl h-32 border border-slate-700 rounded-lg p-4 text-base bg-slate-900 text-slate-100';
    textArea.placeholder = 'Type exactly what you heard...';
    textArea.value = response.typed;
    
    const timerText = document.createElement('p');
    timerText.className = 'text-sm text-slate-400 mt-2';
    timerText.textContent = '25';
    
    textBoxContainer.appendChild(textArea);
    textBoxContainer.appendChild(timerText);
    wrapper.insertBefore(textBoxContainer, wrapper.firstChild);
    
    // Focus on text area
    textArea.focus();
    
    // Start countdown timer
    let timeLeft = 25;
    const countdown = setInterval(() => {
      timeLeft--;
      timerText.textContent = timeLeft;
      if (timeLeft <= 0) {
        clearInterval(countdown);
        response.typed = textArea.value;
        persistState();
        // Auto advance when time runs out
        setTimeout(() => {
          advanceStep(1);
        }, 500);
      }
    }, 1000);
    
    // Also advance when user presses Enter
    textArea.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        clearInterval(countdown);
        response.typed = textArea.value;
        persistState();
        advanceStep(1);
      }
    });
    
    // Update response as user types
    textArea.addEventListener('input', () => {
      response.typed = textArea.value;
      persistState();
    });
  };
  
  wrapper.appendChild(audioContainer);
  
  // Start playing sentence automatically
  setTimeout(() => {
    playSentence();
  }, 1000);

  return { node: wrapper };
}

/**
 * Render passage reconstruction section
 */
function renderPassageSection() {
  const response = appState.responses.passage;

  const wrapper = document.createElement('div');
  wrapper.className = 'space-y-4';

  // Show passage for 30 seconds
  const passageContainer = document.createElement('div');
  passageContainer.className = 'passage-container rounded-2xl border border-slate-800 bg-slate-900/60 p-6 mb-4';
  
  const passageText = document.createElement('p');
  passageText.className = 'text-lg text-slate-200 leading-relaxed';
  passageText.textContent = passagePrompt.text;
  
  const timerText = document.createElement('p');
  timerText.className = 'text-sm text-slate-400 mt-4 text-center';
  timerText.textContent = '30';
  
  passageContainer.appendChild(passageText);
  passageContainer.appendChild(timerText);
  wrapper.appendChild(passageContainer);
  
  // Start countdown timer
  let timeLeft = 30;
  const countdown = setInterval(() => {
    timeLeft--;
    timerText.textContent = timeLeft;
    if (timeLeft <= 0) {
      clearInterval(countdown);
      // Hide passage and show text box
      showTextBox();
    }
  }, 1000);
  
  const showTextBox = () => {
    // Remove passage container
    passageContainer.remove();
    
    // Show text box
    const textBoxContainer = document.createElement('div');
    textBoxContainer.className = 'flex flex-col items-center justify-center my-6';
    
    const textArea = document.createElement('textarea');
    textArea.className = 'w-full max-w-3xl h-64 border border-slate-700 rounded-lg p-4 text-base bg-slate-900 text-slate-100';
    textArea.placeholder = 'Type as much as you can remember from the passage...';
    textArea.value = response.typed;
    
    const timerText = document.createElement('p');
    timerText.className = 'text-sm text-slate-400 mt-2';
    timerText.textContent = '90';
    
    textBoxContainer.appendChild(textArea);
    textBoxContainer.appendChild(timerText);
    wrapper.appendChild(textBoxContainer);
    
    // Focus on text area
    textArea.focus();
    
    // Start writing timer
    let writeTimeLeft = 90;
    const writeCountdown = setInterval(() => {
      writeTimeLeft--;
      timerText.textContent = writeTimeLeft;
      if (writeTimeLeft <= 0) {
        clearInterval(writeCountdown);
        response.typed = textArea.value;
        response.timestamp = new Date().toISOString();
        persistState();
        // Auto advance when time runs out
        setTimeout(() => {
          advanceStep(1);
        }, 500);
      }
    }, 1000);
    
    // Update response as user types
    textArea.addEventListener('input', () => {
      response.typed = textArea.value;
      persistState();
    });
  };

  return { node: wrapper };
}

function handleDragStart(event) {
  event.dataTransfer.setData('fragment-index', event.target.dataset.fragmentIndex);
  requestAnimationFrame(() => event.target.classList.add('opacity-50'));
}

function handleDragEnd(event) {
  event.target.classList.remove('opacity-50');
}

/**
 * Fill in the blank rendering.
 */
function renderFillSection(index) {
  const prompt = fillPrompts[index];
  const response = appState.responses.fill[index];

  const wrapper = document.createElement('div');
  wrapper.className =
    'space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-5';

  const sentenceEl = document.createElement('p');
  sentenceEl.className = 'text-base text-slate-200';
  sentenceEl.innerHTML = prompt.sentence.replace(
    '____',
    '<span class="text-indigo-300 font-semibold">_____</span>'
  );

  wrapper.appendChild(sentenceEl);

  let inputElement;

  if (prompt.type === 'choice') {
    const optionsWrapper = document.createElement('div');
    optionsWrapper.className = 'flex flex-wrap gap-3';

    prompt.options.forEach((option) => {
      const optionBtn = document.createElement('button');
      optionBtn.className =
        'rounded-full border border-slate-700 px-4 py-2 text-sm font-medium text-slate-200 hover:border-emerald-400 hover:text-emerald-300 transition';
      optionBtn.textContent = option;
      optionBtn.addEventListener('click', () => {
        evaluateFillResponse(option);
      });
      optionsWrapper.appendChild(optionBtn);
    });

    inputElement = optionsWrapper;
  } else {
    const input = document.createElement('input');
    input.type = 'text';
    input.value = response.response;
    input.placeholder = 'Type the missing word...';
    input.className =
      'w-full rounded-2xl border border-slate-800 bg-slate-900/60 px-4 py-3 text-sm text-slate-100 placeholder:text-slate-600 focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-400/20';
    input.addEventListener('input', (event) => {
      response.response = event.target.value;
    });
    input.addEventListener('change', () => {
      evaluateFillResponse(input.value);
    });
    inputElement = input;
  }

  const feedback = document.createElement('p');
  feedback.className = 'text-sm text-slate-400';
  if (response.correct !== null) {
    feedback.textContent = response.correct
      ? '✅ Correct!'
      : `⚠️ Correct answer: "${prompt.answer}".`;
    feedback.className = `text-sm ${response.correct ? 'text-emerald-300' : 'text-amber-300'}`;
  }

  function evaluateFillResponse(value) {
    const normalized = value.trim().toLowerCase();
    const isCorrect = normalized === prompt.answer.trim().toLowerCase();
    response.response = value;
    response.correct = isCorrect;
    feedback.textContent = isCorrect
      ? '✅ Correct!'
      : `⚠️ Correct answer: "${prompt.answer}".`;
    feedback.className = `text-sm ${isCorrect ? 'text-emerald-300' : 'text-amber-300'}`;

    appState.log.push({
      section: 'Fill in the Blank',
      promptId: prompt.id,
      text: value,
      correct: isCorrect,
      timestamp: new Date().toISOString()
    });
    persistState();
  }

  wrapper.appendChild(inputElement);
  wrapper.appendChild(feedback);
  return { node: wrapper };
}

/**
 * Voice to text free-form recording.
 */
function renderVoiceSection() {
  const response = appState.responses.voice;

  const wrapper = document.createElement('div');
  wrapper.className = 'space-y-4';

  const promptCard = document.createElement('div');
  promptCard.className =
    'rounded-2xl border border-slate-800 bg-slate-900/60 p-5 text-sm text-slate-300';
  promptCard.textContent = voicePrompt;

  const recorder = renderSpeechRecorder({
    captureKey: 'voice',
    placeholderText: 'Microphone ready. Begin speaking when prompted.',
    savedTranscript: response.transcript,
    savedManual: response.manual || '',
    allowContinuous: true,
    initialStatus: recognitionSupported
      ? 'Preparing...'
      : 'Speech recognition unavailable - type your response.',
    autoRecord: recognitionSupported,
    locked: true,
    onResult: (recognizedText, isManual = false) => {
      if (isManual) {
        response.manual = recognizedText;
        response.transcript = '';
      } else {
        response.transcript = recognizedText;
        response.manual = '';
      }
      response.timestamp = new Date().toISOString();
      appState.log.push({
        section: 'Voice to Text',
        text: recognizedText,
        timestamp: response.timestamp
      });
      persistState();
    }
  });

  wrapper.appendChild(promptCard);
  wrapper.appendChild(recorder);

  const controller = recorder.__controller;

  return {
    node: wrapper,
    onPhaseStart: (phase) => {
      if (!phase) return;
      if (phase.id === 'prep') {
        controller?.lock?.();
        controller?.setStatus?.('Get ready...');
      }
      if (phase.id === 'speak') {
        controller?.unlock?.();
        controller?.setStatus?.('Speak now');
        controller?.startAutoRecording?.(0);
      }
    },
    onTimerComplete: () => {
      controller?.setStatus?.('Time up – response saved');
      controller?.stopRecording?.();
    },
    timerProfile: null
  };
}

/**
 * Story retelling section.
 */
function renderStorySection() {
  const response = appState.responses.story;

  const wrapper = document.createElement('div');
  wrapper.className = 'space-y-4';

  const storyCard = document.createElement('div');
  storyCard.className =
    'space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-5 text-sm text-slate-300';

  storyCard.innerHTML = `
    <p>Listen to the story. As soon as it finishes, you’ll have a brief preparation window before speaking.</p>
    <details class="rounded-2xl border border-slate-800 bg-slate-950/40 p-4 text-xs text-slate-500">
      <summary class="cursor-pointer text-slate-300 font-medium">Show transcript (for accessibility)</summary>
      <p class="mt-2">${storyPrompt.transcript}</p>
    </details>
  `;

  const audioBar = document.createElement('div');
  audioBar.className =
    'flex flex-wrap items-center gap-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-4';
  audioBar.innerHTML = `
    <button
      class="play-story inline-flex h-12 w-12 items-center justify-center rounded-full bg-emerald-500 text-slate-900 font-semibold shadow-lg hover:bg-emerald-400 transition"
      aria-label="Play story audio"
    >
      ▶
    </button>
    <div class="text-sm text-slate-300 max-w-md">
      <p>The story plays automatically. Use the button to replay if needed.</p>
    </div>
  `;

  const playButton = audioBar.querySelector('.play-story');
  const audio = new Audio(storyPrompt.audio);
  audio.preload = 'auto';

  const safePlay = () => {
    audio.currentTime = 0;
    audio.play().catch((error) => console.error('Story playback failed', error));
  };

  playButton.addEventListener('click', safePlay);

  const recorderWrapper = renderSpeechRecorder({
    captureKey: 'story',
    placeholderText: 'Microphone ready. Retell the story when prompted.',
    savedTranscript: response.transcript,
    savedManual: response.manual || '',
    initialStatus: recognitionSupported
      ? 'Listening for the story...'
      : 'Speech recognition unavailable - type your response.',
    autoRecord: recognitionSupported,
    locked: true,
    onResult: (recognizedText, isManual = false) => {
      if (isManual) {
        response.manual = recognizedText;
        response.transcript = '';
      } else {
        response.transcript = recognizedText;
        response.manual = '';
      }
      response.timestamp = new Date().toISOString();
      appState.log.push({
        section: 'Story Retelling',
        text: recognizedText,
        timestamp: response.timestamp
      });
      persistState();
    }
  });

  wrapper.appendChild(storyCard);
  wrapper.appendChild(audioBar);
  wrapper.appendChild(recorderWrapper);

  const controller = recorderWrapper.__controller;
  let storyFinished = false;
  let pendingStart = false;

  audio.addEventListener('ended', () => {
    storyFinished = true;
    controller?.setStatus?.('Prepare to speak...');
    if (pendingStart) {
      pendingStart = false;
      controller?.unlock?.();
      controller?.startAutoRecording?.(0);
    }
  });

  return {
    node: wrapper,
    onPhaseStart: (phase) => {
      if (!phase) return;
      if (phase.id === 'listen') {
        storyFinished = false;
        pendingStart = false;
        controller?.lock?.();
        controller?.setStatus?.('Listening to the story...');
        safePlay();
      }
      if (phase.id === 'prep') {
        controller?.lock?.();
        controller?.setStatus?.('Prepare to retell...');
      }
      if (phase.id === 'speak') {
        controller?.setStatus?.('Retell the story now');
        if (storyFinished) {
          controller?.unlock?.();
          controller?.startAutoRecording?.(0);
        } else {
          pendingStart = true;
        }
      }
    },
    onTimerComplete: () => {
      controller?.setStatus?.('Time up – response saved');
      controller?.stopRecording?.();
    },
    timerProfile: null
  };
}

/**
 * Speech recorder with microphone + manual fallback.
 */
function renderSpeechRecorder({
  captureKey,
  placeholderText,
  savedTranscript,
  savedManual,
  onResult,
  allowContinuous = false,
  autoRecord = false,
  autoStartDelay = 0,
  locked = false,
  initialStatus = '',
  manualPlaceholder = 'Type your response here if your microphone is unavailable.'
}) {
  const wrapper = document.createElement('div');
  wrapper.className =
    'space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-5';

  const status = document.createElement('div');
  status.className = 'text-xs uppercase tracking-[0.3em] text-slate-500';
  status.textContent = initialStatus
    ? initialStatus
    : recognitionSupported
    ? 'Microphone ready'
    : 'Speech recognition unavailable - manual input only';

  wrapper.appendChild(status);

  const useManualInput = !recognitionSupported;
  let manualField = null;

  if (useManualInput) {
    const textarea = document.createElement('textarea');
    textarea.className =
      'min-h-[140px] w-full rounded-2xl border border-slate-800 bg-slate-950/60 px-4 py-3 text-sm text-slate-100 placeholder:text-slate-600 focus:border-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-400/20';
    textarea.placeholder = manualPlaceholder;
    textarea.value = savedManual || savedTranscript || '';
    wrapper.appendChild(textarea);
    manualField = textarea;
  } else {
    const info = document.createElement('p');
    info.className = 'text-sm text-slate-400';
    info.textContent =
      placeholderText || 'Microphone ready. Your response will be captured automatically.';
    wrapper.appendChild(info);
  }

  const helper = document.createElement('p');
  helper.className = 'text-xs text-slate-500';
  helper.textContent = recognitionSupported
    ? 'Recording starts automatically when the timer opens.'
    : 'Type to capture your response.';
  wrapper.appendChild(helper);

  let accumulatedTranscript = savedTranscript || savedManual || '';
  let manualMode = useManualInput;
  let lockedState = locked;
  let scheduledStart = null;
  let pendingAuto = recognitionSupported ? autoRecord : false;
  let requestedDelay = autoStartDelay;

  const updateStatus = (text) => {
    if (text) {
      status.textContent = text;
    }
  };

  const clearScheduledStart = () => {
    if (scheduledStart) {
      clearTimeout(scheduledStart);
      scheduledStart = null;
    }
  };

  const triggerAutoStart = (delay = autoStartDelay) => {
    if (!recognitionSupported) return;
    requestedDelay = delay;

    if (lockedState) {
      pendingAuto = true;
      return;
    }

    pendingAuto = false;
    clearScheduledStart();
    const ms = Math.max(delay, 0);
    scheduledStart = setTimeout(() => {
      scheduledStart = null;
      startRecognition({
        captureKey,
        allowContinuous,
        onResult: (text) => {
          accumulatedTranscript = text;
          if (useManualInput && manualField) {
            manualField.value = text;
          }
          manualMode = false;
          onResult?.(accumulatedTranscript);
        },
        onStart: () => {
          updateStatus('Recording...');
        },
        onEnd: () => {
          updateStatus(
            accumulatedTranscript ? 'Response captured' : 'Recording finished'
          );
        },
        onError: (error) => {
          console.warn('Speech recognition error', error);
          updateStatus(`Error: ${error.error || error.message || 'microphone issue'}`);
        }
      });
    }, ms);
  };

  const stopCurrentRecording = () => {
    clearScheduledStart();
    pendingAuto = false;
    if (recognitionMode === captureKey) {
      stopRecognition();
    }
  };

  if (useManualInput && manualField) {
    manualField.addEventListener('input', () => {
      stopCurrentRecording();
      manualMode = true;
      accumulatedTranscript = manualField.value;
      onResult?.(accumulatedTranscript, true);
      updateStatus('Manual entry captured');
    });
  }

  if (pendingAuto && !lockedState) {
    triggerAutoStart(autoStartDelay);
  }

  wrapper.__controller = {
    startAutoRecording: (delay) => {
      if (!recognitionSupported) return;
      triggerAutoStart(typeof delay === 'number' ? delay : autoStartDelay);
    },
    stopRecording: () => {
      stopCurrentRecording();
    },
    lock: () => {
      lockedState = true;
      clearScheduledStart();
    },
    unlock: () => {
      lockedState = false;
      if (pendingAuto) {
        triggerAutoStart(requestedDelay);
      }
    },
    setStatus: (text) => {
      updateStatus(text);
    },
    isLocked: () => lockedState
  };

  return wrapper;
}

/**
 * Create audio visualization bars
 */
function createAudioVisualization(container) {
  const visualization = document.createElement('div');
  visualization.className = 'audio-visualization flex items-end justify-center gap-1 h-16 my-4';
  
  // Create 20 bars for visualization
  for (let i = 0; i < 20; i++) {
    const bar = document.createElement('div');
    bar.className = 'audio-bar w-2 bg-blue-500 rounded-t transition-all duration-100';
    bar.style.height = '4px';
    visualization.appendChild(bar);
  }
  
  container.appendChild(visualization);
  return visualization;
}

/**
 * Animate audio visualization bars
 */
function animateAudioBars(visualization, isPlaying = true) {
  if (!visualization) return;
  
  const bars = visualization.querySelectorAll('.audio-bar');
  if (!bars.length) return;
  
  if (isPlaying) {
    // Animate bars with random heights
    const animate = () => {
      bars.forEach(bar => {
        const height = Math.floor(Math.random() * 20) + 4;
        bar.style.height = `${height}px`;
      });
      
      if (visualization.dataset.animating === 'true') {
        requestAnimationFrame(animate);
      }
    };
    
    visualization.dataset.animating = 'true';
    animate();
  } else {
    // Stop animation and reset bars
    visualization.dataset.animating = 'false';
    bars.forEach(bar => {
      bar.style.height = '4px';
    });
  }
}

/**
 * Start speech recognition session.
 */
function startRecognition({ captureKey, allowContinuous, onResult, onStart, onEnd, onError }) {
  try {
    stopRecognition();
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      throw new Error('Speech recognition unavailable');
    }

    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.continuous = allowContinuous;

    let finalTranscript = '';

    recognition.onstart = () => {
      activeRecognition = recognition;
      recognitionMode = captureKey;
      onStart?.();
    };

    recognition.onerror = (event) => {
      onError?.(event);
    };

    recognition.onresult = (event) => {
      let interimTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        if (result.isFinal) {
          finalTranscript += result[0].transcript;
        } else {
          interimTranscript += result[0].transcript;
        }
      }
      const aggregate = allowContinuous ? finalTranscript + interimTranscript : finalTranscript || interimTranscript;
      onResult?.(aggregate.trim());
    };

    recognition.onend = () => {
      onEnd?.();
      activeRecognition = null;
      recognitionMode = null;
    };

    recognition.start();
  } catch (error) {
    console.error('Failed to start recognition', error);
    onError?.(error);
  }
}

function stopRecognition() {
  if (activeRecognition) {
    activeRecognition.stop();
    activeRecognition = null;
    recognitionMode = null;
  }
}

/**
 * Countdown timer per section.
 */
function startTimer(timerConfig, hooks = {}) {
  stopTimer();

  let phases = [];
  if (typeof timerConfig === 'number') {
    phases = [{ id: 'default', label: 'Time', duration: timerConfig }];
  } else if (Array.isArray(timerConfig?.phases)) {
    phases = timerConfig.phases;
  } else if (timerConfig && typeof timerConfig === 'object' && 'duration' in timerConfig) {
    phases = [{ id: timerConfig.id || 'default', label: timerConfig.label || 'Time', duration: timerConfig.duration }];
  }

  currentTimerPhases = phases
    .map((phase, index) => ({
      id: phase.id || `phase-${index}`,
      label: phase.label || 'Time',
      duration: typeof phase.duration === 'number' ? phase.duration : 0
    }))
    .filter((phase) => phase.duration >= 0);

  if (!currentTimerPhases.length) {
    dom.timerBadge.classList.add('hidden');
    return;
  }

  currentTimerHooks = hooks;
  currentPhaseIndex = -1;

  dom.timerBadge.classList.remove('hidden');

  const advancePhase = () => {
    currentPhaseIndex += 1;

    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }

    if (currentPhaseIndex >= currentTimerPhases.length) {
      stopTimer(true);
      return;
    }

    const phase = currentTimerPhases[currentPhaseIndex];
    const phaseEnd = Date.now() + phase.duration * 1000;

    hooks.onPhaseStart?.(phase, currentPhaseIndex);

    if (phase.duration <= 0) {
      dom.timerBadge.textContent = `${phase.label} 00:00`;
      advancePhase();
      return;
    }

    const tick = () => {
      const remaining = Math.max(phaseEnd - Date.now(), 0);
      dom.timerBadge.textContent = `${phase.label} ${formatTimerValue(remaining)}`;
      if (remaining <= 0) {
        clearInterval(timerInterval);
        timerInterval = null;
        advancePhase();
      }
    };

    tick();
    timerInterval = setInterval(tick, 200);
  };

  advancePhase();
}

function stopTimer(expired = false) {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }

  if (expired) {
    dom.timerBadge.textContent = 'Time up';
    dom.timerBadge.classList.remove('hidden');
    currentTimerHooks?.onTimerComplete?.();
  } else {
    dom.timerBadge.classList.add('hidden');
  }

  currentTimerHooks = null;
  currentTimerPhases = [];
  currentPhaseIndex = -1;
}

function formatTimerValue(milliseconds) {
  const totalSeconds = Math.ceil(milliseconds / 1000);
  const minutes = Math.floor(totalSeconds / 60)
    .toString()
    .padStart(2, '0');
  const seconds = Math.max(totalSeconds % 60, 0)
    .toString()
    .padStart(2, '0');
  return `${minutes}:${seconds}`;
}

/**
 * Summary rendering after completion.
 */
function renderSummary() {
  dom.startScreen.classList.add('hidden');
  dom.testContainer.classList.add('hidden');
  dom.summaryScreen.classList.remove('hidden');
  updateProgressUI();
  calculateScores();
  populateSummary();
}

function calculateScores() {
  if (appState.scored) return;

  const repeatAccuracy =
    appState.responses.repeats.filter((item) => item.recognized || item.manual).length /
    repeatPrompts.length;
  const jumbledAccuracy =
    appState.responses.jumbled.filter((item) => item.correct).length / jumbledPrompts.length;
  const fillAccuracy =
    appState.responses.fill.filter((item) => item.correct).length / fillPrompts.length;

  appState.scores = {
    fluency: Math.round(60 + repeatAccuracy * 35),
    accuracy: Math.round(55 + fillAccuracy * 40),
    comprehension: Math.round(50 + jumbledAccuracy * 45),
    completeness: appState.responses.story.transcript || appState.responses.story.manual ? 85 : 60
  };

  appState.scored = true;
  persistState();
}

function populateSummary() {
  dom.summaryMetrics.innerHTML = '';
  Object.entries(appState.scores).forEach(([label, value]) => {
    const card = document.createElement('div');
    card.className =
      'rounded-2xl border border-slate-800 bg-slate-900/60 p-4 shadow-inner shadow-slate-900/40';
    card.innerHTML = `
      <p class="text-xs uppercase tracking-[0.3em] text-slate-500">${label}</p>
      <p class="text-2xl font-semibold text-slate-100">${value}</p>
    `;
    dom.summaryMetrics.appendChild(card);
  });

  dom.summaryLog.innerHTML = '';
  const logList = document.createElement('div');
  logList.className = 'space-y-2 text-sm text-slate-300';

  appState.log
    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
    .forEach((entry) => {
      const item = document.createElement('div');
      item.className =
        'rounded-2xl border border-slate-800 bg-slate-950/60 px-4 py-3';
      item.innerHTML = `
        <p class="text-xs uppercase tracking-[0.25em] text-slate-500 mb-1">${entry.section}</p>
        <p class="text-slate-200">${entry.text || '(no transcript captured)'}</p>
        <p class="text-[10px] uppercase tracking-[0.3em] text-slate-600 mt-2">${new Date(
          entry.timestamp
        ).toLocaleTimeString()}</p>
      `;
      logList.appendChild(item);
    });

  dom.summaryLog.appendChild(logList);
}

function resetPractice() {
  stopRecognition();
  stopTimer();
  appState = initialState();
  persistState();
  updateProgressUI();
  showStartScreen();
  dom.summaryScreen.classList.add('hidden');
}

/**
 * Utility helpers
 */
function shuffleArray(array) {
  const clone = [...array];
  for (let i = clone.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [clone[i], clone[j]] = [clone[j], clone[i]];
  }
  return clone;
}

window.addEventListener('beforeunload', () => {
  stopRecognition();
  stopTimer();
});

