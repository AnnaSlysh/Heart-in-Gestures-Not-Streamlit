// ─── Labels & letter mapping ───────────────────────────────────
const LABELS = ['V','Y','R','A','YA','N','I','T','U','P','G','E','Z','L','M','O','C','F','SH','YU','X','CH','B'];

const LETTER_MAP = {
    'V':'В','Y':'У','R':'Р','A':'А','YA':'Я','N':'Н','I':'І','T':'Т',
    'U':'И','P':'П','G':'Г','E':'Е','Z':'Ж','L':'Л','M':'М','O':'О',
    'C':'С','F':'Ф','SH':'Ш','YU':'Ю','X':'Х','CH':'Ч','B':'Б'
};

// ─── Word lists ───────────────────────────────────────────────
const WORDS = {
    easy: ['ЛАМПА','МЕТА','СИЛА','ЛИСТ','ТЕПЛО','ПАН','СЕЛО','МАТИ','ПОЛЕ','САЛО','ЛОТО','ТОН','СТАН','СМОЛА','ЛИПА','СИН','НАСИП','ЛОТОС','КІТ','КОТ','КАЗКА','ЗИМА','НОГА','КИТ','ДІМ','ЛЕД'],
    medium: ['МІСТО','ІСПИТ','РОБОТА','МОТИВ','НЕБО','МІСТ','ВИСОТА','СУМА','ПЕРО','ТІСТО','СТІЛ','ВІТЕР','ТУМАН','ВЕЧІР','ПОБУТ','ЛІТР','СТОВП','БЕТОН','КОЗАК','ДЕРЕВО','ЗЕРНО','КОБРА','ДВЕРІ','ЗІРКА','ЦИРК'],
    hard: ['УСПІХ','ГУМОР','ШИЯ','ЮРИСТ','СИМВОЛ','ФАХ','СПАЛАХ','ІНЖЕНЕР','ЛЮБОВ','ПЕЧИВО','ЛИСТЯ','ФОРМА','ГОРА','ХВІСТ','ФАНЕРА','ШТАНИ','СТРУМ','ДЕРЖАВА','ЄДНІСТЬ','ЦІННІСТЬ','ЩЕДРІСТЬ','ДРУЖБА','КУЛЬТУРА'],
};

// Flower images per level: index 0 = full flower, last = empty
const FLOWER_IMAGES = {
    easy:   ['easy_0','easy_1','easy_2'],
    medium: ['medium_0','medium_1','medium_2','medium_3','medium_4'],
    hard:   ['hard_0','hard_1','hard_2','hard_3','hard_4','hard_5','hard_6','hard_7','hard_8','hard_9'],
};
const WIN_IMAGES  = { easy: 'easywinnn', medium: 'midwinnn', hard: 'hardwinnn' };

// hold ~2 s at ~30 fps
const HOLD_FRAMES = 60;

// ─── Model state ──────────────────────────────────────────────
let mlpLayers = null;   // [{type:'dense', weights, bias}, ..., {type:'softmax'}]

// ─── Game state ───────────────────────────────────────────────
let gameState = null;
let handsDetector = null;
let cameraStream = null;
let animFrameId = null;
let videoEl = null;
let canvasEl = null;
let canvasCtx = null;

// ─── Pure-JS MLP inference ────────────────────────────────────

function preprocessLandmarks(landmarks) {
    // mirrors Python pre_process_landmark
    const bx = landmarks[0].x, by = landmarks[0].y;
    const rel = landmarks.map(lm => [lm.x - bx, lm.y - by]);
    const flat = rel.flat();
    const maxVal = Math.max(...flat.map(Math.abs));
    return maxVal === 0 ? flat : flat.map(v => v / maxVal);
}

function relu(x) { return x > 0 ? x : 0; }

function softmax(arr) {
    const m = Math.max(...arr);
    const e = arr.map(v => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
}

function mlpInfer(input) {
    if (!mlpLayers) return null;
    let x = input;
    const nDense = mlpLayers.filter(l => l.type === 'dense').length;
    let densesSeen = 0;
    for (const layer of mlpLayers) {
        if (layer.type === 'dense') {
            densesSeen++;
            const W = layer.weights;   // [out][in]
            const b = layer.bias;
            const out = new Array(W.length);
            for (let i = 0; i < W.length; i++) {
                let sum = b ? b[i] : 0;
                const wi = W[i];
                for (let j = 0; j < x.length; j++) sum += wi[j] * x[j];
                // apply relu for all but the last dense layer
                out[i] = densesSeen < nDense ? relu(sum) : sum;
            }
            x = out;
        } else if (layer.type === 'softmax') {
            x = softmax(x);
        }
    }
    return x;
}

function classifyLandmarks(landmarks) {
    if (!mlpLayers) return null;
    const input = preprocessLandmarks(landmarks);
    const probs = mlpInfer(input);
    if (!probs) return null;
    const classId = probs.indexOf(Math.max(...probs));
    return LABELS[classId] || null;
}

// ─── Game rendering ───────────────────────────────────────────

function renderGame() {
    const container = document.getElementById('page-game');
    if (!container) return;
    const lang = window.currentLang || 'uk';
    const t = window.texts[lang];

    if (!gameState) {
        container.innerHTML = `
            <div class="page-hero">
                <div class="title_header">${t.game_title}</div>
                <p>${t.game_tagline}</p>
            </div>
            ${mlpLayers ? '' : `<div class="camera-note">${t.model_loading}</div>`}
            <div class="level-grid">
                ${['easy','medium','hard'].map((lvl, i) => `
                <div class="level-card" onclick="startGame('${lvl}')">
                    <div class="level-name">${t.levels[i].name}</div>
                    <div class="level-desc">${t.levels[i].desc}</div>
                    <button class="btn btn-full">${t.play_btn}</button>
                </div>`).join('')}
            </div>`;
        return;
    }

    const { word, letterStates, currentIndex, won, wrongLetter, level, flowerIndex } = gameState;
    const levelTitle = { easy: t.levels[0].name, medium: t.levels[1].name, hard: t.levels[2].name }[level];
    const flowerImages = FLOWER_IMAGES[level];
    const flowerImg = flowerImages[Math.min(flowerIndex, flowerImages.length - 1)];

    const wordHtml = word.split('').map((ch, i) => {
        let cls = 'word-letter';
        if (letterStates[i] === 'correct') cls += ' correct';
        else if (i === currentIndex) cls += ' current';
        return `<div class="${cls}">${letterStates[i] === 'correct' ? ch : '_'}</div>`;
    }).join('');

    const gameBody = won
        ? `<div class="win-area">
               <img src="images/${WIN_IMAGES[level]}.svg" alt="win">
               <div class="win-message">${t.win_msg} 🎉</div>
           </div>`
        : `<div class="game-area">
               <div>
                   <div class="camera-note">${t.camera_note}</div>
                   <div class="camera-container">
                       <canvas id="game-canvas"></canvas>
                       <div class="camera-label" id="camera-label">${t.camera_label}</div>
                   </div>
               </div>
               <div class="game-info">
                   <div class="game-stat">
                       <div class="game-stat-label">${t.show_gesture}</div>
                       <div class="game-stat-value" id="current-letter-display" style="font-size:56px">${currentIndex < word.length ? word[currentIndex] : ''}</div>
                   </div>
                   <div class="game-stat">
                       <div class="game-stat-label">${t.detected_label}</div>
                       <div class="game-stat-value" id="detected-letter" style="font-size:56px">—</div>
                       <div class="progress-bar-wrap"><div class="progress-bar-fill" id="hold-progress"></div></div>
                   </div>
                   ${wrongLetter ? `
                   <div class="game-stat">
                       <div class="game-stat-label">${t.wrong_label}</div>
                       <div class="game-stat-value wrong">${wrongLetter}</div>
                   </div>` : ''}
                   <img src="images/${flowerImg}.svg" id="flower-img" style="width:110px;margin:8px auto 0;display:block;" alt="flower">
               </div>
           </div>`;

    container.innerHTML = `
        <div class="title_subheader">${levelTitle}</div>
        <div class="word-display" id="word-display">${wordHtml}</div>
        <div class="divider"></div>
        ${gameBody}
        <div style="margin-top:16px;">
            <button class="btn btn-full" onclick="backToMenu()">${t.back_btn}</button>
        </div>`;

    if (!won) {
        canvasEl = document.getElementById('game-canvas');
        if (canvasEl && !animFrameId) startCamera();
    }
}

// ─── Game logic ───────────────────────────────────────────────

function startGame(level) {
    const wordList = WORDS[level];
    const word = wordList[Math.floor(Math.random() * wordList.length)];
    gameState = {
        level, word,
        letterStates: Array(word.length).fill('pending'),
        currentIndex: 0,
        won: false,
        wrongLetter: null,
        flowerIndex: 0,
        holdCounter: 0,
        lastDetected: null,
    };
    renderGame();
}

function backToMenu() {
    stopCamera();
    gameState = null;
    renderGame();
}

function processLetter(letter) {
    if (!gameState) return;
    const { word, letterStates, currentIndex } = gameState;
    if (currentIndex >= word.length) return;
    const maxFlower = FLOWER_IMAGES[gameState.level].length - 1;

    if (letter === word[currentIndex]) {
        letterStates[currentIndex] = 'correct';
        gameState.currentIndex = currentIndex + 1;
        gameState.wrongLetter = null;
        if (gameState.currentIndex >= word.length) {
            gameState.won = true;
            stopCamera();
            renderGame();
            return;
        }
    } else {
        gameState.wrongLetter = letter;
        gameState.flowerIndex = Math.min(gameState.flowerIndex + 1, maxFlower);
    }
    gameState.holdCounter = 0;
    gameState.lastDetected = null;
    updateGameUI();
}

function updateGameUI() {
    if (!gameState) return;
    const { word, letterStates, currentIndex, wrongLetter, flowerIndex, level } = gameState;

    // Word display
    const wordEl = document.getElementById('word-display');
    if (wordEl) {
        wordEl.innerHTML = word.split('').map((ch, i) => {
            let cls = 'word-letter';
            if (letterStates[i] === 'correct') cls += ' correct';
            else if (i === currentIndex) cls += ' current';
            return `<div class="${cls}">${letterStates[i] === 'correct' ? ch : '_'}</div>`;
        }).join('');
    }

    // Current letter
    const clEl = document.getElementById('current-letter-display');
    if (clEl) clEl.textContent = currentIndex < word.length ? word[currentIndex] : '';

    // Flower
    const flowerEl = document.getElementById('flower-img');
    if (flowerEl) {
        const imgs = FLOWER_IMAGES[level];
        flowerEl.src = `images/${imgs[Math.min(flowerIndex, imgs.length-1)]}.svg`;
    }

    // Wrong letter display (just update inline without re-render)
    const lang = window.currentLang || 'uk';
    const t = window.texts[lang];
    const wrongDiv = document.querySelector('.game-stat-value.wrong');
    if (wrongDiv) {
        wrongDiv.textContent = wrongLetter || '';
    } else if (wrongLetter) {
        // add wrong block to game-info if not present
        const gameInfo = document.querySelector('.game-info');
        if (gameInfo) {
            const div = document.createElement('div');
            div.className = 'game-stat';
            div.innerHTML = `<div class="game-stat-label">${t.wrong_label}</div><div class="game-stat-value wrong">${wrongLetter}</div>`;
            gameInfo.insertBefore(div, flowerEl || null);
        }
    }
}

// ─── Camera & MediaPipe ───────────────────────────────────────

async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 640, height: 480 }
        });
    } catch (e) {
        const lbl = document.getElementById('camera-label');
        if (lbl) lbl.textContent = '❌ Камера недоступна / Camera unavailable';
        return;
    }

    videoEl = document.createElement('video');
    videoEl.srcObject = cameraStream;
    videoEl.playsInline = true;
    videoEl.muted = true;
    await videoEl.play();

    canvasEl = document.getElementById('game-canvas');
    if (!canvasEl) return;
    canvasEl.width  = videoEl.videoWidth  || 640;
    canvasEl.height = videoEl.videoHeight || 480;
    canvasCtx = canvasEl.getContext('2d');

    if (!handsDetector) {
        handsDetector = new Hands({
            locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });
        handsDetector.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.5,
        });
        handsDetector.onResults(onHandResults);
    }

    processFrame();
}

function stopCamera() {
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    if (cameraStream) { cameraStream.getTracks().forEach(t => t.stop()); cameraStream = null; }
    if (videoEl) { videoEl.srcObject = null; videoEl = null; }
    if (handsDetector) { handsDetector.close(); handsDetector = null; }
    canvasEl = null; canvasCtx = null;
}

async function processFrame() {
    if (!videoEl || !canvasEl || !handsDetector || !gameState || gameState.won) return;
    if (videoEl.readyState >= 2) {
        await handsDetector.send({ image: videoEl });
    }
    animFrameId = requestAnimationFrame(processFrame);
}

function onHandResults(results) {
    if (!canvasEl || !canvasCtx) return;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    // Draw mirrored video
    canvasCtx.translate(canvasEl.width, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);
    canvasCtx.restore();

    if (!results.multiHandLandmarks || !results.multiHandLandmarks.length) {
        resetHoldState();
        return;
    }

    const landmarks = results.multiHandLandmarks[0];

    // Draw hand skeleton (mirrored)
    canvasCtx.save();
    canvasCtx.translate(canvasEl.width, 0);
    canvasCtx.scale(-1, 1);
    if (typeof drawConnectors !== 'undefined') {
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#ffffff', lineWidth: 2 });
        drawLandmarks(canvasCtx, landmarks, { color: '#478C72', lineWidth: 1, radius: 4 });
    }
    canvasCtx.restore();

    if (!gameState || gameState.won) return;

    const label = classifyLandmarks(landmarks);
    if (!label) { resetHoldState(); return; }

    const ukrainianLetter = LETTER_MAP[label] || '?';

    if (label === gameState.lastDetected) {
        gameState.holdCounter++;
    } else {
        gameState.holdCounter = 1;
        gameState.lastDetected = label;
    }

    const progress = Math.min(100, (gameState.holdCounter / HOLD_FRAMES) * 100);

    const detEl   = document.getElementById('detected-letter');
    const progEl  = document.getElementById('hold-progress');
    const lblEl   = document.getElementById('camera-label');
    if (detEl)  detEl.textContent = ukrainianLetter;
    if (progEl) progEl.style.width = progress + '%';
    if (lblEl)  lblEl.textContent  = `✋ ${ukrainianLetter}`;

    if (gameState.holdCounter >= HOLD_FRAMES) {
        processLetter(ukrainianLetter);
    }
}

function resetHoldState() {
    if (gameState) { gameState.holdCounter = 0; gameState.lastDetected = null; }
    const detEl  = document.getElementById('detected-letter');
    const progEl = document.getElementById('hold-progress');
    if (detEl)  detEl.textContent = '—';
    if (progEl) progEl.style.width = '0%';
}

// ─── Model loading ────────────────────────────────────────────

async function loadModel() {
    const loadingEl = document.getElementById('model-loading');
    try {
        const resp = await fetch('model/keypoint_classifier/model_weights.json');
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        mlpLayers = await resp.json();
        console.log('Model loaded:', mlpLayers.length, 'layers');
    } catch (e) {
        console.warn('Could not load model weights:', e);
    }
    if (loadingEl) loadingEl.style.display = 'none';
    // Refresh game page if it's showing the loading notice
    const gamePage = document.getElementById('page-game');
    if (gamePage && gamePage.style.display !== 'none' && !gameState) renderGame();
}

function hideLoading() {
    const el = document.getElementById('model-loading');
    if (el) el.style.display = 'none';
}

window.startGame   = startGame;
window.backToMenu  = backToMenu;
window.renderGame  = renderGame;
window.stopCamera  = stopCamera;

document.addEventListener('DOMContentLoaded', () => {
    setTimeout(hideLoading, 8000);  // safety fallback
    loadModel();
});
