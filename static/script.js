const chatBox = document.getElementById('chatBox');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const micBtn = document.getElementById('micBtn');
const speakBtn = document.getElementById('speakBtn');
const botSelect = document.getElementById('botSelect');
const historyBtn = document.getElementById('historyBtn');
const newSessionBtn = document.getElementById('newSessionBtn');

function createSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        return window.crypto.randomUUID();
    }
    return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

let sessionId = localStorage.getItem('botai_session_id') || createSessionId();
localStorage.setItem('botai_session_id', sessionId);
const urlParams = new URLSearchParams(window.location.search);
if (urlParams.get('widget') === '1') {
    document.body.classList.add('widget-mode');
}
let selectedBotId = urlParams.get('bot_id') || localStorage.getItem('botai_selected_bot_id') || '';

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isListening = false;
let lastBotAnswerText = '';
let availableVoices = [];
let silenceTimer = null;
let shouldKeepListening = false;
let recognitionMode = 'wake';
let isSending = false;
let pendingWakeCommand = '';
let commandHasSpeech = false;
let wakeEnabled = false;
const SILENCE_TIMEOUT_MS = 1200;
const COMMAND_START_TIMEOUT_MS = 5000;
const WAKE_PHRASES = ['hey hoa'];
const WAKE_GREETING = 'Xin chào ông chủ, em có thể giúp gì được cho ngài';

function setButtonState(button, active, label) {
    if (!button) return;
    button.classList.toggle('active', active);
    if (label) {
        button.title = label;
        button.setAttribute('aria-label', label);
    }
}

function safeCancelSpeech() {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
    }
}

function setWakeIdleState(label = 'Bật chờ "Hey Hoa"') {
    wakeEnabled = false;
    shouldKeepListening = false;
    commandHasSpeech = false;
    clearSilenceTimer();
    setButtonState(micBtn, false, label);
}

function stripSources(text) {
    return String(text || '').split('-----------------------------------')[0].trim();
}

async function loadBots() {
    if (!botSelect) return;
    try {
        const response = await fetch('/api/bots');
        const data = await response.json();
        const bots = data.items || [];
        botSelect.innerHTML = bots.map((bot) => (
            `<option value="${bot.id}">${bot.name}</option>`
        )).join('');
        if (bots.length) {
            const stillExists = bots.some((bot) => String(bot.id) === String(selectedBotId));
            selectedBotId = stillExists ? selectedBotId : String(bots[0].id);
            botSelect.value = selectedBotId;
            localStorage.setItem('botai_selected_bot_id', selectedBotId);
        }
    } catch (error) {
        botSelect.innerHTML = '<option value="">Qwen AI</option>';
    }
}

function normalizeSpeechText(text) {
    return stripSources(text).replace(/\s+/g, ' ').trim();
}

function normalizeWakeText(text) {
    return String(text || '')
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

function extractWakeCommand(text) {
    const directMatch = String(text || '').match(/(?:^|\s)(?:hey\s+)?h[oòóỏõọôồốổỗộơờớởỡợ]a\b/i);
    if (directMatch) {
        return String(text || '').slice(directMatch.index + directMatch[0].length).trim();
    }

    const original = String(text || '');
    const normalized = normalizeWakeText(text);
    for (const phrase of WAKE_PHRASES) {
        const index = normalized.indexOf(phrase);
        if (index >= 0) {
            return original.slice(index + phrase.length).trim();
        }
    }
    return null;
}

function clearSilenceTimer() {
    if (silenceTimer) {
        clearTimeout(silenceTimer);
        silenceTimer = null;
    }
}

function scheduleSilenceStop() {
    clearSilenceTimer();
    silenceTimer = setTimeout(() => {
        shouldKeepListening = false;
        if (recognition && isListening) {
            recognition.stop();
        }
    }, SILENCE_TIMEOUT_MS);
}

function scheduleCommandStartTimeout() {
    clearSilenceTimer();
    silenceTimer = setTimeout(() => {
        if (!commandHasSpeech && recognition && isListening) {
            shouldKeepListening = false;
            recognition.stop();
        }
    }, COMMAND_START_TIMEOUT_MS);
}

async function handleWakeGreeting() {
    await speakText(WAKE_GREETING);

    const command = pendingWakeCommand.trim();
    pendingWakeCommand = '';

    if (command) {
        userInput.value = command;
        sendMessage({ speakAfter: true, restartWakeAfter: true });
    } else {
        startCommandListening();
    }
}

function scrollToBottom() {
    chatBox.scrollTop = chatBox.scrollHeight;
}

function appendTextWithBreaks(container, text) {
    const lines = String(text || '').split('\n');
    lines.forEach((line, index) => {
        if (index > 0) container.appendChild(document.createElement('br'));
        container.appendChild(document.createTextNode(line));
    });
}

function appendBotContent(bubble, text) {
    const separator = '-----------------------------------';
    const [answerText, sourceText = ''] = String(text || '').split(separator);

    const answer = document.createElement('div');
    answer.className = 'answer-main';
    appendTextWithBreaks(answer, answerText.trim());
    bubble.appendChild(answer);

    if (!sourceText.trim()) return;

    const source = document.createElement('div');
    source.className = 'answer-source';

    sourceText.trim().split('\n').forEach((line, index) => {
        if (index > 0) source.appendChild(document.createElement('br'));

        const trimmed = line.trim();
        const href = trimmed
            .replace(/^đường dẫn:\s*/i, '')
            .replace(/^duong dan:\s*/i, '');

        if (/^trích đoạn:\s*/i.test(trimmed) || /^trich doan:\s*/i.test(trimmed)) {
            const excerpt = document.createElement('span');
            excerpt.className = 'source-excerpt';
            excerpt.textContent = trimmed;
            source.appendChild(excerpt);
        } else if (/^loại nguồn:\s*/i.test(trimmed) || /^loai nguon:\s*/i.test(trimmed)) {
            const type = document.createElement('span');
            type.className = 'source-type';
            type.textContent = trimmed;
            source.appendChild(type);
        } else if (href !== trimmed) {
            const label = document.createElement('span');
            label.textContent = 'Đường dẫn: ';

            const link = document.createElement('a');
            link.href = href.trim();
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = href.trim();

            source.appendChild(label);
            source.appendChild(link);
        } else {
            source.appendChild(document.createTextNode(trimmed));
        }
    });

    bubble.appendChild(source);
}

function appendFeedbackControls(bubble, question, responseText) {
    if (!question || !responseText || responseText === 'typing') return;

    const [answer, sources = ''] = String(responseText || '').split('-----------------------------------');
    const controls = document.createElement('div');
    controls.className = 'feedback-controls';

    const options = [
        ['helpful', 'Hữu ích'],
        ['wrong', 'Chưa đúng'],
        ['bad_source', 'Sai nguồn'],
    ];

    options.forEach(([rating, label]) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'feedback-btn';
        button.textContent = label;
        button.addEventListener('click', async () => {
            let comment = '';
            if (rating !== 'helpful') {
                comment = window.prompt('Bạn cho mình biết lỗi ngắn gọn được không?', '') || '';
            }

            button.disabled = true;
            try {
                await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        question,
                        answer: answer.trim(),
                        sources: sources.trim(),
                        rating,
                        comment,
                        bot_id: selectedBotId ? Number(selectedBotId) : null,
                    }),
                });
                controls.querySelectorAll('button').forEach((item) => {
                    item.disabled = true;
                });
                const saved = document.createElement('span');
                saved.className = 'feedback-saved';
                saved.textContent = 'Đã ghi nhận';
                controls.appendChild(saved);
            } catch (error) {
                button.disabled = false;
            }
        });
        controls.appendChild(button);
    });

    ['docx', 'pdf'].forEach((format) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'feedback-btn';
        button.textContent = `Xuất ${format.toUpperCase()}`;
        button.addEventListener('click', async () => {
            button.disabled = true;
            try {
                const response = await fetch('/api/export', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question,
                        answer: answer.trim(),
                        sources: sources.trim(),
                        format,
                    }),
                });
                const data = await response.json();
                if (data.ok && data.url) {
                    window.open(data.url, '_blank', 'noopener,noreferrer');
                }
            } finally {
                button.disabled = false;
            }
        });
        controls.appendChild(button);
    });

    bubble.appendChild(controls);
}

function appendMessage(sender, text, meta = {}) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerText = sender === 'user' ? 'Bạn' : 'AI';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    if (text === 'typing') {
        bubble.innerHTML = `
            <div class="typing">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>`;
        msgDiv.id = 'loading-msg';
    } else if (sender === 'ai') {
        appendBotContent(bubble, text);
        appendFeedbackControls(bubble, meta.question, text);
    } else {
        appendTextWithBreaks(bubble, text);
    }

    if (sender === 'user') {
        msgDiv.appendChild(bubble);
        msgDiv.appendChild(avatar);
    } else {
        msgDiv.appendChild(avatar);
        msgDiv.appendChild(bubble);
    }

    chatBox.appendChild(msgDiv);
    scrollToBottom();
    return bubble;
}

function renderGreeting() {
    chatBox.innerHTML = '';
    appendMessage('ai', 'Xin chào! Mình là AI Assistant. Mình có thể giúp gì cho bạn hôm nay?');
}

async function loadChatHistory({ silent = false } = {}) {
    if (!sessionId) return;
    const params = new URLSearchParams({ session_id: sessionId });
    if (selectedBotId) {
        params.set('bot_id', selectedBotId);
    }
    try {
        const response = await fetch(`/api/chat/history?${params.toString()}`);
        const data = await response.json();
        const items = data.items || [];
        if (!items.length) {
            if (!silent) {
                renderGreeting();
                appendMessage('ai', 'Phiên này chưa có lịch sử hội thoại.');
            }
            return;
        }
        chatBox.innerHTML = '';
        items.forEach((item) => {
            appendMessage('user', item.question || '');
            appendMessage('ai', item.response || item.answer || '', { question: item.question || '' });
        });
    } catch (error) {
        if (!silent) {
            appendMessage('ai', `Không tải được lịch sử: ${error.message}`);
        }
    }
}

function startNewSession() {
    sessionId = createSessionId();
    localStorage.setItem('botai_session_id', sessionId);
    lastBotAnswerText = '';
    renderGreeting();
}

function removeLoading() {
    const loading = document.getElementById('loading-msg');
    if (loading) loading.remove();
}

function appendStreamingMessage(question) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message ai';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerText = 'AI';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    const answer = document.createElement('div');
    answer.className = 'answer-main';
    bubble.appendChild(answer);

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatBox.appendChild(msgDiv);
    scrollToBottom();

    return {
        answer,
        finish(finalText) {
            bubble.innerHTML = '';
            appendBotContent(bubble, finalText);
            appendFeedbackControls(bubble, question, finalText);
            scrollToBottom();
        },
    };
}

async function sendMessageFallback(question, options = {}) {
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: question, session_id: sessionId, bot_id: selectedBotId ? Number(selectedBotId) : null })
    });

    const data = await response.json();
    removeLoading();

    if (data.response) {
        lastBotAnswerText = normalizeSpeechText(data.response);
        appendMessage('ai', data.response, { question });
        if (options.speakAfter) {
            await speakText(lastBotAnswerText);
        }
    } else {
        lastBotAnswerText = '';
        appendMessage('ai', 'Xin lỗi, mình chưa nhận được câu trả lời.');
    }
}

async function sendMessage(options = {}) {
    if (isSending) return;
    const text = userInput.value.trim();
    if (!text) return;
    const question = text;

    isSending = true;
    appendMessage('user', question);
    userInput.value = '';
    appendMessage('ai', 'typing');

    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: question, session_id: sessionId, bot_id: selectedBotId ? Number(selectedBotId) : null })
        });

        if (!response.ok || !response.body) {
            await sendMessageFallback(question, options);
            return;
        }

        removeLoading();
        const stream = appendStreamingMessage(question);
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let finalText = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            finalText += chunk;
            stream.answer.textContent = finalText;
            scrollToBottom();
        }

        finalText += decoder.decode();
        stream.finish(finalText);
        lastBotAnswerText = normalizeSpeechText(finalText);
        if (options.speakAfter) {
            await speakText(lastBotAnswerText);
        }
    } catch (error) {
        try {
            await sendMessageFallback(question, options);
        } catch (fallbackError) {
            removeLoading();
            appendMessage('ai', `Lỗi kết nối: ${fallbackError.message}. Hãy chắc chắn Ollama đang chạy!`);
        }
    } finally {
        isSending = false;
        if (options.restartWakeAfter && wakeEnabled) {
            startWakeListening();
        }
    }
}

sendBtn.addEventListener('click', sendMessage);
if (historyBtn) {
    historyBtn.addEventListener('click', () => loadChatHistory());
}
if (newSessionBtn) {
    newSessionBtn.addEventListener('click', startNewSession);
}

userInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        sendMessage();
    }
});

function initializeSpeechRecognition() {
    if (!SpeechRecognition || !micBtn) {
        if (micBtn) {
            micBtn.disabled = true;
            micBtn.title = 'Trình duyệt không hỗ trợ nhập giọng nói';
            micBtn.setAttribute('aria-label', 'Trình duyệt không hỗ trợ nhập giọng nói');
        }
        return;
    }

    recognition = new SpeechRecognition();
    recognition.lang = 'vi-VN';
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onstart = () => {
        isListening = true;
        if (recognitionMode === 'wake') {
            setButtonState(micBtn, false, 'Đang chờ "Hey Hoa"');
        } else {
            setButtonState(micBtn, true, 'Đang nghe câu hỏi');
            commandHasSpeech = false;
            scheduleCommandStartTimeout();
        }
    };

    recognition.onresult = (event) => {
        let transcript = '';
        for (let index = event.resultIndex; index < event.results.length; index += 1) {
            transcript += event.results[index][0].transcript;
        }
        if (transcript.trim()) {
            if (recognitionMode === 'wake') {
                const command = extractWakeCommand(transcript);
                if (command !== null) {
                    pendingWakeCommand = command;
                    recognitionMode = 'greeting';
                    shouldKeepListening = false;
                    clearSilenceTimer();
                    recognition.stop();
                }
                return;
            }

            commandHasSpeech = true;
            userInput.value = transcript.trim();
            scheduleSilenceStop();
        }
    };

    recognition.onerror = (event) => {
        isListening = false;
        const error = event && event.error;
        const label = error === 'not-allowed' || error === 'service-not-allowed'
            ? 'Trình duyệt chưa cấp quyền micro'
            : 'Bật chờ "Hey Hoa"';
        setWakeIdleState(label);
    };

    recognition.onnomatch = () => {
        if (recognitionMode === 'command') {
            commandHasSpeech = false;
            scheduleCommandStartTimeout();
        }
    };

    recognition.onspeechend = () => {
        if (recognitionMode === 'command' && commandHasSpeech) {
            scheduleSilenceStop();
        }
    };

    recognition.onend = () => {
        isListening = false;
        clearSilenceTimer();
        setButtonState(micBtn, false, 'Bật chờ "Hey Hoa"');

        if (recognitionMode === 'greeting') {
            handleWakeGreeting();
            return;
        }

        if (recognitionMode === 'command') {
            const text = userInput.value.trim();
            recognitionMode = 'wake';
            commandHasSpeech = false;
            if (text) {
                sendMessage({ speakAfter: true, restartWakeAfter: true });
            } else {
                if (wakeEnabled) startWakeListening();
            }
            return;
        }

        if (recognitionMode === 'wake' && shouldKeepListening && wakeEnabled) {
            startWakeListening();
            return;
        }

        if (shouldKeepListening && wakeEnabled) {
            try {
                recognition.start();
                return;
            } catch (error) {
                setWakeIdleState();
            }
        }
        userInput.focus();
    };
}

function toggleSpeechRecognition() {
    if (!recognition) return;
    if (isListening) {
        setWakeIdleState();
        clearSilenceTimer();
        recognition.stop();
    } else {
        startWakeListening();
    }
}

function startWakeListening() {
    if (!recognition || isSending || isListening) return;
    safeCancelSpeech();
    wakeEnabled = true;
    recognitionMode = 'wake';
    commandHasSpeech = false;
    shouldKeepListening = true;
    try {
        recognition.start();
    } catch (error) {
        setTimeout(() => {
            if (shouldKeepListening && !isListening && !isSending) {
                try {
                    recognition.start();
                } catch (retryError) {
                    setWakeIdleState();
                }
            }
        }, 350);
    }
}

function startCommandListening() {
    if (!recognition || isSending || isListening) return;
    safeCancelSpeech();
    recognitionMode = 'command';
    commandHasSpeech = false;
    shouldKeepListening = true;
    userInput.value = '';
    try {
        recognition.start();
    } catch (error) {
        setTimeout(() => {
            if (!isListening && !isSending) {
                try {
                    recognition.start();
                } catch (retryError) {
                    recognitionMode = 'wake';
                    setWakeIdleState();
                }
            }
        }, 350);
    }
}

function loadVoices() {
    availableVoices = window.speechSynthesis ? window.speechSynthesis.getVoices() : [];
}

function vietnameseVoice() {
    return (
        availableVoices.find((voice) => voice.lang && voice.lang.toLowerCase().startsWith('vi')) ||
        availableVoices.find((voice) => /vietnam|vietnamese|việt/i.test(voice.name)) ||
        null
    );
}

function speakText(text) {
    if (!('speechSynthesis' in window)) return Promise.resolve();

    return new Promise((resolve) => {
        window.speechSynthesis.cancel();
        const speechText = text || 'Chưa có câu trả lời để đọc.';
        if (!speechText.trim()) {
            resolve();
            return;
        }

        const utterance = new SpeechSynthesisUtterance(speechText);
        utterance.lang = 'vi-VN';
        utterance.rate = 0.95;
        utterance.pitch = 1;

        const voice = vietnameseVoice();
        if (voice) utterance.voice = voice;

        utterance.onstart = () => setButtonState(speakBtn, true, 'Dừng đọc');
        utterance.onend = () => {
            setButtonState(speakBtn, false, 'Đọc câu trả lời');
            resolve();
        };
        utterance.onerror = () => {
            setButtonState(speakBtn, false, 'Đọc câu trả lời');
            resolve();
        };

        window.speechSynthesis.speak(utterance);
    });
}

function speakLatestAnswer() {
    if (!('speechSynthesis' in window) || !speakBtn) return;

    if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        setButtonState(speakBtn, false, 'Đọc câu trả lời');
        return;
    }

    speakText(lastBotAnswerText);
}

initializeSpeechRecognition();
loadVoices();
if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = loadVoices;
} else if (speakBtn) {
    speakBtn.disabled = true;
    speakBtn.title = 'Trình duyệt không hỗ trợ đọc văn bản';
    speakBtn.setAttribute('aria-label', 'Trình duyệt không hỗ trợ đọc văn bản');
}

if (micBtn) micBtn.addEventListener('click', toggleSpeechRecognition);
if (speakBtn) speakBtn.addEventListener('click', speakLatestAnswer);
if (botSelect) {
    botSelect.addEventListener('change', () => {
        selectedBotId = botSelect.value;
        localStorage.setItem('botai_selected_bot_id', selectedBotId);
        loadChatHistory({ silent: true });
    });
}
loadBots().then(() => loadChatHistory({ silent: true }));
setWakeIdleState();
