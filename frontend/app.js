document.addEventListener('DOMContentLoaded', () => {
    const API_BASE = (window.API_BASE || '').replace(/\/$/, '');

    function apiUrl(path) {
        return `${API_BASE}${path}`;
    }

    function wsUrl(path) {
        if (API_BASE) {
            const origin = new URL(API_BASE);
            const protocol = origin.protocol === 'https:' ? 'wss:' : 'ws:';
            return `${protocol}//${origin.host}${path}`;
        }
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${protocol}//${window.location.host}${path}`;
    }

    const form = document.getElementById('analyze-form');
    const fileUpload = document.getElementById('resume-upload');
    const fileNameDisplay = document.getElementById('file-name-display');
    const analyzeBtn = document.getElementById('analyze-btn');
    const startInterviewBtn = document.getElementById('start-interview-btn');

    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const resultsState = document.getElementById('results-state');

    const overallScore = document.getElementById('overall-score');
    const overallScoreCircle = document.getElementById('overall-score-circle');
    const gapList = document.getElementById('gap-list');
    const interviewerControls = document.getElementById('interviewer-controls');

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function renderResumeSuggestions(gap) {
        if (gap.match_score >= 8) return '';

        const raw = gap.resume_suggestions && gap.resume_suggestions.length
            ? gap.resume_suggestions
            : (gap.tailored_bullets || []).map((text) => ({ action: 'add', target_line: '', text }));

        if (!raw.length) return '';

        const items = raw.map((s) => {
            const isEdit = (s.action || '').toLowerCase() === 'edit';
            const actionLabel = isEdit ? 'Edit line' : 'Add line';
            const icon = isEdit ? 'fa-pen-to-square' : 'fa-plus';
            const targetHtml = isEdit && s.target_line
                ? `<blockquote class="suggestion-target">${escapeHtml(s.target_line)}</blockquote>`
                : '';

            return `
                <li class="resume-suggestion ${isEdit ? 'edit' : 'add'}">
                    <div class="suggestion-action"><i class="fa-solid ${icon}"></i> ${actionLabel}</div>
                    ${targetHtml}
                    <p class="suggestion-text">${escapeHtml(s.text || '')}</p>
                </li>`;
        }).join('');

            return `
            <div class="gap-bullets">
                <h5>Suggestions</h5>
                <ul class="resume-suggestions">${items}</ul>
            </div>`;
    }

    function setScoreCircle(score) {
        overallScore.textContent = score.toFixed(1);
        if (score >= 8.0) {
            overallScoreCircle.style.borderColor = 'var(--success)';
            overallScoreCircle.style.background = 'rgba(5, 150, 105, 0.1)';
            overallScore.style.color = 'var(--success)';
        } else if (score >= 5.0) {
            overallScoreCircle.style.borderColor = 'var(--warning)';
            overallScoreCircle.style.background = 'rgba(217, 119, 6, 0.1)';
            overallScore.style.color = 'var(--warning)';
        } else {
            overallScoreCircle.style.borderColor = 'var(--danger)';
            overallScoreCircle.style.background = 'rgba(220, 38, 38, 0.1)';
            overallScore.style.color = 'var(--danger)';
        }
    }

    fileUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileNameDisplay.textContent = e.target.files[0].name;
            fileNameDisplay.style.color = 'var(--success)';
        } else {
            fileNameDisplay.textContent = 'No file chosen';
            fileNameDisplay.style.color = 'var(--text-secondary)';
        }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        emptyState.classList.add('hidden');
        resultsState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        analyzeBtn.disabled = true;

        const formData = new FormData(form);

        try {
            const response = await fetch(apiUrl('/api/analyze-resume'), {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Analysis failed');
            }

            const data = await response.json();
            window.__lastAnalysisData = data;
            renderResults(data);

            loadingState.classList.add('hidden');
            resultsState.classList.remove('hidden');
            interviewerControls.classList.remove('hidden');
            startInterviewBtn.disabled = false;

            document.querySelector('.main-content').classList.add('centered');

            window.sessionStorage.setItem('current_pair_id', data.pair_id);
            window.sessionStorage.setItem('current_resume_id', data.resume_id);

            if (data.cached) {
                console.info('Analysis loaded from cache for this resume+JD pair.');
            }
        } catch (error) {
            console.error('Error:', error);
            alert(`Error: ${error.message}`);

            loadingState.classList.add('hidden');
            emptyState.classList.remove('hidden');
            document.querySelector('.main-content').classList.remove('centered');
        } finally {
            analyzeBtn.disabled = false;
        }
    });

    function renderResults(data) {
        window.__lastAnalysisData = data;
        setScoreCircle(data.overall_fit_score);

        const categoryLabel = (cat) => {
            if (cat === 'responsibility') return 'Experience';
            return 'Skill';
        };

        const strengthList = document.getElementById('strength-list');
        const weakList = document.getElementById('weak-list');

        gapList.innerHTML = '';
        if (strengthList) strengthList.innerHTML = '';
        if (weakList) weakList.innerHTML = '';

        data.gap_analyses.forEach((gap) => {
            const card = document.createElement('div');
            card.className = 'gap-card';

            const score = gap.match_score;
            const isStrength = score >= 8;
            const isWeak = score >= 5 && score < 8;
            const suggestionsSection = score < 8 ? renderResumeSuggestions(gap) : '';

            const scoreColor = isStrength ? 'var(--success)' : isWeak ? 'var(--warning)' : 'inherit';
            const scoreBg = isStrength ? 'rgba(5, 150, 105, 0.1)' : isWeak ? 'rgba(217, 119, 6, 0.1)' : 'var(--bg-dark)';

            card.innerHTML = `
                <div class="gap-header">
                    <h4><span class="gap-category">${categoryLabel(gap.category || 'required_skill')}</span> ${gap.requirement}</h4>
                    <span class="gap-score" style="background: ${scoreBg}; color: ${scoreColor};">Score: ${score}/10</span>
                </div>
                <p class="gap-description">${gap.gap_description}</p>
                ${suggestionsSection}
            `;

            if (isStrength) {
                if (strengthList) strengthList.appendChild(card);
            } else if (isWeak) {
                if (weakList) weakList.appendChild(card);
            } else {
                gapList.appendChild(card);
            }
        });
    }

    function renderDebrief(data) {
        const sections = [];
        const addSection = (title, topic, assessment) => {
            if (!topic && !assessment) return;
            sections.push(`
                <div style="margin-bottom: 1rem;">
                    <strong>${title}:</strong> ${topic || '—'}
                    ${assessment ? `<div style="color: var(--text-secondary); margin-top: 0.25rem;"><em>${assessment}</em></div>` : ''}
                </div>
            `);
        };
        addSection('Strong Topic', data.strong_topic, data.strong_assessment);
        addSection('Weak Topic', data.weak_topic, data.weak_assessment);
        addSection('Gap Topic', data.gap_topic, data.gap_assessment);
        if (data.communication_notes) {
            sections.push(`
                <div style="margin-bottom: 1rem; margin-top: 1rem;">
                    <strong>Communication:</strong>
                    <div style="color: var(--text-secondary);"><em>${data.communication_notes}</em></div>
                </div>
            `);
        }
        if (data.overall_readiness) {
            sections.push(`
                <div style="margin-bottom: 1rem; margin-top: 1rem;">
                    <strong>Overall Readiness:</strong>
                    <div style="color: var(--text-secondary);"><em>${data.overall_readiness}</em></div>
                </div>
            `);
        }
        if (data.study_topics && data.study_topics.length > 0) {
            const items = data.study_topics.map(t => `<li>${t}</li>`).join('');
            sections.push(`
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--panel-bg);">
                    <strong><i class="fa-solid fa-book"></i> Topics to Study:</strong>
                    <ul style="margin-top: 0.5rem; padding-left: 1.25rem; color: var(--text-secondary);">${items}</ul>
                </div>
            `);
        }
        return sections.join('') || '<span>No debrief data available.</span>';
    }

    // Voice mock interview
    let audioContext;
    let ws;
    let isInterviewing = false;
    let microphone;
    let scriptProcessor;
    let mediaStream;
    let activeAudioSources = [];
    let nextAudioTime = 0;

    let analyser;
    let visualizerFrame;
    const canvas = document.getElementById('audio-visualizer');
    const canvasCtx = canvas ? canvas.getContext('2d') : null;

    let isMuted = false;
    let userInitiatedStop = false;
    let timerInterval;
    let secondsRemaining = 5 * 60;
    let currentSessionId = null;
    let micLive = false;
    let countdownTimer = null;

    const interviewBtnText = document.getElementById('interview-btn-text');
    const interviewBtnIcon = document.getElementById('interview-btn-icon');
    const interviewStatus = document.getElementById('interview-status');
    const muteBtn = document.getElementById('mute-btn');
    const muteIcon = document.getElementById('mute-icon');
    const timerDisplay = document.getElementById('interview-timer');


    function setInterviewPhase(phase, detail = '') {
        if (!interviewStatus) return;
        interviewStatus.classList.remove('hidden');
        interviewStatus.style.color = 'var(--text-secondary)';
        const labels = {
            connecting: '<i class="fa-solid fa-spinner fa-spin"></i> Connecting…',
            countdown: `<i class="fa-solid fa-hourglass-start"></i> ${detail || 'Starting…'}`,
            ai_speaking: '<i class="fa-solid fa-volume-high"></i> Interviewer speaking…',
            awaiting_user: '<i class="fa-solid fa-microphone fa-beat-fade"></i> Your turn — speak now',
        };
        interviewStatus.innerHTML = labels[phase] || detail || 'In session';
        if (phase === 'awaiting_user') {
            interviewStatus.style.color = 'var(--success)';
        }
    }

    function runCountdownThenReady(onReady) {
        let n = 3;
        setInterviewPhase('countdown', `Starting in ${n}…`);
        countdownTimer = setInterval(() => {
            n -= 1;
            if (n > 0) {
                setInterviewPhase('countdown', `Starting in ${n}…`);
                return;
            }
            clearInterval(countdownTimer);
            countdownTimer = null;
            micLive = true;
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ready' }));
            }
            onReady();
        }, 1000);
    }

    function stopAllPlayback() {
        activeAudioSources.forEach(source => {
            try { source.stop(); } catch (_) {}
        });
        activeAudioSources = [];
        if (audioContext) {
            nextAudioTime = audioContext.currentTime;
        }
    }

    if (muteBtn) {
        muteBtn.addEventListener('click', () => {
            if (!isInterviewing || !mediaStream) return;
            isMuted = !isMuted;

            mediaStream.getAudioTracks().forEach(track => {
                track.enabled = !isMuted;
            });

            if (isMuted) {
                muteIcon.className = 'fa-solid fa-microphone-slash';
                muteIcon.style.color = 'var(--danger)';
                interviewStatus.innerHTML = '<i class="fa-solid fa-microphone-slash"></i> Microphone Muted';
                interviewStatus.style.color = 'var(--danger)';
            } else {
                muteIcon.className = 'fa-solid fa-microphone';
                muteIcon.style.color = 'inherit';
                setInterviewPhase('awaiting_user');
                interviewStatus.style.color = 'var(--success)';
            }
        });
    }

    function updateTimerDisplay() {
        if (!timerDisplay) return;
        const m = Math.floor(secondsRemaining / 60).toString().padStart(2, '0');
        const s = (secondsRemaining % 60).toString().padStart(2, '0');
        timerDisplay.textContent = `${m}:${s}`;

        if (secondsRemaining <= 60) {
            timerDisplay.style.color = 'var(--warning)';
        } else {
            timerDisplay.style.color = 'inherit';
        }
    }

    startInterviewBtn.addEventListener('click', async () => {
        if (isInterviewing) {
            userInitiatedStop = true;
            stopInterview(true);
            return;
        }

        userInitiatedStop = false;
        const pairId = window.sessionStorage.getItem('current_pair_id');
        if (!pairId) {
            alert("No analysis found. Please generate match analysis first.");
            return;
        }

        try {
            ws = new WebSocket(wsUrl(`/api/interview/ws/${pairId}`));

            ws.onopen = async () => {
                isInterviewing = true;
                micLive = false;
                interviewBtnText.textContent = 'End session';
                interviewBtnIcon.className = 'fa-solid fa-stop';
                interviewBtnIcon.style.color = 'var(--danger)';
                setInterviewPhase('connecting');

                if (muteBtn) muteBtn.classList.remove('hidden');
                if (timerDisplay) {
                    timerDisplay.classList.remove('hidden');
                    secondsRemaining = 5 * 60;
                    currentSessionId = null;
                    updateTimerDisplay();
                    timerInterval = setInterval(() => {
                        secondsRemaining--;
                        updateTimerDisplay();
                        if (secondsRemaining <= 0) {
                            userInitiatedStop = true;
                            stopInterview(true);
                        }
                    }, 1000);
                }

                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                    },
                });
                microphone = audioContext.createMediaStreamSource(mediaStream);

                await audioContext.audioWorklet.addModule('processor.js');

                scriptProcessor = new AudioWorkletNode(audioContext, 'pcm-processor', {
                    processorOptions: {
                        sampleRate: audioContext.sampleRate,
                    },
                });

                analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                microphone.connect(analyser);

                scriptProcessor.port.onmessage = (e) => {
                    if (micLive && ws.readyState === WebSocket.OPEN) {
                        ws.send(e.data);
                    }
                };

                microphone.connect(scriptProcessor);

                if (canvas) canvas.classList.remove('hidden');
                drawVisualizer();

                runCountdownThenReady(() => {
                    setInterviewPhase('ai_speaking');
                });
            };

            ws.onmessage = async (event) => {
                if (typeof event.data === 'string') {
                    try {
                        const msg = JSON.parse(event.data);
                        if (msg.type === 'session_started') {
                            currentSessionId = msg.session_id;
                            window.sessionStorage.setItem('current_session_id', msg.session_id);
                        }
                        if (msg.type === 'session_phase') {
                            if (!micLive) return;
                            setInterviewPhase(msg.phase, msg.detail || '');
                        }
                        if (msg.type === 'stop_playback') {
                            stopAllPlayback();
                        }
                    } catch (_) {}
                    return;
                }

                try {
                    const arrayBuffer = await event.data.arrayBuffer();
                    audioContext.decodeAudioData(arrayBuffer, (buffer) => {
                        const source = audioContext.createBufferSource();
                        source.buffer = buffer;
                        source.connect(audioContext.destination);
                        if (analyser) source.connect(analyser);

                        activeAudioSources.push(source);
                        source.onended = () => {
                            activeAudioSources = activeAudioSources.filter(s => s !== source);
                        };

                        const currentTime = audioContext.currentTime;
                        if (nextAudioTime < currentTime) {
                            nextAudioTime = currentTime;
                        }

                        source.start(nextAudioTime);
                        nextAudioTime += buffer.duration;
                    }, (e) => {
                        console.error("Error decoding audio data", e);
                    });
                } catch (e) {
                    console.error("Error handling incoming message", e);
                }
            };

            ws.onclose = () => {
                stopInterview(userInitiatedStop);
            };

            ws.onerror = (e) => {
                console.error("WebSocket error:", e);
                stopInterview(false);
            };
        } catch (error) {
            console.error("Audio error:", error);
            alert("Could not start audio. Please allow microphone permissions.");
        }
    });

    function drawVisualizer() {
        if (!isInterviewing || !canvasCtx) return;
        visualizerFrame = requestAnimationFrame(drawVisualizer);

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyser.getByteTimeDomainData(dataArray);

        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = 'var(--primary)';

        canvasCtx.beginPath();
        const sliceWidth = canvas.width * 1.0 / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;

            if (i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }
            x += sliceWidth;
        }

        canvasCtx.lineTo(canvas.width, canvas.height / 2);
        canvasCtx.stroke();
    }

    async function stopInterview(shouldExtract = false) {
        if (!isInterviewing) return;
        isInterviewing = false;

        const sessionId = currentSessionId || window.sessionStorage.getItem('current_session_id');
        const extractState = document.getElementById('extraction-state');
        const extractContent = document.getElementById('extraction-content');

        if (interviewBtnText) interviewBtnText.textContent = 'Start session';
        if (interviewBtnIcon) {
            interviewBtnIcon.className = 'fa-solid fa-play';
            interviewBtnIcon.style.color = 'inherit';
        }
        if (interviewStatus) interviewStatus.classList.add('hidden');
        if (muteBtn) muteBtn.classList.add('hidden');
        if (timerDisplay) timerDisplay.classList.add('hidden');

        if (timerInterval) clearInterval(timerInterval);
        if (countdownTimer) clearInterval(countdownTimer);
        micLive = false;

        stopAllPlayback();

        if (ws && ws.readyState === WebSocket.OPEN) ws.close();
        if (scriptProcessor) scriptProcessor.disconnect();
        if (microphone) microphone.disconnect();
        if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
        if (audioContext) audioContext.close();
        if (visualizerFrame) cancelAnimationFrame(visualizerFrame);
        if (canvas) {
            canvas.classList.add('hidden');
            if (canvasCtx) canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
        }

        if (!shouldExtract) {
            if (extractState && extractContent) {
                extractState.classList.remove('hidden');
                extractContent.innerHTML = '<span style="color: var(--danger);">Voice session disconnected unexpectedly. Please try again.</span>';
            }
            return;
        }

        await new Promise(resolve => setTimeout(resolve, 1000));

        try {
            if (extractState && extractContent && sessionId) {
                extractState.classList.remove('hidden');
                extractContent.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating your debrief...';

                let debrief = null;
                for (let attempt = 0; attempt < 4; attempt++) {
                    const res = await fetch(apiUrl(`/api/interview/debrief/${sessionId}`), { method: 'POST' });
                    if (res.ok) {
                        debrief = await res.json();
                        break;
                    }
                    if (res.status !== 404 || attempt === 3) {
                        extractContent.innerHTML = '<span style="color: var(--danger);">Failed to generate debrief.</span>';
                        return;
                    }
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }

                if (debrief) {
                    extractContent.innerHTML = renderDebrief(debrief);
                }
            } else if (extractState && extractContent && shouldExtract) {
                extractState.classList.remove('hidden');
                extractContent.innerHTML = '<span style="color: var(--warning);">Session ID missing — debrief unavailable.</span>';
            }
        } catch (e) {
            console.error("Extraction error:", e);
        }
    }
});
