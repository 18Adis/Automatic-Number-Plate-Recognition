(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const state = {
    mode: 'upload', ready: false, model: null, stream: null, live: false,
    inFlight: false, starting: false, controller: null, timer: null,
    generation: 0, session: crypto.randomUUID(), image: null, imageUrl: null,
    videoUrl: null, lastResult: null, resultAt: 0,
    sessionIds: new Set(),
    ocrMethod: 'cct', methods: [], preparing: false, comparison: null,
  };
  const video = $('sourceVideo');
  const sourceImage = $('sourceImage');
  const overlay = $('overlay');
  const stage = $('stage');
  const captureCanvas = document.createElement('canvas');
  const captureContext = captureCanvas.getContext('2d');
  const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const percent = (value) => Number.isFinite(value) ? `${Math.round(value * 100)}%` : '—';
  const statusName = (value) => ({confirmed: 'Confirmed', needs_review: 'Needs review', reading: 'Reading', corrected: 'Corrected'}[value] || 'Needs review');
  const statusClass = (value) => value === 'needs_review' ? 'review' : value === 'reading' ? 'reading' : '';
  const icon = (id) => `<svg aria-hidden="true"><use href="#i-${id}"/></svg>`;

  async function api(url, options = {}) {
    const response = await fetch(url, options);
    let body;
    try { body = await response.json(); } catch { body = {}; }
    if (!response.ok) {
      const detail = typeof body.detail === 'string' ? body.detail : (body.message || 'The request could not be completed.');
      throw new Error(detail);
    }
    return body;
  }

  function feedback(message, type = '') {
    $('feedback').textContent = message;
    $('feedback').className = `feedback ${type}`;
    $('feedback').hidden = !message;
  }

  function clearResults() {
    state.comparison = null;
    $('comparisonResults').hidden = true;
    $('comparisonStatus').hidden = true;
    $('comparisonFooter').hidden = true;
    state.lastResult = null;
    overlay.hidden = true;
    $('frameLabel').hidden = true;
    $('plateCount').textContent = '—';
    $('latency').textContent = '—';
    $('resultCount').textContent = '0';
    $('results').innerHTML = `<div class="result-empty"><span class="result-empty-icon">${icon('scan')}</span><h3>No results yet</h3><p>Select an input to begin.</p></div>`;
  }

  function updateButtons() {
    $('ocrMethod').disabled = state.inFlight || state.preparing;
    $('compareBtn').disabled = state.inFlight || state.preparing || (state.mode === 'upload' ? !state.image : !video.videoWidth);
    $('expectedText').disabled = state.inFlight;
    if (state.preparing) {
      for (const id of ['reanalyzeBtn','snapshotBtn','pauseBtn','videoPlayBtn','cameraStartBtn']) $(id).disabled = true;
      return;
    }
    $('reanalyzeBtn').disabled = !state.image || !state.ready || state.inFlight;
    $('cameraStartBtn').disabled = state.starting || state.inFlight;
    $('cameraStartBtn').hidden = !!state.stream;
    $('pauseBtn').hidden = !state.stream;
    $('pauseBtn').disabled = !state.ready || (state.inFlight && !state.live);
    $('pauseBtn').innerHTML = `${icon(state.live ? 'pause' : 'play')}<span>${state.live ? 'Pause' : 'Resume'}</span>`;
    $('snapshotBtn').disabled = !state.stream || state.inFlight || !state.ready;
    $('stopBtn').disabled = !state.stream && !state.starting;
    $('videoPlayBtn').disabled = !state.videoUrl || !state.ready || state.inFlight && !state.live;
    $('videoPlayBtn').innerHTML = `${icon(state.live ? 'pause' : 'play')}<span>${state.live ? 'Pause' : 'Play & analyze'}</span>`;
    $('videoStopBtn').disabled = !state.videoUrl;
    $('cameraSelect').disabled = state.starting;
  }

  function haltAnalysis() {
    state.generation += 1;
    state.live = false;
    clearTimeout(state.timer);
    state.timer = null;
    if (state.controller) state.controller.abort();
    state.controller = null;
    state.inFlight = false;
    $('busyOverlay').hidden = true;
    $('viewerDot').classList.remove('live');
  }

  function stopMedia(resetVideo = true) {
    haltAnalysis();
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
    state.starting = false;
    video.pause();
    video.srcObject = null;
    if (resetVideo) {
      video.removeAttribute('src');
      video.load();
      if (state.videoUrl) URL.revokeObjectURL(state.videoUrl);
      state.videoUrl = null;
    }
    updateButtons();
  }

  function setMode(mode) {
    if (state.mode === mode) return;
    stopMedia();
    state.mode = mode;
    state.session = crypto.randomUUID();
    feedback('');
    clearResults();
    document.querySelectorAll('.input-tab').forEach((tab) => {
      const active = tab.dataset.mode === mode;
      tab.classList.toggle('selected', active);
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      if (active) $('inputPanel').setAttribute('aria-labelledby', tab.id);
    });
    $('uploadToolbar').hidden = mode !== 'upload';
    $('cameraToolbar').hidden = mode !== 'camera';
    $('videoToolbar').hidden = mode !== 'video';
    video.hidden = true;
    sourceImage.hidden = true;
    $('emptyState').hidden = false;
    stage.classList.remove('has-media');
    $('viewerSubtitle').textContent = 'Ready for input';
    $('chooseFileBtn').innerHTML = `${icon(mode === 'camera' ? 'camera' : 'upload')}<span>${mode === 'camera' ? 'Start camera' : mode === 'video' ? 'Choose video' : 'Choose image'}</span>${icon('arrow')}`;
    const labels = {
      upload: ['Image preview', 'IMAGE', 'Upload a vehicle image', 'Drag and drop an image, or select a file.', 'JPG, PNG, WEBP · up to 30 MB'],
      camera: ['Live camera', 'CAMERA', 'Connect a camera', 'Choose a camera to start live detection.', 'Webcam or connected camera'],
      video: ['Video workspace', 'VIDEO', 'Upload a video', 'Select a video file for frame-by-frame detection.', 'Local video · up to 250 MB · sound off'],
    }[mode];
    ['viewerTitle','sourceChip','emptyTitle','emptyDescription','fileHint'].forEach((id, index) => $(id).textContent = labels[index]);
    if (mode === 'upload' && state.image && state.imageUrl) {
      sourceImage.hidden = false;
      $('emptyState').hidden = true;
      stage.classList.add('has-media');
      $('viewerSubtitle').textContent = 'Image loaded';
    }
    $('videoName').textContent = 'No video selected';
    updateButtons();
  }

  async function refreshStatus() {
    try {
      const status = await api('/api/status');
      state.methods = status.ocr_methods || [];
      $('ocrMethod').innerHTML = state.methods.map(method => `<option value="${escape(method.id)}" ${method.available ? '' : 'disabled'}>${escape(method.label)}${method.available ? '' : ' · setup needed'}</option>`).join('');
      $('ocrMethod').value = state.ocrMethod;
      state.model = status;
      const currentMethod = state.methods.find(method => method.id === state.ocrMethod);
      state.ready = Boolean(status.ready && (currentMethod?.ready ?? true));
      const custom = /custom/i.test(status.model_mode || '');
      $('modelBadge').className = `model-badge ${state.ready ? 'ready' : 'error'}`;
      $('modelBadgeText').textContent = state.ready ? (custom ? 'Custom model ready' : 'Model ready') : 'OCR method not loaded';
      $('modelMessage').textContent = status.message || (status.ready ? (custom ? 'Your custom model is ready for recognition.' : 'Pretrained baseline active. Results have not yet been validated on your dataset.') : 'Load the recognition model to start analyzing images.');
      if (status.ready && !state.ready) $('modelMessage').textContent = 'Load the selected OCR method to start recognition.';
      $('modelNotice').hidden = state.ready;
      $('warmupBtn').hidden = state.ready;
      $('device').textContent = status.device ? String(status.device).toUpperCase() : '—';
      const selected = state.methods.find(method => method.id === state.ocrMethod);
      $('engineName').textContent = selected?.label || 'Plate OCR · CCT';
      $('engineDescription').textContent = selected?.message || 'Plate detection + character recognition';
      $('engineName').title = [status.detector, status.recognizer].filter(Boolean).join(' / ');
    } catch (error) {
      state.ready = false;
      $('modelNotice').hidden = false;
      $('modelBadge').className = 'model-badge error';
      $('modelBadgeText').textContent = 'Connection unavailable';
      $('modelMessage').textContent = 'The local recognition service is not responding. Start the server, then reload this page.';
      $('warmupBtn').hidden = true;
      $('engineName').textContent = 'Service unavailable';
      $('engineDescription').textContent = 'Connect to the local server to inspect model details.';
    }
    updateButtons();
  }

  async function warmup() {
    state.preparing = true;
    updateButtons();
    $('warmupBtn').disabled = true;
    $('warmupBtn').textContent = 'Loading…';
    $('modelMessage').textContent = 'Loading the recognition engine. The first load may take a little longer.';
    try {
      await api('/api/warmup', {method: 'POST', body:new URLSearchParams({ocr_method:state.ocrMethod})});
      await refreshStatus();
    } catch (error) {
      await refreshStatus();
      feedback(`The model could not load: ${error.message}`, 'error');
    } finally {
      state.preparing = false;
      $('warmupBtn').disabled = false;
      $('warmupBtn').textContent = 'Load model ↗';
      updateButtons();
    }
  }

  function showResults(result, elapsed, source) {
    state.lastResult = result;
    state.resultAt = performance.now();
    const detections = Array.isArray(result.detections) ? result.detections : [];
    $('plateCount').textContent = String(detections.length);
    $('resultCount').textContent = String(detections.length);
    $('latency').innerHTML = Number.isFinite(result.inference_ms) ? `${Math.round(result.inference_ms)}<small>ms</small>` : '—';
    if (source === 'live' || source === 'video') {
      $('frameLabel').hidden = false;
      $('frameLabel').textContent = `Latest analyzed frame · ${Math.round(elapsed)} ms round trip`;
    } else {
      $('frameLabel').hidden = true;
    }
    if (detections.length === 0) {
      $('results').innerHTML = `<div class="result-empty"><span class="result-empty-icon">${icon('scan')}</span><h3>No plate detected</h3><p>Try a clearer, closer view of the plate, or lower the detection threshold.</p></div>`;
      if (source === 'upload' || source === 'snapshot') feedback('No number plate was found in this image. Try a closer, sharper view.', 'warning');
    } else {
      if (source === 'upload' || source === 'snapshot') feedback('');
      $('results').innerHTML = detections.map((detection, index) => {
        if (detection.id && detection.text && detection.status === 'confirmed') state.sessionIds.add(detection.id);
        const crop = typeof detection.crop_data_url === 'string' && /^data:image\/(jpeg|png|webp);base64,/.test(detection.crop_data_url) ? `<img class="result-crop" src="${escape(detection.crop_data_url)}" alt="Detected number plate ${index + 1}">` : '';
        return `<article class="result-card"><div class="result-topline"><span>PLATE ${String(index + 1).padStart(2,'0')}</span><span class="status-label ${statusClass(detection.status)}">${escape(statusName(detection.status))}</span></div>${crop}<div class="plate-number">${escape(detection.text || 'Reading…')}</div>${!detection.text ? '<p class="result-description">Plate located. Characters could not be read confidently.</p>' : ''}<div class="confidence-row"><span>Plate detection</span><strong>${percent(detection.detection_confidence)}</strong></div><div class="confidence-bar"><span style="width:${Math.max(0,Math.min(100,(detection.detection_confidence || 0)*100))}%"></span></div><div class="confidence-row"><span>Character recognition</span><strong>${percent(detection.recognition_confidence)}</strong></div><div class="confidence-bar"><span style="width:${Math.max(0,Math.min(100,(detection.recognition_confidence || 0)*100))}%"></span></div></article>`;
      }).join('');
      const methodLabel = state.methods.find(method => method.id === result.ocr_method)?.label || result.ocr_method;
      $('results').insertAdjacentHTML('afterbegin', `<div class="result-method">${escape(methodLabel)} · ${detections.reduce((sum,item) => sum + (item.ocr_ms || 0),0).toFixed(1)} ms OCR</div>`);
      $('readingCount').textContent = String(state.sessionIds.size);
    }
    drawOverlay();

  }

  function drawOverlay() {
    const result = state.lastResult;
    if (!result || !result.width || !result.height) { overlay.hidden = true; return; }
    // Live images move; only retain boxes briefly so older positions are not presented as current.
    const stale = (state.mode === 'camera' || state.mode === 'video') && !video.paused && performance.now() - state.resultAt > 1500;
    overlay.hidden = stale;
    if (stale) {
      $('frameLabel').hidden = false;
      $('frameLabel').textContent = 'Reading current frame… previous boxes cleared';
      return;
    }
    const rect = stage.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    overlay.width = Math.round(rect.width * ratio);
    overlay.height = Math.round(rect.height * ratio);
    const ctx = overlay.getContext('2d');
    ctx.setTransform(ratio,0,0,ratio,0,0);
    ctx.clearRect(0,0,rect.width,rect.height);
    const scale = Math.min(rect.width / result.width, rect.height / result.height);
    const offsetX = (rect.width - result.width * scale) / 2;
    const offsetY = (rect.height - result.height * scale) / 2;
    ctx.lineWidth = 2;
    ctx.font = '600 11px Segoe UI, sans-serif';
    for (const detection of result.detections || []) {
      if (!Array.isArray(detection.box) || detection.box.length !== 4) continue;
      const [x1,y1,x2,y2] = detection.box;
      const x = offsetX + x1 * scale, y = offsetY + y1 * scale;
      const width = (x2-x1)*scale, height = (y2-y1)*scale;
      ctx.strokeStyle = detection.status === 'confirmed' ? '#bde284' : '#e8c574';
      ctx.strokeRect(x,y,width,height);
      const label = `${detection.text || 'Reading…'} · ${percent(detection.detection_confidence)}`;
      const textWidth = Math.min(ctx.measureText(label).width + 14, rect.width - 8);
      const labelX = Math.max(4,Math.min(x,rect.width - textWidth - 4));
      const labelY = y > 26 ? y-24 : Math.min(y+height+3,rect.height-24);
      ctx.fillStyle = '#183d2fee';
      ctx.fillRect(labelX,labelY,textWidth,21);
      ctx.fillStyle = '#edf6df';
      ctx.fillText(label,labelX+7,labelY+14,textWidth-14);
    }
  }

  async function analyzeBlob(blob, source, generation = state.generation) {
    if (state.inFlight || state.preparing || !state.ready || generation !== state.generation) return;
    const controller = new AbortController();
    state.controller = controller;
    state.inFlight = true;
    const uploaded = source === 'upload' || source === 'snapshot';
    $('busyOverlay').hidden = !uploaded;
    $('viewerSubtitle').textContent = 'Analyzing…';
    if (uploaded) feedback('');
    updateButtons();
    const timeout = setTimeout(() => controller.abort(), 120000);
    const started = performance.now();
    try {
      const data = new FormData();
      data.append('file', blob, 'frame.jpg');
      data.append('source', source);
      data.append('session_id', state.session);
      data.append('confidence', String(Number($('confidence').value) / 100));
      data.append('ocr_method', state.ocrMethod);
      const result = await api('/api/analyze', {method:'POST', body:data, signal:controller.signal});
      if (generation !== state.generation) return;
      showResults(result, performance.now()-started, source);
      $('viewerSubtitle').textContent = state.live ? 'Live analysis' : source === 'snapshot' ? 'Snapshot analyzed' : 'Analysis complete';
    } catch (error) {
      if (generation !== state.generation) return;
      state.live = false;
      video.pause();
      $('viewerDot').classList.remove('live');
      $('viewerSubtitle').textContent = 'Analysis paused';
      feedback(error.name === 'AbortError' ? 'Recognition took too long. Try a smaller image or reload the model.' : `Recognition could not finish: ${error.message}`, 'error');
      refreshStatus();
    } finally {
      clearTimeout(timeout);
      if (generation === state.generation) {
        state.inFlight = false;
        state.controller = null;
        $('busyOverlay').hidden = true;
        updateButtons();
      }
    }
  }

  async function imageBlob(image) {
    const maxSide = 1280;
    const scale = Math.min(1,maxSide/image.naturalWidth,maxSide/image.naturalHeight);
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1,Math.round(image.naturalWidth*scale));
    canvas.height = Math.max(1,Math.round(image.naturalHeight*scale));
    canvas.getContext('2d').drawImage(image,0,0,canvas.width,canvas.height);
    return new Promise((resolve,reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error('This image could not be prepared.')), 'image/jpeg', .92));
  }

  async function loadImage(file, name = file.name || 'Sample image') {
    if (!file.type.startsWith('image/')) { feedback('Choose a JPG, PNG, WEBP or another supported image file.', 'error'); return; }
    if (file.size > 30*1024*1024) { feedback('This image is larger than 30 MB. Choose a smaller image.', 'error'); return; }
    if (state.mode !== 'upload') setMode('upload');
    haltAnalysis();
    state.session = crypto.randomUUID();
    const generation = state.generation;
    clearResults();
    feedback('');
    const nextUrl = URL.createObjectURL(file);
    try {
      const image = new Image();
      image.src = nextUrl;
      await image.decode();
      if (generation !== state.generation) { URL.revokeObjectURL(nextUrl); return; }
      if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
      state.imageUrl = nextUrl;
      state.image = await imageBlob(image);
      if (generation !== state.generation) return;
      sourceImage.src = nextUrl;
      sourceImage.alt = `Vehicle image: ${name}`;
      sourceImage.hidden = false;
      video.hidden = true;
      $('emptyState').hidden = true;
      stage.classList.add('has-media');
      $('fileName').textContent = name;
      $('viewerSubtitle').textContent = 'Image loaded';
      updateButtons();
      if (!state.ready) await warmup();
      if (state.ready) await analyzeBlob(state.image,'upload',generation);
    } catch (error) {
      URL.revokeObjectURL(nextUrl);
      feedback(`This image could not be opened. ${error.message}`, 'error');
    }
  }

  async function frameBlob() {
    if (!video.videoWidth || !video.videoHeight || video.readyState < 2) return null;
    const scale = Math.min(1,1280/video.videoWidth,1280/video.videoHeight);
    captureCanvas.width = Math.max(1,Math.round(video.videoWidth*scale));
    captureCanvas.height = Math.max(1,Math.round(video.videoHeight*scale));
    captureContext.drawImage(video,0,0,captureCanvas.width,captureCanvas.height);
    return new Promise((resolve) => captureCanvas.toBlob(resolve,'image/jpeg',.88));
  }

  async function tick(generation) {
    if (!state.live || generation !== state.generation) return;
    if (!state.inFlight && !video.ended) {
      try {
        const blob = await frameBlob();
        if (blob && generation === state.generation) await analyzeBlob(blob,state.mode === 'camera' ? 'live' : 'video',generation);
      } catch (error) {
        state.live = false;
        feedback(`This frame could not be read: ${error.message}`, 'error');
        updateButtons();
      }
    }
    if (state.live && generation === state.generation) state.timer = setTimeout(() => tick(generation),200);
  }

  async function enumerateCameras() {
    try {
      const selected = $('cameraSelect').value;
      const cameras = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === 'videoinput');
      $('cameraSelect').innerHTML = '<option value="">Default camera</option>' + cameras.map((camera,index) => `<option value="${escape(camera.deviceId)}">${escape(camera.label || `Camera ${index+1}`)}</option>`).join('');
      if (cameras.some((camera) => camera.deviceId === selected)) $('cameraSelect').value = selected;
    } catch { /* The default device remains usable if device enumeration is restricted. */ }
  }

  async function startCamera() {
    if (state.starting) return;
    if (state.mode !== 'camera') setMode('camera');
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      feedback('Camera access is unavailable here. Open the app on localhost or over HTTPS in a browser that supports camera access.', 'error');
      return;
    }
    stopMedia();
    clearResults();
    feedback('');
    state.starting = true;
    state.session = crypto.randomUUID();
    const generation = state.generation;
    updateButtons();
    $('viewerSubtitle').textContent = 'Waiting for camera access…';
    try {
      const deviceId = $('cameraSelect').value;
      const stream = await navigator.mediaDevices.getUserMedia({audio:false,video:{...(deviceId ? {deviceId:{exact:deviceId}} : {facingMode:{ideal:'environment'}}),width:{ideal:1280},height:{ideal:720}}});
      if (generation !== state.generation || state.mode !== 'camera') { stream.getTracks().forEach((track) => track.stop()); return; }
      state.stream = stream;
      video.srcObject = stream;
      await video.play();
      video.hidden = false;
      sourceImage.hidden = true;
      $('emptyState').hidden = true;
      stage.classList.add('has-media');
      state.live = state.ready;
      $('viewerDot').classList.toggle('live', state.live);
      $('viewerSubtitle').textContent = state.live ? 'Live analysis' : 'Camera preview';
      if (!state.ready) feedback('Camera preview is active. Load the model above, then select Resume to recognize plates.', 'warning');
      await enumerateCameras();
      const track = stream.getVideoTracks()[0];
      track.addEventListener('ended', () => {
        if (state.stream === stream) { stopCamera(); feedback('The camera disconnected. Reconnect it and start again.', 'warning'); }
      }, {once:true});
      if (state.live) tick(generation);
    } catch (error) {
      if (generation !== state.generation) return;
      const messages = {
        NotAllowedError:'Camera access was denied. Allow camera access in your browser’s address bar, then try again.',
        NotFoundError:'No camera was found. Connect a webcam and try again.',
        NotReadableError:'The camera is busy or unavailable. Close other apps using it, then try again.',
        OverconstrainedError:'That camera is unavailable. Select the default camera and try again.',
      };
      feedback(messages[error.name] || `The camera could not start: ${error.message}`, 'error');
      stopMedia();
      $('viewerSubtitle').textContent = 'Camera unavailable';
    } finally {
      if (generation === state.generation) state.starting = false;
      updateButtons();
    }
  }

  function stopCamera() {
    stopMedia();
    clearResults();
    video.hidden = true;
    $('emptyState').hidden = false;
    stage.classList.remove('has-media');
    $('viewerSubtitle').textContent = 'Camera stopped';
    feedback('Camera stopped. Your camera is no longer in use.');
  }

  async function toggleCameraAnalysis() {
    if (!state.stream) return;
    if (state.live) {
      haltAnalysis();
      clearResults();
      $('viewerSubtitle').textContent = 'Preview only · analysis paused';
      feedback('Analysis paused. The camera preview is still active; select Stop to release the camera.');
    } else if (state.ready) {
      await video.play();
      state.live = true;
      $('viewerDot').classList.add('live');
      feedback('');
      tick(state.generation);
    }
    updateButtons();
  }

  async function snapshot() {
    if (!state.stream || state.inFlight) return;
    haltAnalysis();
    clearResults();
    const generation = state.generation;
    const blob = await frameBlob();
    if (!blob || generation !== state.generation) return;
    // Freeze the preview so the result's bounding boxes remain on the captured frame.
    video.pause();
    await analyzeBlob(blob,'snapshot',generation);
    feedback('Snapshot analyzed. Select Resume for live recognition, or Stop to release the camera.');
    updateButtons();
  }

  async function loadVideo(file) {
    if (!file.type.startsWith('video/')) { feedback('Choose a video file supported by your browser.', 'error'); return; }
    if (file.size > 250*1024*1024) { feedback('Choose a video smaller than 250 MB.', 'error'); return; }
    if (state.mode !== 'video') setMode('video');
    stopMedia();
    clearResults();
    feedback('');
    state.session = crypto.randomUUID();
    state.videoUrl = URL.createObjectURL(file);
    video.src = state.videoUrl;
    video.hidden = false;
    sourceImage.hidden = true;
    $('emptyState').hidden = true;
    stage.classList.add('has-media');
    $('videoName').textContent = file.name;
    $('viewerSubtitle').textContent = 'Video loaded';
    video.load();
    updateButtons();
    if (!state.ready) feedback('Video loaded. Load the model above, then select Play & analyze.', 'warning');
  }

  async function toggleVideo() {
    if (!state.videoUrl || !state.ready) return;
    if (state.live) {
      haltAnalysis();
      video.pause();
      $('viewerSubtitle').textContent = 'Video paused';
      updateButtons();
      return;
    }
    try {
      if (video.ended) { video.currentTime = 0; state.session = crypto.randomUUID(); }
      await video.play();
      state.live = true;
      $('viewerDot').classList.add('live');
      feedback('Frames are analyzed during playback. Some frames may be skipped to keep the video responsive.');
      tick(state.generation);
      updateButtons();
    } catch (error) { feedback(`This video could not play: ${error.message}`, 'error'); }
  }

  function renderComparison(result) {
    state.comparison = {...result, compared_at:new Date().toISOString()};
    const setup = (result.initialization_ms || 0) + Object.values(result.preparation_ms || {}).reduce((a,b) => a+b,0);
    $('comparisonStatus').textContent = result.detections.length ? `Detection ${result.detection_ms.toFixed(1)} ms · model setup ${setup.toFixed(0)} ms (excluded below)` : 'No plate found. Try a clearer view or lower the detection threshold.';
    $('comparisonResults').innerHTML = result.detections.map((plate,index) => {
      const crop = /^data:image\/jpeg;base64,/.test(plate.crop_data_url || '') ? `<img src="${escape(plate.crop_data_url)}" alt="Comparison plate ${index+1}">` : '';
      return `<div class="comparison-plate"><div class="comparison-plate-heading">${crop}<strong>Plate ${String(index+1).padStart(2,'0')}</strong>${result.expected_text ? `<span>Expected: <b>${escape(result.expected_text)}</b></span>` : ''}</div><div class="comparison-table-wrap"><table><thead><tr><th>OCR METHOD</th><th>READING</th><th>OCR TIME</th><th>CONFIDENCE</th><th>EXACT MATCH</th><th>CHAR. ERROR</th></tr></thead><tbody>${plate.readings.map(reading => `<tr><td>${escape(reading.label)}</td><td class="comparison-reading">${reading.error ? `<span class="comparison-error">${escape(reading.error)}</span>` : escape(reading.text || 'No text read')}</td><td>${Number.isFinite(reading.ocr_ms) ? `${reading.ocr_ms.toFixed(1)} ms` : '—'}</td><td>${reading.recognition_confidence === null ? '—' : percent(reading.recognition_confidence)}</td><td>${reading.exact_match === null ? '—' : reading.exact_match ? '<span class="match-yes">Match</span>' : '<span class="match-no">Mismatch</span>'}</td><td>${Number.isFinite(reading.character_error_rate) ? `${(reading.character_error_rate * 100).toFixed(1)}%` : '—'}</td></tr>`).join('')}</tbody></table></div></div>`;
    }).join('');
    $('comparisonResults').hidden = !result.detections.length;
    $('comparisonFooter').hidden = !result.detections.length;
  }

  async function compareOCR() {
    if (state.inFlight || state.preparing) return;
    haltAnalysis();
    if (state.mode !== 'upload') video.pause();
    const generation = state.generation;
    const blob = state.mode === 'upload' ? state.image : await frameBlob();
    if (!blob || generation !== state.generation) { updateButtons(); return; }
    const methods = state.methods.filter(method => method.available).map(method => method.id);
    if (methods.length < 2) { feedback('Install at least two OCR methods using Setup OCR.ps1, then restart the app.', 'warning'); return; }
    state.comparison = null;
    state.inFlight = true;
    const controller = new AbortController();
    state.controller = controller;
    const timeout = setTimeout(() => controller.abort(), 300000);
    $('comparisonStatus').hidden = false;
    $('comparisonStatus').textContent = 'Preparing engines and comparing the same plate crops…';
    $('comparisonResults').hidden = true;
    $('comparisonFooter').hidden = true;
    $('viewerSubtitle').textContent = 'Comparing OCR…';
    updateButtons();
    try {
      const data = new FormData();
      data.append('file', blob, 'comparison.jpg');
      data.append('confidence', String(Number($('confidence').value) / 100));
      data.append('expected_text', $('expectedText').value);
      methods.forEach(method => data.append('methods', method));
      const result = await api('/api/compare', {method:'POST', body:data, signal:controller.signal});
      if (generation !== state.generation) return;
      renderComparison(result);
      $('viewerSubtitle').textContent = 'OCR comparison complete';
    } catch (error) {
      if (generation !== state.generation) return;
      $('comparisonStatus').textContent = error.name === 'AbortError' ? 'Comparison timed out. Try again after preparing the models.' : error.message;
      $('viewerSubtitle').textContent = 'Comparison paused';
    } finally {
      clearTimeout(timeout);
      if (generation === state.generation) { state.inFlight = false; state.controller = null; updateButtons(); }
    }
  }

  function exportComparison() {
    const result = state.comparison;
    if (!result) return;
    const columns = ['compared_at','plate','ocr_method','expected_text','reading','raw_text','ocr_ms','confidence','exact_match','character_error_rate','detection_ms','model_setup_ms','crop_sha256','error'];
    const rows = result.detections.flatMap((plate,index) => plate.readings.map(reading => [result.compared_at,index+1,reading.ocr_method,result.expected_text || '',reading.text,reading.raw_text,reading.ocr_ms,reading.recognition_confidence,reading.exact_match,reading.character_error_rate,result.detection_ms,result.preparation_ms[reading.ocr_method],plate.crop_sha256,reading.error]));
    const cell = value => {
      let text = String(value ?? '');
      if (/^\s*[=+@-]/.test(text)) text = "'" + text;
      return '"' + text.replaceAll('"','""') + '"';
    };
    const csv = [columns,...rows].map(row => row.map(cell).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob(['\uFEFF',csv],{type:'text/csv;charset=utf-8'}));
    const link = document.createElement('a');
    link.href = url; link.download = 'platesight-ocr-comparison.csv'; link.click();
    setTimeout(() => URL.revokeObjectURL(url),1000);
  }

  $('compareBtn').addEventListener('click', compareOCR);
  $('exportComparisonBtn').addEventListener('click', exportComparison);
  $('ocrMethod').addEventListener('change', async () => {
    haltAnalysis();
    if (state.mode !== 'upload') video.pause();
    state.ocrMethod = $('ocrMethod').value;
    state.session = crypto.randomUUID();
    clearResults();
    feedback('');
    await warmup();
    if (state.ready) {
      $('viewerSubtitle').textContent = 'OCR method changed';
      feedback(state.mode === 'upload' && state.image ? 'OCR method changed. Select Analyze to read this image.' : 'OCR method changed. Start or resume recognition when ready.');
    }
  });
  video.addEventListener('loadeddata', updateButtons);
  document.querySelectorAll('.input-tab').forEach((tab,index,tabs) => {
    tab.addEventListener('click',() => setMode(tab.dataset.mode));
    tab.addEventListener('keydown',(event) => {
      let next;
      if (event.key === 'ArrowRight') next = (index+1)%tabs.length;
      if (event.key === 'ArrowLeft') next = (index+tabs.length-1)%tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length-1;
      if (next !== undefined) { event.preventDefault(); tabs[next].focus(); setMode(tabs[next].dataset.mode); }
    });
  });
  $('chooseFileBtn').addEventListener('click',() => state.mode === 'camera' ? startCamera() : $(state.mode === 'video' ? 'videoInput' : 'imageInput').click());
  $('replaceBtn').addEventListener('click',() => $('imageInput').click());
  $('replaceVideoBtn').addEventListener('click',() => $('videoInput').click());
  $('imageInput').addEventListener('change',(event) => { const file = event.target.files[0]; if (file) loadImage(file); event.target.value = ''; });
  $('videoInput').addEventListener('change',(event) => { const file = event.target.files[0]; if (file) loadVideo(file); event.target.value = ''; });
  $('reanalyzeBtn').addEventListener('click',() => state.image && analyzeBlob(state.image,'upload'));
  $('warmupBtn').addEventListener('click',warmup);
  $('cameraStartBtn').addEventListener('click',startCamera);
  $('cameraSelect').addEventListener('change',() => { if (state.stream) startCamera(); });
  $('pauseBtn').addEventListener('click',toggleCameraAnalysis);
  $('stopBtn').addEventListener('click',stopCamera);
  $('snapshotBtn').addEventListener('click',snapshot);
  $('videoPlayBtn').addEventListener('click',toggleVideo);
  $('videoStopBtn').addEventListener('click',() => { haltAnalysis(); video.pause(); video.currentTime = 0; clearResults(); $('viewerSubtitle').textContent = 'Video stopped'; updateButtons(); });
  $('confidence').addEventListener('input',() => { $('confidenceValue').textContent = `${$('confidence').value}%`; });
  $('confidence').addEventListener('change',() => { if (state.mode === 'upload' && state.image) feedback('Threshold updated. Select Analyze to apply it to this image.'); });
  video.addEventListener('ended',() => { if (state.mode === 'video') { haltAnalysis(); $('viewerSubtitle').textContent = 'Playback complete'; updateButtons(); } });
  video.addEventListener('error',() => { if (state.mode === 'video' && state.videoUrl) { haltAnalysis(); feedback('This video format could not be played. Try an MP4 video supported by your browser.', 'error'); updateButtons(); } });
  stage.addEventListener('dragover',(event) => { if (state.mode === 'camera') return; event.preventDefault(); stage.classList.add('dragover'); });
  stage.addEventListener('dragleave',() => stage.classList.remove('dragover'));
  stage.addEventListener('drop',(event) => {
    event.preventDefault(); stage.classList.remove('dragover');
    if (state.mode === 'camera') return;
    const file = event.dataTransfer.files[0];
    if (file) state.mode === 'video' ? loadVideo(file) : loadImage(file);
  });
  new ResizeObserver(drawOverlay).observe(stage);
  const staleInterval = setInterval(() => { if (state.lastResult && (state.mode === 'camera' || state.mode === 'video')) drawOverlay(); },250);
  window.addEventListener('pagehide',() => {
    stopMedia();
    clearInterval(staleInterval);
    if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  });
  document.addEventListener('visibilitychange',() => {
    if (document.hidden && state.stream) { stopCamera(); feedback('Camera stopped when you left this tab. Start it again to continue.'); }
    else if (document.hidden && state.mode === 'video' && state.live) { haltAnalysis(); video.pause(); $('viewerSubtitle').textContent = 'Video paused'; updateButtons(); }
  });

  refreshStatus();
})();
