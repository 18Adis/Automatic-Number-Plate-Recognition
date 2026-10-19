(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = {items: [], page: 0, pageSize: 25, loading: false, editId: null, capture: null, view: 'plate', enhanced: false};
  const escape = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
  const imagePath = value => typeof value === 'string' && /^\/api\/history\/[a-f0-9]{32}\/(image|thumbnail)$/.test(value) ? value : null;
  const date = value => {
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? 'Unknown date' : parsed.toLocaleString([], {day:'2-digit', month:'short', year:'numeric', hour:'2-digit', minute:'2-digit'});
  };
  const percent = value => Number.isFinite(Number(value)) ? `${Math.round(Number(value) * 100)}%` : '—';
  const source = value => ({upload:'Image upload', snapshot:'Snapshot', live:'Live camera', video:'Video'}[value] || 'Image');

  async function request(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(typeof body.detail === 'string' ? body.detail : 'Unable to complete this request. Please try again.');
    }
    return response.json();
  }

  function render() {
    const query = $('search').value.replace(/[^a-z0-9]/gi, '').toUpperCase();
    const items = state.items.filter(item => String(item.text).toUpperCase().includes(query));
    const pages = Math.ceil(items.length / state.pageSize);
    state.page = Math.max(0, Math.min(state.page, pages - 1));
    const start = state.page * state.pageSize;
    $('totalLabel').textContent = `${state.items.length === 1000 ? 'Latest ' : ''}${state.items.length.toLocaleString()} saved ${state.items.length === 1 ? 'detection' : 'detections'}`;
    $('clearHistoryBtn').disabled = !state.items.length || state.loading;
    $('previousBtn').disabled = state.page === 0;
    $('nextBtn').disabled = state.page + 1 >= pages;
    $('pageLabel').textContent = items.length ? `${start + 1}–${Math.min(start + state.pageSize, items.length)} of ${items.length}` : '0 detections';
    $('historyBody').innerHTML = items.length ? items.slice(start, start + state.pageSize).map(item => {
      const thumbnail = imagePath(item.thumbnail_url);
      return `<tr><td>${thumbnail ? `<button class="capture-thumb" data-image="${escape(item.id)}" aria-label="View image for ${escape(item.text)}"><img src="${escape(thumbnail)}" alt="Plate ${escape(item.text)}" loading="lazy"><span>View image ↗</span></button>` : '<span class="no-capture">No saved image</span>'}</td><td><button class="edit-plate" data-edit="${escape(item.id)}" aria-label="Correct plate ${escape(item.text)}">${escape(item.text)}<svg aria-hidden="true"><use href="#i-edit"/></svg></button></td><td>${escape(date(item.timestamp))}</td><td>${escape(source(item.source))}</td><td>${escape(({cct:"Plate OCR · CCT",easyocr:"EasyOCR",tesseract:"Tesseract",paddleocr:"PaddleOCR"})[item.ocr_method] || item.ocr_method || "Plate OCR · CCT")}</td><td>${percent(item.detection_confidence)}</td><td>${percent(item.recognition_confidence)}</td><td><span class="status-label ${item.status === 'confirmed' ? '' : 'review'}">${item.status === 'confirmed' ? 'Confirmed' : 'Needs review'}</span></td></tr>`;
    }).join('') : `<tr><td colspan="8"><div class="history-empty">${query ? 'No matching plate numbers.' : 'No saved detections yet.'}</div></td></tr>`;
  }

  async function refresh() {
    if (state.loading) return;
    state.loading = true;
    $('refreshBtn').disabled = true;
    $('historyError').hidden = true;
    try {
      state.items = (await request('/api/history')).items;
      render();
    } catch (error) {
      $('historyError').textContent = error.message;
      $('historyError').hidden = false;
      if (!state.items.length) {
        $('totalLabel').textContent = 'History unavailable';
        $('historyBody').innerHTML = '<tr><td colspan="8"><div class="history-empty">Unable to load history. Select Refresh to try again.</div></td></tr>';
      }
    } finally {
      state.loading = false;
      $('refreshBtn').disabled = false;
      $('clearHistoryBtn').disabled = !state.items.length;
    }
  }

  function applyZoom() {
    $('reviewImage').style.width = `${$('zoom').value}%`;
    $('zoomValue').textContent = `${$('zoom').value}%`;
  }

  function setView(view) {
    state.view = view;
    const url = imagePath(view === 'plate' ? state.capture.thumbnail_url : state.capture.image_url);
    $('plateViewBtn').setAttribute('aria-pressed', String(view === 'plate'));
    $('frameViewBtn').setAttribute('aria-pressed', String(view === 'frame'));
    $('plateViewBtn').classList.toggle('selected', view === 'plate');
    $('frameViewBtn').classList.toggle('selected', view === 'frame');
    $('zoom').value = '100';
    applyZoom();
    $('imageViewport').scrollTo(0, 0);
    $('reviewImage').hidden = !url;
    $('imageError').hidden = Boolean(url);
    $('reviewImage').alt = `${view === 'plate' ? 'Plate crop' : 'Vehicle image'} for ${state.capture.text}`;
    if (url) $('reviewImage').src = url;
    else $('reviewImage').removeAttribute('src');
  }

  function openImage(identity) {
    state.capture = state.items.find(item => item.id === identity);
    if (!state.capture) return;
    $('captureTitle').textContent = state.capture.text;
    $('captureTime').textContent = `${date(state.capture.timestamp)} · ${source(state.capture.source)}`;
    state.enhanced = false;
    $('reviewImage').classList.remove('enhanced');
    $('contrastBtn').setAttribute('aria-pressed', 'false');
    setView('plate');
    $('imageDialog').showModal();
  }

  function openEdit(identity) {
    const item = state.items.find(item => item.id === identity);
    if (!item) return;
    state.editId = identity;
    $('editText').value = item.text;
    $('editError').hidden = true;
    $('editDialog').showModal();
    $('editText').focus();
    $('editText').select();
  }

  $('historyBody').addEventListener('click', event => {
    const image = event.target.closest('[data-image]');
    const edit = event.target.closest('[data-edit]');
    if (image) openImage(image.dataset.image);
    else if (edit) openEdit(edit.dataset.edit);
  });
  $('historyBody').addEventListener('error', event => {
    if (event.target.tagName === 'IMG') {
      const button = event.target.closest('.capture-thumb');
      if (button) button.innerHTML = '<span>Image unavailable</span>';
    }
  }, true);
  $('reviewImage').addEventListener('error', () => { $('reviewImage').hidden = true; $('imageError').hidden = false; });
  $('search').addEventListener('input', () => { state.page = 0; render(); });
  $('refreshBtn').addEventListener('click', refresh);
  $('previousBtn').addEventListener('click', () => { state.page--; render(); });
  $('nextBtn').addEventListener('click', () => { state.page++; render(); });
  $('closeImageBtn').addEventListener('click', () => $('imageDialog').close());
  $('plateViewBtn').addEventListener('click', () => setView('plate'));
  $('frameViewBtn').addEventListener('click', () => setView('frame'));
  $('zoom').addEventListener('input', applyZoom);
  $('contrastBtn').addEventListener('click', () => {
    state.enhanced = !state.enhanced;
    $('reviewImage').classList.toggle('enhanced', state.enhanced);
    $('contrastBtn').setAttribute('aria-pressed', String(state.enhanced));
  });
  for (const id of ['cancelEditBtn', 'cancelEditBottomBtn']) $(id).addEventListener('click', () => $('editDialog').close());
  $('editForm').addEventListener('submit', async event => {
    event.preventDefault();
    $('saveEditBtn').disabled = true;
    $('editError').hidden = true;
    try {
      await request(`/api/history/${encodeURIComponent(state.editId)}`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text:$('editText').value})});
      $('editDialog').close();
      await refresh();
    } catch (error) { $('editError').textContent = error.message; $('editError').hidden = false; }
    finally { $('saveEditBtn').disabled = false; }
  });
  $('clearHistoryBtn').addEventListener('click', async () => {
    if (!window.confirm('Clear all saved readings and their images?')) return;
    $('clearHistoryBtn').disabled = true;
    try { await request('/api/history', {method:'DELETE'}); await refresh(); }
    catch (error) { $('historyError').textContent = error.message; $('historyError').hidden = false; $('clearHistoryBtn').disabled = false; }
  });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
  refresh();
})();
