const chat = document.getElementById("chat");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("sendBtn");
const plusBtn = document.getElementById("plusBtn");
const plusMenu = document.getElementById("plusMenu");
const fileInput = document.getElementById("fileInput");
const recordBtn = document.getElementById("recordBtn");
const docTypeEl = document.getElementById("docType");
const webSearchToggle = document.getElementById("webSearchToggle");
const ocrEngineEl = document.getElementById("ocrEngine");
const ocrPathEl = document.getElementById("ocrPath");
const fileNote = document.getElementById("fileNote");
const retrievalEl = document.getElementById("retrieval");
const retrievalWrap = document.getElementById("retrievalWrap");
const retrievalToggle = document.getElementById("retrievalToggle");
const retrievalDot = document.getElementById("retrievalDot");
const retrievalChevron = document.getElementById("retrievalChevron");
const searchWrap = document.getElementById("searchWrap");
const searchToggle = document.getElementById("searchToggle");
const searchDot = document.getElementById("searchDot");
const searchResultsEl = document.getElementById("searchResults");
const downloadsEl = document.getElementById("downloads");
const agentModeEl = document.getElementById("agentMode");
const traceStatus = document.getElementById("traceStatus");
const traceListEl = document.getElementById("traceList");
const moreWrap = document.getElementById("moreWrap");
const moreToggle = document.getElementById("moreToggle");
const connectBtn = document.getElementById("connectBtn");
const connStatus = document.getElementById("connStatus");
const siteRegionEl = document.getElementById("siteRegion");
const diagBtn = document.getElementById("diagBtn");
const diagOutput = document.getElementById("diagOutput");
const searchTargetEl = document.getElementById("searchTarget");
const progressFill = document.getElementById("progressFill");
const progressLabel = document.getElementById("progressLabel");
const planDetailBtn = document.getElementById("planDetailBtn");
const mindmapStatus = document.getElementById("mindmapStatus");
const planModal = document.getElementById("planModal");
const planClose = document.getElementById("planClose");
const planContent = document.getElementById("planContent");
const mindmapDetailModal = document.getElementById("mindmapDetailModal");
const mindmapDetailClose = document.getElementById("mindmapDetailClose");
const mindmapDetailBody = document.getElementById("mindmapDetailBody");
const errorsEl = document.getElementById("errors");
const docModal = document.getElementById("docModal");
const docModalCard = document.querySelector(".doc-modal");
const docClose = document.getElementById("docClose");
const docEditor = document.getElementById("docEditor");
const docGenBtn = document.getElementById("docGenBtn");
const docAiInput = document.getElementById("docAiInput");
const docAiBtn = document.getElementById("docAiBtn");
const docModalTitle = document.getElementById("docModalTitle");
const docPanelBtn = document.getElementById("docPanelBtn");
const newChatBtn = document.getElementById("newChatBtn");
const myChatsToggle = document.getElementById("myChatsToggle");
const myChatsWrap = document.getElementById("myChatsWrap");
const myChatsList = document.getElementById("myChatsList");
const currentChatTitle = document.getElementById("currentChatTitle");

const apiKeyEl = document.getElementById("apiKey");
const baseUrlEl = document.getElementById("baseUrl");
const modelEl = document.getElementById("model");
const topKEl = document.getElementById("topK");
const kwWeightEl = document.getElementById("kwWeight");
const vecWeightEl = document.getElementById("vecWeight");

let messages = [];
let fileText = "";
let fileAssets = [];
let mediaRecorder = null;
let audioChunks = [];
let streaming = false;
let traceItems = [];
let lastPlanDetail = "";
let documentsCache = [];
let downloadsCache = [];
let activeDocIndex = -1;
let sessionId = "";
let lastUserQuery = "";
let lastAssistantEl = null;
let forceWebSearch = false;
let lastMindmap = null;
let mindmapLoading = false;
let highlightKeywords = [];
let mindmapDetailEl = null;
let caseResultsCache = [];
let searchResultsCache = [];
let caseAnalysisCache = {};
let autoProgressTimer = null;
let autoProgressVal = 0;
let conversations = [];

const SESSION_STORE_KEY = "LAW_MASTER_CHAT_SESSIONS_V1";
const SESSION_ACTIVE_KEY = "LAW_MASTER_ACTIVE_SESSION_ID";
const SESSION_DEFAULT_TITLE = "\u65B0\u804A\u5929";

function escapeHtml(text) {
  return (text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function formatAssistantText(text) {
  const safe = escapeHtml(text || "").replace(/\n/g, "<br>");
  let out = safe;
  const casePattern = /([【\[])?案例\s*(\d+)([】\]])?/g;
  out = out.replace(
    casePattern,
    (_m, _pre, n) => `<a href="#" class="case-link" data-case="${n}">\u3010\u6848\u4f8b${n}\u3011</a>`
  );
  const webPattern = /([【\[])?网页\s*(\d+)([】\]])?/g;
  out = out.replace(
    webPattern,
    (_m, _pre, n) => `<a href="#" class="web-link" data-web="${n}">\u3010\u7f51\u9875${n}\u3011</a>`
  );
  return out;
}


function renderMessage(role, content) {
  const msg = document.createElement("div");
  msg.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") {
    bubble.innerHTML = formatAssistantText(content);
  } else {
    bubble.textContent = content;
  }
  msg.appendChild(bubble);
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
  if (role === "assistant") {
    lastAssistantEl = msg;
  }
  return msg;
}

function touchActiveConversation() {
  const item = conversations.find((c) => c.id === sessionId);
  if (!item) return;
  item.updatedAt = Date.now();
}

function addMessage(role, content) {
  messages.push({ role, content });
  touchActiveConversation();
  saveConversationStore();
  renderSessionList();
  renderMessage(role, content);
}

function addTyping() {
  const msg = document.createElement("div");
  msg.className = "message assistant";
  msg.id = "typing";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<div class="typing"><span></span><span></span><span></span></div>`;
  msg.appendChild(bubble);
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById("typing");
  if (t) t.remove();
}

function attachAssistantActions() {}

function setError(text) {
  errorsEl.textContent = text || "";
}

function cleanOcrText(text) {
  if (!text) return "";
  let t = text.replace(/\uFFFD/g, "");
  t = t.replace(/[|\\/]{3,}/g, "");
  t = t.replace(/[^\S\n]+/g, " ");
  for (let i = 0; i < 2; i++) {
    t = t.replace(/([\u4e00-\u9fff])\s+([\u4e00-\u9fff])/g, "$1$2");
  }
  const lines = t
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => {
      if (!l) return false;
      const useful = (l.match(/[\w\u4e00-\u9fff]/g) || []).length;
      return useful > 0 || l.length > 6;
    });
  return lines.join("\n").trim();
}

function genSessionId(prefix = "s") {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function clipSessionTitle(text) {
  const raw = (text || "").replace(/\s+/g, " ").trim();
  if (!raw) return SESSION_DEFAULT_TITLE;
  return raw.slice(0, 28);
}

function normalizeConversation(raw) {
  const now = Date.now();
  const item = raw || {};
  return {
    id: item.id || genSessionId(),
    title: clipSessionTitle(item.title || SESSION_DEFAULT_TITLE),
    manualTitle: !!item.manualTitle,
    favorite: !!item.favorite,
    createdAt: Number(item.createdAt) || now,
    updatedAt: Number(item.updatedAt) || now,
    messages: Array.isArray(item.messages) ? item.messages : [],
  };
}

function saveConversationStore() {
  try {
    localStorage.setItem(SESSION_STORE_KEY, JSON.stringify(conversations));
    localStorage.setItem(SESSION_ACTIVE_KEY, sessionId || "");
    localStorage.setItem("LAW_MASTER_SESSION_ID", sessionId || "");
  } catch (_e) {
    // Ignore quota errors to avoid breaking chat flow.
  }
}

function getSortedConversations() {
  return [...conversations].sort((a, b) => {
    if (!!a.favorite !== !!b.favorite) return a.favorite ? -1 : 1;
    return (b.updatedAt || 0) - (a.updatedAt || 0);
  });
}

function formatSessionTime(ts) {
  if (!ts) return "";
  try {
    return new Date(ts).toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

function renderCurrentSessionTitle() {
  if (!currentChatTitle) return;
  const current = conversations.find((c) => c.id === sessionId);
  currentChatTitle.textContent = current ? current.title : SESSION_DEFAULT_TITLE;
}

function renderSessionList() {
  if (!myChatsList) return;
  const sorted = getSortedConversations();
  if (!sorted.length) {
    myChatsList.innerHTML = '<div class="chat-empty">\u6682\u65E0\u4F1A\u8BDD</div>';
    return;
  }
  myChatsList.innerHTML = "";
  sorted.forEach((item) => {
    const row = document.createElement("div");
    row.className = `chat-session-item${item.id === sessionId ? " active" : ""}`;
    const starLabel = item.favorite ? "\u2605" : "\u2606";
    row.innerHTML = `
      <div class="chat-session-row">
        <button class="chat-session-main" data-action="switch" data-id="${item.id}">
          ${escapeHtml(item.title || SESSION_DEFAULT_TITLE)}
        </button>
        <div class="chat-session-actions">
          <button class="chat-mini-btn favorite ${item.favorite ? "on" : ""}" data-action="favorite" data-id="${item.id}" title="\u6536\u85CF">${starLabel}</button>
          <button class="chat-mini-btn" data-action="rename" data-id="${item.id}" title="\u91CD\u547D\u540D">\u91CD\u547D\u540D</button>
          <button class="chat-mini-btn" data-action="delete" data-id="${item.id}" title="\u5220\u9664">\u5220\u9664</button>
        </div>
      </div>
      <div class="chat-session-meta">${formatSessionTime(item.updatedAt)}</div>
    `;
    myChatsList.appendChild(row);
  });
}

function renderActiveConversation() {
  chat.innerHTML = "";
  lastAssistantEl = null;
  (messages || []).forEach((msg) => renderMessage(msg.role, msg.content));
}

function resetTransientPanels() {
  retrievalEl.textContent = "";
  searchResultsEl.textContent = "";
  downloadsEl.textContent = "";
  traceListEl.textContent = "";
  traceItems = [];
  caseAnalysisCache = {};
  caseResultsCache = [];
  searchResultsCache = [];
  documentsCache = [];
  downloadsCache = [];
  activeDocIndex = -1;
  highlightKeywords = [];
  lastPlanDetail = "";
  lastMindmap = null;
  mindmapLoading = false;
  autoProgressVal = 0;
  stopAutoProgress();
  setProgress(0);
  setDot(retrievalDot, false);
  setDot(searchDot, false);
  setStatus("待命", false);
  if (mindmapStatus) {
    mindmapStatus.className = "mindmap-status idle";
    mindmapStatus.textContent = "思维导图：待生成";
  }
  fileText = "";
  fileAssets = [];
  if (fileNote) fileNote.textContent = "";
  if (promptEl) promptEl.value = "";
  setError("");
}

function setActiveSession(nextSessionId) {
  if (streaming) {
    setError("\u8BF7\u7B49\u5F85\u5F53\u524D\u56DE\u7B54\u5B8C\u6210\u540E\u518D\u5207\u6362\u4F1A\u8BDD");
    return false;
  }
  const target = conversations.find((c) => c.id === nextSessionId);
  if (!target) return false;
  sessionId = target.id;
  messages = target.messages;
  const lastUser = [...messages].reverse().find((m) => m && m.role === "user");
  lastUserQuery = (lastUser && lastUser.content) || "";
  renderActiveConversation();
  renderCurrentSessionTitle();
  renderSessionList();
  saveConversationStore();
  resetTransientPanels();
  return true;
}

function createConversation(seed = {}) {
  const item = normalizeConversation({
    id: seed.id || genSessionId(),
    title: seed.title || SESSION_DEFAULT_TITLE,
    manualTitle: !!seed.manualTitle,
    favorite: !!seed.favorite,
    createdAt: seed.createdAt || Date.now(),
    updatedAt: seed.updatedAt || Date.now(),
    messages: Array.isArray(seed.messages) ? seed.messages : [],
  });
  conversations.push(item);
  return item;
}

function createNewSession() {
  if (streaming) {
    setError("\u8BF7\u7B49\u5F85\u5F53\u524D\u56DE\u7B54\u5B8C\u6210\u540E\u518D\u65B0\u5EFA\u4F1A\u8BDD");
    return;
  }
  const item = createConversation();
  setActiveSession(item.id);
}

function renameConversation(id) {
  const item = conversations.find((c) => c.id === id);
  if (!item) return;
  const next = window.prompt("\u8BF7\u8F93\u5165\u4F1A\u8BDD\u540D\u79F0", item.title || SESSION_DEFAULT_TITLE);
  if (next == null) return;
  const clean = clipSessionTitle(next);
  item.title = clean || SESSION_DEFAULT_TITLE;
  item.manualTitle = true;
  item.updatedAt = Date.now();
  if (item.id === sessionId) {
    renderCurrentSessionTitle();
  }
  renderSessionList();
  saveConversationStore();
}

function toggleFavoriteConversation(id) {
  const item = conversations.find((c) => c.id === id);
  if (!item) return;
  item.favorite = !item.favorite;
  item.updatedAt = Date.now();
  renderSessionList();
  saveConversationStore();
}

function deleteConversation(id) {
  const idx = conversations.findIndex((c) => c.id === id);
  if (idx < 0) return;
  const item = conversations[idx];
  const ok = window.confirm(`确认删除会话“${item.title || SESSION_DEFAULT_TITLE}”？`);
  if (!ok) return;

  const deletingActive = item.id === sessionId;
  conversations.splice(idx, 1);

  if (!conversations.length) {
    const created = createConversation();
    sessionId = created.id;
    messages = created.messages;
    lastUserQuery = "";
  } else if (deletingActive) {
    const next = getSortedConversations()[0];
    sessionId = next.id;
    messages = next.messages;
    const lastUser = [...messages].reverse().find((m) => m && m.role === "user");
    lastUserQuery = (lastUser && lastUser.content) || "";
  }

  renderActiveConversation();
  renderCurrentSessionTitle();
  renderSessionList();
  saveConversationStore();
  resetTransientPanels();
}

function initSessionStore() {
  let loaded = [];
  try {
    loaded = JSON.parse(localStorage.getItem(SESSION_STORE_KEY) || "[]");
  } catch {
    loaded = [];
  }
  conversations = Array.isArray(loaded) ? loaded.map((c) => normalizeConversation(c)) : [];

  const legacyId = localStorage.getItem("LAW_MASTER_SESSION_ID");
  if (!conversations.length) {
    createConversation({ id: legacyId || genSessionId() });
  }

  const active = localStorage.getItem(SESSION_ACTIVE_KEY) || legacyId || conversations[0].id;
  const activeItem = conversations.find((c) => c.id === active) || conversations[0];
  sessionId = activeItem.id;
  messages = activeItem.messages;
  const lastUser = [...messages].reverse().find((m) => m && m.role === "user");
  lastUserQuery = (lastUser && lastUser.content) || "";
  renderActiveConversation();
  renderCurrentSessionTitle();
  renderSessionList();
  saveConversationStore();
  resetTransientPanels();
}

function setRetrievalFromResults(lawResults, caseResults, keywords) {
  const wrap = document.createElement("div");
  wrap.className = "retrieval-list";
  const kw = document.createElement("div");
  kw.className = "mono";
  kw.textContent = `检索关键词: ${keywords && keywords.length ? keywords.join(", ") : "(无)"}`;
  wrap.appendChild(kw);
  caseResultsCache = (caseResults || []).map((r, idx) => ({ ...r, case_no: idx + 1 }));

  const highlightText = (text) => {
    if (!text) return "";
    let out = escapeHtml(text);
    (keywords || []).forEach((k) => {
      if (!k) return;
      const re = new RegExp(escapeRegExp(escapeHtml(k)), "g");
      out = out.replace(re, (m) => `<span class="hl">${m}</span>`);
    });
    return out;
  };

  const buildEntry = (r, fallbackTitle, index = 0, isCase = false) => {
    const card = document.createElement("div");
    card.className = "entry";
    const title = r.article_number || r.title || fallbackTitle || "条目";
    const metaParts = [];
    if (typeof r.score === "number") {
      metaParts.push(`评分 ${r.score.toFixed(3)}`);
    }
    if (typeof r.score_keyword === "number") {
      metaParts.push(`关键词 ${r.score_keyword.toFixed(3)}`);
    }
    if (typeof r.score_vector === "number") {
      metaParts.push(`语义 ${r.score_vector.toFixed(3)}`);
    }
    if (r.source) metaParts.push(`来源 ${r.source}`);
    if (r.page_start || r.page_end) {
      metaParts.push(`页码 ${r.page_start || ""}-${r.page_end || ""}`.replace(/-$/, ""));
    }
    const bodyText = isCase ? (r.summary || r.content || "") : (r.content || "");
    const safeTitle = highlightText(title);
    const safeBody = highlightText((bodyText || "").trim());
    const prefix = isCase && index ? `<span class="case-no">案例${index}</span> ` : "";
    card.innerHTML = `
      <div class="entry-title">${prefix}${safeTitle}</div>
      <div class="entry-meta">${metaParts.join(" | ")}</div>
      <div class="entry-body">
        <div class="entry-content">${safeBody}</div>
        ${isCase ? '<div class="entry-actions"><button class="btn-light entry-preview">案例预览</button></div>' : ""}
      </div>
    `;
    if (isCase && index) {
      card.dataset.caseNo = String(index);
    }
    card.addEventListener("click", () => card.classList.toggle("open"));
    if (isCase) {
      const btn = card.querySelector(".entry-preview");
      if (btn) {
        btn.addEventListener("click", (e) => {
          e.stopPropagation();
          openCasePreview(
            title,
            (r.content || "").trim(),
            r.source || "",
            r.page_start || "",
            r.page_end || "",
            index
          );
        });
      }
    }
    return card;
  };

  const buildGroup = (title, results, isCase = false) => {
    const group = document.createElement("div");
    group.className = "retrieval-group";
    group.dataset.group = isCase ? "case" : "law";
    const header = document.createElement("button");
    header.type = "button";
    header.className = "retrieval-group-toggle";
    header.innerHTML = `<span>${title}（${results.length}）</span><span class="chevron">▼</span>`;
    const body = document.createElement("div");
    body.className = "retrieval-group-body collapsed";
    if (!results.length) {
      const empty = document.createElement("div");
      empty.className = "mono muted";
      empty.textContent = "\u65E0\u7ED3\u679C";
      body.appendChild(empty);
    } else {
      results.forEach((r, idx) => body.appendChild(buildEntry(r, title, idx + 1, isCase)));
    }
    header.addEventListener("click", () => {
      body.classList.toggle("collapsed");
      header.classList.toggle("open");
    });
    group.appendChild(header);
    group.appendChild(body);
    return group;
  };

  wrap.appendChild(buildGroup("法条搜索", lawResults || [], false));
  wrap.appendChild(buildGroup("案例搜索", caseResultsCache || [], true));

  retrievalEl.innerHTML = "";
  retrievalEl.appendChild(wrap);
  if (retrievalWrap.classList.contains("collapsed")) {
    retrievalWrap.classList.remove("collapsed");
    retrievalToggle.classList.add("open");
  }
  if (caseResultsCache.length) {
    setTimeout(prefetchCaseAnalyses, 80);
  }
}

function highlight(text, keywords) {
  if (!keywords || !keywords.length) return text;
  let result = text;
  keywords.forEach((k) => {
    if (!k) return;
    const re = new RegExp(k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
    result = result.replace(re, (m) => `<span class="hl">${m}</span>`);
  });
  return result;
}

function setSearchResults(results, keywords) {
  searchResultsCache = (results || []).map((r, idx) => ({ ...r, web_no: idx + 1 }));
  const wrap = document.createElement("div");
  wrap.className = "search-list";
  searchResultsCache.forEach((r) => {
    const item = document.createElement("div");
    item.className = "search-item";
    if (r.web_no) item.dataset.webNo = String(r.web_no);
    const title = r.title || "\u7f51\u9875\u7ed3\u679c";
    const url = r.url || "#";
    const snippet = r.snippet || "";
    const content = r.content || "";
    item.innerHTML = `
      <div class="search-title"><span class="web-no">\u7f51\u9875${r.web_no || ""}</span><a href="${url}" target="_blank" rel="noopener noreferrer">${title}</a></div>
      <div class="search-snippet">${highlight(snippet, keywords)}</div>
      ${content ? `<div class="search-snippet">${highlight(content, keywords)}</div>` : ""}
    `;
    wrap.appendChild(item);
  });
  searchResultsEl.innerHTML = "";
  searchResultsEl.appendChild(wrap);
  if (searchWrap.classList.contains("collapsed")) {
    searchWrap.classList.remove("collapsed");
    searchToggle.classList.add("open");
  }
}


function highlightHtml(text, keywords) {
  if (!text) return "";
  if (!keywords || !keywords.length) return text;
  let out = text;
  keywords.forEach((k) => {
    if (!k) return;
    const re = new RegExp(k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
    out = out.replace(re, (m) => `<mark class="kw">${m}</mark>`);
  });
  return out;
}

function addTrace(label, detail) {
  traceItems.push({ label, detail });
  const wrap = document.createElement("div");
  wrap.className = "trace-list";
  traceItems.forEach((t) => {
    const item = document.createElement("div");
    item.className = "trace-item";
    item.innerHTML = `<div class="label">${t.label}</div><div class="detail">${t.detail}</div>`;
    wrap.appendChild(item);
  });
  traceListEl.innerHTML = "";
  traceListEl.appendChild(wrap);
}

function setStatus(label, done = false) {
  if (!traceStatus) return;
  traceStatus.classList.toggle("done", !!done);
  traceStatus.innerHTML = done
    ? `<span class="pulse"></span>${label}`
    : `<span class="pulse"></span>${label}`;
}

function scrollToCase(caseNo) {
  if (!caseNo) return;
  if (retrievalWrap && retrievalWrap.classList.contains("collapsed")) {
    retrievalWrap.classList.remove("collapsed");
    retrievalToggle.classList.add("open");
  }
  const group = retrievalEl.querySelector('.retrieval-group[data-group="case"]');
  if (!group) return;
  const header = group.querySelector(".retrieval-group-toggle");
  const body = group.querySelector(".retrieval-group-body");
  if (body && body.classList.contains("collapsed")) {
    body.classList.remove("collapsed");
    if (header) header.classList.add("open");
  }
  const entry = group.querySelector(`.entry[data-case-no="${caseNo}"]`);
  if (entry) {
    entry.scrollIntoView({ behavior: "smooth", block: "center" });
    const title = entry.querySelector(".entry-title");
    if (title) {
      title.classList.add("case-flash");
      setTimeout(() => title.classList.remove("case-flash"), 2500);
    }
  }
}


function scrollToWeb(webNo) {
  if (!webNo) return;
  if (searchWrap && searchWrap.classList.contains("collapsed")) {
    searchWrap.classList.remove("collapsed");
    searchToggle.classList.add("open");
  }
  const entry = searchResultsEl.querySelector(`.search-item[data-web-no="${webNo}"]`);
  if (entry) {
    entry.scrollIntoView({ behavior: "smooth", block: "center" });
    const title = entry.querySelector(".search-title");
    if (title) {
      title.classList.add("web-flash");
      setTimeout(() => title.classList.remove("web-flash"), 2500);
    }
  }
}

function setProgress(pct) {
  const p = Math.max(0, Math.min(100, pct));
  if (progressFill) progressFill.style.width = `${p}%`;
  if (progressLabel) progressLabel.textContent = `进度 ${p}%`;
}


function setDot(dotEl, on) {
  if (!dotEl) return;
  dotEl.classList.toggle("show", !!on);
  dotEl.classList.toggle("blink", !!on);
}

function startAutoProgress() {
  stopAutoProgress();
  autoProgressVal = Math.max(autoProgressVal, 5);
  setProgress(autoProgressVal);
  autoProgressTimer = setInterval(() => {
    if (autoProgressVal < 40) {
      autoProgressVal += 3;
    } else if (autoProgressVal < 70) {
      autoProgressVal += 2;
    } else if (autoProgressVal < 88) {
      autoProgressVal += 1;
    }
    autoProgressVal = Math.min(autoProgressVal, 88);
    setProgress(autoProgressVal);
  }, 700);
}

function stopAutoProgress() {
  if (autoProgressTimer) {
    clearInterval(autoProgressTimer);
    autoProgressTimer = null;
  }
}


function initApiKeyFromStorage() {
  const stored = localStorage.getItem("DEEPSEEK_API_KEY") || "";
  if (stored && apiKeyEl) apiKeyEl.value = stored;
  const region = localStorage.getItem("SITE_REGION") || "cn_priority";
  if (siteRegionEl) siteRegionEl.value = region;
  const docType = localStorage.getItem("DOC_TYPE") || "通用文书";
  if (docTypeEl) docTypeEl.value = docType;
  const ocrEngine = localStorage.getItem("OCR_ENGINE") || "browser";
  if (ocrEngineEl) ocrEngineEl.value = ocrEngine;
}

async function loadOcrInfo() {
  if (!ocrPathEl) return;
  try {
    if (ocrEngineEl && ocrEngineEl.value === "browser") {
      ocrPathEl.value = "Browser OCR (Tesseract.js)";
      return;
    }
    const resp = await fetch("/api/ocr_info");
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "OCR \u68C0\u6D4B\u5931\u8D25");
    if (data.tesseract_cmd) {
      ocrPathEl.value = data.tesseract_cmd;
    } else {
      ocrPathEl.value = "未检测到 tesseract";
    }
  } catch (e) {
    ocrPathEl.value = "OCR \u68C0\u6D4B\u5931\u8D25";
  }
}

if (apiKeyEl) {
  apiKeyEl.addEventListener("change", () => {
    localStorage.setItem("DEEPSEEK_API_KEY", apiKeyEl.value.trim());
  });
}

if (connectBtn) {
  connectBtn.addEventListener("click", () => {
    const key = apiKeyEl.value.trim();
    if (key) {
      localStorage.setItem("DEEPSEEK_API_KEY", key);
      connStatus.innerHTML = '<span class="pulse"></span>\u5DF2\u8FDE\u63A5';
    } else {
      localStorage.removeItem("DEEPSEEK_API_KEY");
      connStatus.innerHTML = '<span class="pulse"></span>使用环境变量';
    }
    if (siteRegionEl) {
      localStorage.setItem("SITE_REGION", siteRegionEl.value);
    }
  });
}

if (docTypeEl) {
  docTypeEl.addEventListener("change", () => {
    localStorage.setItem("DOC_TYPE", docTypeEl.value);
  });
}

if (ocrEngineEl) {
  ocrEngineEl.addEventListener("change", () => {
    localStorage.setItem("OCR_ENGINE", ocrEngineEl.value);
    if (ocrPathEl) {
      if (ocrEngineEl.value === "browser") {
        ocrPathEl.value = "Browser OCR (Tesseract.js)";
      } else if (ocrEngineEl.value === "paddle") {
        ocrPathEl.value = ocrPathEl.value || "PaddleOCR";
      }
    }
  });
}

if (webSearchToggle) {
  webSearchToggle.addEventListener("click", () => {
    forceWebSearch = !forceWebSearch;
    webSearchToggle.classList.toggle("active", forceWebSearch);
  });
}

if (agentModeEl) {
  agentModeEl.addEventListener("change", () => {
    if (agentModeEl.checked) {
      forceWebSearch = true;
      if (webSearchToggle) webSearchToggle.classList.add("active");
    } else {
      forceWebSearch = false;
      if (webSearchToggle) webSearchToggle.classList.remove("active");
    }
  });
}

function setDownloads(files, documents) {
  downloadsCache = files || [];
  if (documents && documents.length) {
    documentsCache = documents;
  }
  if (!downloadsCache || !downloadsCache.length) {
    downloadsEl.textContent = "";
    return;
  }
  downloadsEl.innerHTML = "";
  downloadsCache.forEach((f, idx) => {
    const row = document.createElement("div");
    row.className = "doc-row";
    const a = document.createElement("a");
    a.href = f.url;
    a.textContent = `${f.title || "文档"} - 下载`;
    a.target = "_blank";
    const preview = document.createElement("button");
    preview.className = "btn-light";
    preview.textContent = "预览";
    preview.addEventListener("click", () => openDocPreview(idx));
    row.appendChild(a);
    row.appendChild(preview);
    downloadsEl.appendChild(row);
  });
}

function togglePlusMenu() {
  plusMenu.classList.toggle("hidden");
}

plusBtn.addEventListener("click", togglePlusMenu);

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  fileNote.textContent = "正在解析文件...";
  fileNote.classList.add("loading");
  const form = new FormData();
  form.append("file", file);
  let ocrEngine = ocrEngineEl ? ocrEngineEl.value : "browser";
  if (ocrEngine !== "browser") {
    ocrEngine = "browser";
    if (ocrEngineEl) ocrEngineEl.value = "browser";
    localStorage.setItem("OCR_ENGINE", "browser");
  }
  form.append("ocr_engine", ocrEngine);
  try {
    if (ocrEngine === "browser") {
      setError("");
    }
    let ocrText = "";
    if (ocrEngine === "browser") {
      if (typeof Tesseract === "undefined") {
        throw new Error("\u6D4F\u89C8\u5668 OCR \u672A\u52A0\u8F7D\uFF0C\u8BF7\u68C0\u67E5\u7F51\u7EDC");
      }
      const startOcrProgress = () => {
        let p = 0;
        fileNote.textContent = "浏览器 OCR 中... 0%";
        const timer = setInterval(() => {
          p = Math.min(90, p + Math.random() * 8 + 3);
          fileNote.textContent = `浏览器 OCR 中... ${Math.round(p)}%`;
        }, 300);
        return () => {
          clearInterval(timer);
          fileNote.textContent = "浏览器 OCR 中... 100%";
        };
      };
      const stopProgress = startOcrProgress();
      const assetBase = "/assets/tesseract";
      let resultText = "";
      try {
        const options = {
          workerPath: `${assetBase}/worker.min.js`,
          langPath: `${assetBase}/lang-data`,
          corePath: `${assetBase}/tesseract-core-simd.wasm.js`,
          gzip: true,
        };
        // Use recognize() directly to avoid createWorker signature mismatch errors
        const res = await Tesseract.recognize(file, "chi_sim", options);
        resultText = (res && res.data && res.data.text) || "";
      } catch (err) {
        // fallback without options
        const res = await Tesseract.recognize(file, "chi_sim");
        resultText = (res && res.data && res.data.text) || "";
      } finally {
        stopProgress();
      }
      ocrText = cleanOcrText(resultText);
    }

    let data = {};
    try {
      const resp = await fetch("/api/parse_file", { method: "POST", body: form });
      data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || "文件解析失败");
      }
    } catch (err) {
      // ignore backend OCR errors in browser mode
      if (ocrEngine !== "browser") throw err;
    }
    fileText = cleanOcrText(ocrText || data.text || "");
    if (data.asset_id) {
      fileAssets.push(data.asset_id);
    }
    const assetInfo = fileAssets.length ? ` | 图片数 ${fileAssets.length}` : "";
    const engineInfo = ocrEngine ? ` | OCR:${ocrEngine}` : "";
    fileNote.textContent = (data.note || "\u56FE\u7247\u5DF2\u4E0A\u4F20") + assetInfo + engineInfo;
    if (ocrEngine === "browser") {
      setError("");
    }
  } catch (e) {
    const msg = String(e && e.message ? e.message : e);
    if (ocrEngine === "browser" && /PaddleOCR|OCR失败|无法导入|paddle/i.test(msg)) {
      fileNote.textContent = "\u56FE\u7247\u5DF2\u4E0A\u4F20";
    } else {
      setError(msg);
      fileNote.textContent = "文件解析失败";
    }
  } finally {
    fileNote.classList.remove("loading");
  }
});

function openDocPreview(index) {
  activeDocIndex = index;
  const doc = documentsCache[index] || {};
  const title = doc.title || "文书";
  if (docModalCard) docModalCard.classList.remove("readonly");
  if (docModalTitle) docModalTitle.textContent = title;
  if (docEditor) {
    docEditor.textContent = doc.body || "";
    docEditor.contentEditable = "true";
  }
  if (docModal) docModal.classList.remove("hidden");
}

function buildCaseSections(text) {
  const headings = [
    "案由",
    "法院",
    "\u5F53\u4E8B\u4EBA",
    "基本案情",
    "事实",
    "诉讼请求",
    "争议焦点",
    "裁判要旨",
    "裁判观点",
    "裁判理由",
    "法条引用",
    "审理过程",
    "审理经过",
    "裁判结果",
    "判决结果",
    "法律依据",
  ];
  const pattern = new RegExp(`(^|\\n)(${headings.join("|")})\\s*[:：]`, "g");
  const matches = [];
  let m;
  while ((m = pattern.exec(text)) !== null) {
    matches.push({
      title: m[2],
      rawIndex: m.index + (m[1] ? m[1].length : 0),
      start: pattern.lastIndex,
    });
  }
  if (!matches.length) {
    return [{ title: "正文", content: (text || "").trim() }];
  }
  const sections = [];
  for (let i = 0; i < matches.length; i++) {
    const cur = matches[i];
    const next = matches[i + 1];
    const end = next ? next.rawIndex : text.length;
    const slice = text.slice(cur.start, end).trim();
    if (!slice) continue;
    sections.push({ title: cur.title, content: slice });
  }
  return sections.length ? sections : [{ title: "正文", content: (text || "").trim() }];
}

function renderCasePreview(content) {
  const sections = buildCaseSections(content || "");
  return sections
    .map(
      (s) => `
        <div class="case-section">
          <div class="case-heading">${escapeHtml(s.title)}</div>
          <div class="case-text">${highlight(s.content || "", highlightKeywords)}</div>
        </div>
      `
    )
    .join("");
}

function renderCasePreviewHeader(source, pageStart, pageEnd) {
  const pageText = pageStart || pageEnd ? `页码 ${pageStart || ""}${pageEnd ? "-" + pageEnd : ""}` : "";
  const meta = [source ? `来源：${source}` : "", pageText].filter(Boolean).join(" | ");
  return `
    <div class="case-preview-header">
      <img class="case-preview-logo" src="/assets/logo.jpg" alt="LAW MASTER" />
      <div class="case-preview-meta">${escapeHtml(meta || "\u6765\u6E90\uFF1A\u672A\u77E5")}</div>
    </div>
  `;
}

function renderCaseAnalysisSection(text) {
  const loading = !text;
  return `
    <div class="case-analysis">
      <div class="case-analysis-title">相似点与可借鉴之处（AI分析）</div>
      <div class="case-analysis-body ${loading ? "loading" : ""}">${escapeHtml(text || "分析生成中...")}</div>
    </div>
  `;
}

async function fetchCaseAnalysis(caseItem) {
  const cacheKey = String(caseItem.case_no || caseItem.id || caseItem.title || "");
  if (caseAnalysisCache[cacheKey]) return caseAnalysisCache[cacheKey];
  try {
    const batch =
      caseResultsCache && caseResultsCache.length > 1
        ? caseResultsCache.slice(0, 3)
        : [caseItem];
    const payload = {
      api_key: apiKeyEl.value.trim() || undefined,
      base_url: baseUrlEl.value.trim(),
      model: modelEl.value.trim(),
      user_query: lastUserQuery || "",
      cases: batch.map((c) => ({
        case_no: c.case_no,
        title: c.title,
        content: c.content || "",
        source: c.source || "",
        page_start: c.page_start || "",
        page_end: c.page_end || "",
      })),
    };
    const resp = await fetch("/api/case_analysis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "\u5206\u6790\u5931\u8d25");
    const analyses = data.analyses || [];
    analyses.forEach((a) => {
      const key = String(a.case_no || a.title || "");
      if (key) caseAnalysisCache[key] = a.analysis || "\u672a\u751f\u6210\u5206\u6790\u3002";
    });
    const found =
      analyses.find((a) => String(a.case_no || "") === String(caseItem.case_no || "")) || analyses[0];
    const text = (found && found.analysis) || "\u672a\u751f\u6210\u5206\u6790\u3002";
    caseAnalysisCache[cacheKey] = text;
    return text;
  } catch (e) {
    return `\u5206\u6790\u5931\u8d25\uff1a${e.message || e}`;
  }
}

function prefetchCaseAnalyses() {
  if (!caseResultsCache || !caseResultsCache.length) return;
  const batch = caseResultsCache.slice(0, 3);
  if (batch[0]) fetchCaseAnalysis(batch[0]).catch(() => {});
  if (batch[1]) fetchCaseAnalysis(batch[1]).catch(() => {});
  if (batch[2]) fetchCaseAnalysis(batch[2]).catch(() => {});
}


function openCasePreview(title, content, source, pageStart, pageEnd, caseNo) {
  if (docModalCard) docModalCard.classList.add("readonly");
  if (docModalTitle) docModalTitle.textContent = title || "案例预览";
  if (docEditor) {
    const header = renderCasePreviewHeader(source, pageStart, pageEnd);
    const body = renderCasePreview(content || "");
    const analysis = renderCaseAnalysisSection("");
    docEditor.innerHTML = `${header}${body}${analysis}`;
    docEditor.contentEditable = "false";
  }
  if (docModal) docModal.classList.remove("hidden");
  const caseItem =
    caseResultsCache.find((c) => String(c.case_no) === String(caseNo)) ||
    caseResultsCache[caseNo - 1] ||
    { case_no: caseNo, title, content, source, page_start: pageStart, page_end: pageEnd };
  fetchCaseAnalysis(caseItem).then((text) => {
    const el = docEditor ? docEditor.querySelector(".case-analysis-body") : null;
    if (el) el.textContent = text;
  });
}

if (docClose) {
  docClose.addEventListener("click", () => {
    if (docModal) docModal.classList.add("hidden");
  });
}
if (docModal) {
  docModal.addEventListener("click", (e) => {
    if (e.target === docModal) docModal.classList.add("hidden");
  });
}

recordBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    recordBtn.textContent = "语音录入";
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      const blob = new Blob(audioChunks, { type: "audio/webm" });
      const form = new FormData();
      form.append("file", blob, "voice.webm");
      try {
        const resp = await fetch("/api/transcribe", { method: "POST", body: form });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "语音识别失败");
        promptEl.value = (promptEl.value + " " + (data.text || "")).trim();
      } catch (e) {
        setError(e.message);
      }
    };
    mediaRecorder.start();
    recordBtn.textContent = "停止录音";
  } catch (e) {
    setError("\u65E0\u6CD5\u83B7\u53D6\u9EA6\u514B\u98CE\u6743\u9650");
  }
});

async function sendMessage() {
  if (streaming) return;
  const content = promptEl.value.trim();
  if (!content) return;
  const sendingSessionId = sessionId;
  let titleRequested = false;
  let assistantMessageIndex = -1;
  setError("");
  resetTransientPanels();
  addMessage("user", content);
  lastUserQuery = content;
  promptEl.value = "";
  addTyping();
  sendBtn.classList.add("loading");
  streaming = true;

  const payload = {
    api_key: apiKeyEl.value.trim() || undefined,
    base_url: baseUrlEl.value.trim(),
    model: modelEl.value.trim(),
    reasoning_model: "deepseek-reasoner",
    site_region: siteRegionEl ? siteRegionEl.value : "cn_priority",
    search_target: searchTargetEl ? parseInt(searchTargetEl.value, 10) : 20,
    top_k: parseInt(topKEl.value, 10),
    keyword_weight: parseFloat(kwWeightEl.value),
    vector_weight: parseFloat(vecWeightEl.value),
    messages: [...messages],
    file_text: fileText,
    file_assets: fileAssets,
    session_id: sendingSessionId,
    doc_type: docTypeEl ? docTypeEl.value : "通用文书",
    force_web_search: forceWebSearch || (agentModeEl && agentModeEl.checked),
  };

  try {
    const endpoint = agentModeEl && agentModeEl.checked ? "/api/agent_stream" : "/api/chat_stream";
    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(err || "请求失败");
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let assistantText = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        const data = JSON.parse(line);
        if (data.type === "status") {
          setStatus(data.label || "处理中");
        }
        if (data.type === "mindmap_loading") {
          mindmapLoading = true;
          lastMindmap = null;
          if (planContent) {
            planContent.innerHTML = '<div class="plan-loading">思维导图生成中...</div>';
          }
          if (mindmapStatus) {
            mindmapStatus.className = "mindmap-status loading";
            mindmapStatus.textContent = "思维导图：生成中...";
          }
          if (data.mindmap) {
            setTimeout(prefetchCaseAnalyses, 100);
          }
        }
        if (data.type === "trace") {
          addTrace(data.label || "过程", data.detail || "");
        }
        if (data.type === "progress") {
          stopAutoProgress();
          setProgress(data.percent || 0);
        }
        if (data.type === "delta") {
          assistantText += data.text;
          if (!titleRequested) {
            requestSessionAutoTitle(sendingSessionId, content, assistantText);
            titleRequested = true;
          }
          removeTyping();
          if (assistantMessageIndex < 0) {
            messages.push({ role: "assistant", content: assistantText });
            assistantMessageIndex = messages.length - 1;
            touchActiveConversation();
            saveConversationStore();
            renderMessage("assistant", assistantText);
          } else {
            messages[assistantMessageIndex].content = assistantText;
            touchActiveConversation();
            saveConversationStore();
            const last = chat.querySelector(".message.assistant:last-child");
            if (last && last.querySelector(".bubble")) {
              last.querySelector(".bubble").innerHTML = formatAssistantText(assistantText);
              lastAssistantEl = last;
            }
          }
          chat.scrollTop = chat.scrollHeight;
        }
        if (data.type === "meta") {
          stopAutoProgress();
          const lawResults = data.law_results || data.results || [];
          const caseResults = data.case_results || [];
          if (data.results || data.law_results || data.case_results) {
            setRetrievalFromResults(lawResults, caseResults, data.keywords || []);
            const hasRetrieval = (lawResults && lawResults.length) || (caseResults && caseResults.length);
            setDot(retrievalDot, !!hasRetrieval);
          }
          if (data.search_results) {
            setSearchResults(data.search_results || [], data.highlight_keywords || []);
          }
          if (data.case_analyses && data.case_analyses.length) {
            data.case_analyses.forEach((a) => {
              const key = String(a.case_no || a.title || "");
              if (key) caseAnalysisCache[key] = a.analysis || "未生成分析。";
            });
          }
          const hasSearch = data.search_results && data.search_results.length;
          setDot(searchDot, !!hasSearch);
          if (data.downloads) {
            setDownloads(data.downloads || [], data.documents || []);
          }
          if (data.plan_detail) {
            lastPlanDetail = data.plan_detail;
          }
          if (data.mindmap) {
            lastMindmap = data.mindmap;
            mindmapLoading = false;
            if (mindmapStatus) {
              mindmapStatus.className = "mindmap-status done";
              mindmapStatus.textContent = "思维导图：已生成";
            }
          }
          if (data.highlight_keywords && data.highlight_keywords.length) {
            highlightKeywords = data.highlight_keywords;
          } else if (data.keywords && data.keywords.length) {
            highlightKeywords = data.keywords;
          }
          setStatus("任务完成！请查收", true);
        }
      }
    }
  } catch (e) {
    setError(e.message);
  } finally {
    if (!titleRequested) {
      requestSessionAutoTitle(sendingSessionId, content, "");
      titleRequested = true;
    }
    stopAutoProgress();
    removeTyping();
    sendBtn.classList.remove("loading");
    streaming = false;
    renderSessionList();
  }
}

function normalizeGeneratedTitle(raw) {
  if (!raw) return "";
  let text = String(raw).trim();
  text = text.split(/\r?\n/)[0] || "";
  text = text.replace(/^\s*[-*#]+\s*/, "");
  text = text.replace(/^\s*(\d+[\.\)\-、]\s*)+/, "");
  text = text.replace(/^\s*[一二三四五六七八九十]+\s*[、\.．]\s*/, "");
  text = text.replace(/^(会话标题|标题|建议标题)\s*[:：\-]\s*/i, "");
  text = text.replace(/^(结论|总结|摘要|概述|分析)\s*[:：\-]?\s*/i, "");
  text = text.replace(/^["'“”`]+|["'“”`]+$/g, "");
  text = text.replace(/[。！？!?；;，,：:]+$/g, "");
  text = text.replace(/(问题|咨询|结论|摘要)$/g, "");
  text = text.replace(/\s+/g, " ").trim();
  if (!text) return "";
  if (/^(结论|总结|摘要|概述|分析|回答)$/i.test(text)) return "";
  if (text.length < 2) return "";
  return text.slice(0, 20);
}

function buildFallbackTitleFromQuestion(userText) {
  let text = String(userText || "").trim();
  if (!text) return "";
  text = text.replace(/\s+/g, " ");
  text = text.replace(/[“”"'`]/g, "");
  text = text.replace(/[？?。！!；;:：]+/g, " ");
  text = text.replace(/[，,、]/g, " ");
  text = text.replace(/\s+/g, " ").trim();
  if (!text) return "";
  text = text.replace(/^(请问|咨询|我想问|我想咨询|我想了解|我今天|我现在|我最近|我|本人)\s*/, "");
  text = text.replace(/\s*(怎么办|怎么处理|是否合法|违法吗|可以吗|能否|吗|呢)\s*$/g, "");
  text = text.replace(/\s+/g, " ").trim();
  if (text.endsWith("问题")) text = text.slice(0, -2);
  if (text.endsWith("咨询")) text = text.slice(0, -2);
  return text.slice(0, 18);
}

function requestSessionAutoTitle(targetSessionId, userContent, assistantContent) {
  const item = conversations.find((c) => c.id === targetSessionId);
  if (!item) return;
  if (item.manualTitle) return;
  if ((item.title || "").trim() !== SESSION_DEFAULT_TITLE) return;
  const userText = (userContent || "").trim();
  const assistantText = (assistantContent || "").trim();
  if (!userText) return;

  const payload = {
    api_key: apiKeyEl.value.trim() || undefined,
    base_url: baseUrlEl.value.trim(),
    model: modelEl.value.trim(),
    user_query: userText,
    assistant_text: assistantText.slice(0, 320),
  };

  fetch("/api/session_title", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then((resp) => resp.json().then((data) => ({ ok: resp.ok, data })))
    .then(({ ok, data }) => {
      const normalized = normalizeGeneratedTitle((data && data.title) || "");
      const fallback = buildFallbackTitleFromQuestion(userText);
      const title = normalized || fallback;
      if (!title) return;
      const target = conversations.find((c) => c.id === targetSessionId);
      if (!target) return;
      if (target.manualTitle) return;
      if ((target.title || "").trim() !== SESSION_DEFAULT_TITLE) return;
      target.title = title;
      target.updatedAt = Date.now();
      saveConversationStore();
      renderSessionList();
      if (sessionId === targetSessionId) {
        renderCurrentSessionTitle();
      }
    })
    .catch(() => {});
}

if (sendBtn) {
  sendBtn.addEventListener("click", sendMessage);
}

if (promptEl) {
  promptEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}

if (chat) {
  chat.addEventListener("click", (e) => {
    const caseLink = e.target.closest(".case-link");
    if (caseLink) {
      e.preventDefault();
      const no = caseLink.getAttribute("data-case");
      scrollToCase(no);
      return;
    }
    const webLink = e.target.closest(".web-link");
    if (webLink) {
      e.preventDefault();
      const no = webLink.getAttribute("data-web");
      scrollToWeb(no);
    }
  });
}

if (retrievalToggle) {
  retrievalToggle.addEventListener("click", () => {
    retrievalWrap.classList.toggle("collapsed");
    retrievalToggle.classList.toggle("open");
    setDot(retrievalDot, false);
  });
}

if (searchToggle) {
  searchToggle.addEventListener("click", () => {
    searchWrap.classList.toggle("collapsed");
    searchToggle.classList.toggle("open");
    setDot(searchDot, false);
  });
}

if (myChatsToggle) {
  myChatsToggle.addEventListener("click", () => {
    myChatsWrap.classList.toggle("collapsed");
    myChatsToggle.classList.toggle("open");
  });
}

if (newChatBtn) {
  newChatBtn.addEventListener("click", () => {
    createNewSession();
  });
}

if (myChatsList) {
  myChatsList.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    const action = btn.getAttribute("data-action");
    const id = btn.getAttribute("data-id");
    if (!id) return;
    if (action === "switch") {
      setActiveSession(id);
      return;
    }
    if (action === "rename") {
      renameConversation(id);
      return;
    }
    if (action === "favorite") {
      toggleFavoriteConversation(id);
      return;
    }
    if (action === "delete") {
      deleteConversation(id);
    }
  });
}

if (moreToggle) {
  moreToggle.addEventListener("click", () => {
    moreWrap.classList.toggle("collapsed");
    moreToggle.classList.toggle("open");
  });
}

initApiKeyFromStorage();
initSessionStore();
loadOcrInfo();

if (planDetailBtn) {
  planDetailBtn.addEventListener("click", () => {
    if (mindmapLoading) {
      if (planContent) planContent.innerHTML = "<div class=\"plan-loading\">思维导图生成中...</div>";
    } else if (lastMindmap) {
      renderMindmap(lastMindmap);
    } else if (!lastPlanDetail) {
      if (planContent) planContent.textContent = "暂无计划详情";
    } else if (typeof lastPlanDetail === "string") {
      if (planContent) planContent.textContent = lastPlanDetail;
    } else if (typeof lastPlanDetail === "object") {
      // fallback to old plan_detail rendering
      const goal = lastPlanDetail.goal || "";
      const steps = lastPlanDetail.steps || [];
      const svgWrap = document.createElement("div");
      svgWrap.className = "svg-wrap";

      const nodeW = 240;
      const nodeH = 70;
      const gapX = 40;
      const gapY = 80;

      const childCount = steps.length || 1;
      const width = Math.max(800, childCount * (nodeW + gapX));
      let maxSub = 0;
      steps.forEach((s) => {
        const subs = s.subtasks || [];
        if (subs.length > maxSub) maxSub = subs.length;
      });
      const height = 220 + maxSub * 90;

      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      svg.setAttribute("width", "100%");
      svg.setAttribute("height", "480");
      svg.classList.add("mindmap");

      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("id", "viewport");

      // Root node
      const rootX = width / 2 - nodeW / 2;
      const rootY = 20;
      g.appendChild(drawNode(rootX, rootY, nodeW, nodeH, "目标分析", goal, "root"));

      // Step nodes
      steps.forEach((s, idx) => {
        const x = idx * (nodeW + gapX) + gapX;
        const y = 140;
        g.appendChild(drawLine(rootX + nodeW / 2, rootY + nodeH, x + nodeW / 2, y));
        const detailText = [s.step || "", s.detail || "", s.result || ""].filter(Boolean).join(" | ");
        const stepNode = drawNode(x, y, nodeW, nodeH, `步骤${idx + 1}`, detailText, "step");
        stepNode.dataset.stepIndex = String(idx);
        stepNode.addEventListener("click", () => {
          renderPlanDetail(idx);
        });
        g.appendChild(stepNode);
        const subs = s.subtasks || [];
        subs.forEach((sub, i) => {
          const sy = y + gapY + i * 70;
          g.appendChild(drawLine(x + nodeW / 2, y + nodeH, x + nodeW / 2, sy));
          const subNode = drawNode(x, sy, nodeW, 48, "\u5B50\u4EFB\u52A1", sub, "sub");
          subNode.addEventListener("click", () => {
            renderPlanDetail(idx);
          });
          g.appendChild(subNode);
        });
      });

      svg.appendChild(g);
      svgWrap.appendChild(svg);
      if (planContent) {
        planContent.innerHTML = "";
        planContent.appendChild(svgWrap);
        const hint = document.createElement("div");
        hint.className = "plan-hint";
        hint.textContent = "点击节点查看该步骤的详细内容";
        planContent.appendChild(hint);
      }

      enablePanZoom(svg, g);
    }
    if (planModal) planModal.classList.remove("hidden");
  });
}

async function generateDocumentFromAction() {
  if (streaming) return;
  setError("");
  const prompt = lastUserQuery || promptEl.value.trim();
  if (!prompt) {
    setError("\u8BF7\u5148\u8F93\u5165\u6587\u4E66\u9700\u6C42");
    return;
  }
  addTyping();
  streaming = true;
  if (docPanelBtn) docPanelBtn.classList.add("loading");
  try {
    const payload = {
      api_key: apiKeyEl.value.trim() || undefined,
      base_url: baseUrlEl.value.trim(),
      model: modelEl.value.trim(),
      session_id: sessionId,
      query: prompt,
      file_text: fileText,
      file_assets: fileAssets,
      doc_type: docTypeEl ? docTypeEl.value : "通用文书",
    };
    const resp = await fetch("/api/generate_document", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "生成文书失败");
    const answer = data.answer || "\u5DF2\u751F\u6210\u6587\u4E66\u3002";
    addMessage("assistant", answer);
    if (data.downloads) {
      setDownloads(data.downloads || [], data.documents || []);
    }
  } catch (e) {
    setError(e.message);
  } finally {
    if (docPanelBtn) docPanelBtn.classList.remove("loading");
    removeTyping();
    streaming = false;
  }
}

if (docPanelBtn) {
  docPanelBtn.addEventListener("click", () => {
    generateDocumentFromAction();
  });
}

if (docGenBtn) {
  docGenBtn.addEventListener("click", async () => {
    if (activeDocIndex < 0) return;
    const doc = documentsCache[activeDocIndex] || {};
    const title = doc.title || (docTypeEl ? docTypeEl.value : "文书");
    const body = docEditor ? docEditor.textContent : doc.body || "";
    try {
      docGenBtn.classList.add("loading");
      const payload = {
        api_key: apiKeyEl.value.trim() || undefined,
        base_url: baseUrlEl.value.trim(),
        model: modelEl.value.trim(),
        session_id: sessionId,
        title,
        body,
        doc_type: docTypeEl ? docTypeEl.value : "通用文书",
        file_assets: doc.images || fileAssets,
      };
      const resp = await fetch("/api/generate_document", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "生成PDF失败");
      if (data.downloads && data.downloads.length) {
        downloadsCache[activeDocIndex] = data.downloads[0];
        setDownloads(downloadsCache, documentsCache);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      docGenBtn.classList.remove("loading");
    }
  });
}

if (docAiBtn) {
  docAiBtn.addEventListener("click", async () => {
    if (activeDocIndex < 0) return;
    const instruction = (docAiInput && docAiInput.value.trim()) || "";
    if (!instruction) return;
    const text = docEditor ? docEditor.textContent : "";
    try {
      docAiBtn.classList.add("loading");
      const payload = {
        api_key: apiKeyEl.value.trim() || undefined,
        base_url: baseUrlEl.value.trim(),
        model: modelEl.value.trim(),
        session_id: sessionId,
        text,
        instruction,
        doc_type: docTypeEl ? docTypeEl.value : "通用文书",
      };
      const resp = await fetch("/api/doc_refine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "AI微调失败");
      if (docEditor) docEditor.textContent = data.text || text;
      documentsCache[activeDocIndex].body = docEditor.textContent;
      if (docAiInput) docAiInput.value = "";
    } catch (e) {
      setError(e.message);
    } finally {
      docAiBtn.classList.remove("loading");
    }
  });
}

const chipEls = document.querySelectorAll(".chip");
chipEls.forEach((chip) => {
  chip.addEventListener("click", () => {
    const p = chip.getAttribute("data-prompt") || "";
    if (docAiInput) {
      docAiInput.value = p;
      docAiInput.focus();
    }
  });
});

function drawNode(x, y, w, h, title, text, cls) {
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  g.classList.add("mind-node");
  g.dataset.x = String(x);
  g.dataset.y = String(y);
  g.dataset.w = String(w);
  g.dataset.h = String(h);
  const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  rect.setAttribute("x", x);
  rect.setAttribute("y", y);
  rect.setAttribute("rx", 12);
  rect.setAttribute("ry", 12);
  rect.setAttribute("width", w);
  rect.setAttribute("height", h);
  rect.setAttribute("class", `node-rect ${cls || ""}`.trim());
  const t1 = document.createElementNS("http://www.w3.org/2000/svg", "text");
  t1.setAttribute("class", "node-title");
  const titleLines = splitText(title || "", 12, 2);
  const lineH = 14;
  t1.setAttribute("x", x + 12);
  t1.setAttribute("y", y + 20);
  titleLines.forEach((line, idx) => {
    const tsp = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
    tsp.setAttribute("x", x + 12);
    tsp.setAttribute("dy", idx === 0 ? 0 : lineH);
    tsp.textContent = line;
    t1.appendChild(tsp);
  });
  const t2 = document.createElementNS("http://www.w3.org/2000/svg", "text");
  t2.setAttribute("class", "node-text");
  const textLines = splitText(text || "", 20, 3);
  const textStartY = y + 20 + Math.max(1, titleLines.length) * lineH + 12;
  t2.setAttribute("x", x + 12);
  t2.setAttribute("y", textStartY);
  textLines.forEach((line, idx) => {
    const tsp = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
    tsp.setAttribute("x", x + 12);
    tsp.setAttribute("dy", idx === 0 ? 0 : 14);
    tsp.textContent = line;
    t2.appendChild(tsp);
  });
  g.dataset.titleLines = String(titleLines.length || 1);
  g.dataset.textLines = String(textLines.length || 1);
  g.dataset.textStartY = String(textStartY);
  g.appendChild(rect);
  g.appendChild(t1);
  g.appendChild(t2);
  return g;
}

function splitText(text, maxLen, maxLines) {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return [];
  const lines = [];
  let buf = "";
  let i = 0;
  const punct = /[，。；、,.;:!?]/;
  while (i < clean.length && lines.length < maxLines) {
    const ch = clean[i];
    buf += ch;
    const hitLimit = buf.length >= maxLen;
    const hitPunct = punct.test(ch) && buf.length >= Math.floor(maxLen * 0.6);
    if (hitLimit || hitPunct) {
      lines.push(buf.trim());
      buf = "";
      if (lines.length >= maxLines) break;
    }
    i += 1;
  }
  if (lines.length < maxLines && buf.trim()) {
    lines.push(buf.trim());
  }
  if (i < clean.length && lines.length) {
    lines[lines.length - 1] = lines[lines.length - 1].slice(0, Math.max(0, maxLen - 1)) + "...";
  }
  return lines;
}

function drawLine(x1, y1, x2, y2) {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("class", "node-line");
  return line;
}

function enablePanZoom(svg, g) {
  let scale = 1;
  let panX = 0;
  let panY = 0;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  const update = () => {
    g.setAttribute("transform", `translate(${panX}, ${panY}) scale(${scale})`);
  };
  update();

  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.05 : 0.05;
    scale = Math.max(0.4, Math.min(3.5, scale + delta));
    update();
  });
  svg.addEventListener("mousedown", (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    panX += e.clientX - lastX;
    panY += e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    update();
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
  });
}

function setNodePos(nodeEl, x, y) {
  const rect = nodeEl.querySelector("rect");
  const t1 = nodeEl.querySelector(".node-title");
  const t2 = nodeEl.querySelector(".node-text");
  if (!rect || !t1 || !t2) return;
  rect.setAttribute("x", x);
  rect.setAttribute("y", y);
  t1.setAttribute("x", x + 12);
  t1.setAttribute("y", y + 20);
  const titleSpans = t1.querySelectorAll("tspan");
  titleSpans.forEach((tsp, idx) => {
    tsp.setAttribute("x", x + 12);
    tsp.setAttribute("dy", idx === 0 ? 0 : 14);
  });
  const titleLines = parseInt(nodeEl.dataset.titleLines || "1", 10);
  const textStartY = y + 20 + Math.max(1, titleLines) * 14 + 12;
  t2.setAttribute("x", x + 12);
  t2.setAttribute("y", textStartY);
  const tspans = t2.querySelectorAll("tspan");
  tspans.forEach((tsp, idx) => {
    tsp.setAttribute("x", x + 12);
    tsp.setAttribute("dy", idx === 0 ? 0 : 14);
  });
  nodeEl.dataset.x = String(x);
  nodeEl.dataset.y = String(y);
}

function toSvgPoint(svg, g, clientX, clientY) {
  const pt = svg.createSVGPoint();
  pt.x = clientX;
  pt.y = clientY;
  const ctm = g.getScreenCTM();
  return ctm ? pt.matrixTransform(ctm.inverse()) : pt;
}

function renderPlanDetail(stepIndex) {
  if (!lastPlanDetail || !planContent) return;
  const old = planContent.querySelector(".plan-detail");
  if (old) old.remove();
  const steps = lastPlanDetail.steps || [];
  const step = steps[stepIndex] || {};
  const inspect = step.inspect || {};
  const kb = inspect.kb || [];
  const web = inspect.web || [];
  const wrap = document.createElement("div");
  wrap.className = "plan-detail";
  const title = document.createElement("div");
  title.className = "plan-detail-title";
  title.textContent = `步骤${stepIndex + 1} 详情`;
  wrap.appendChild(title);

  if (kb.length) {
    const sec = document.createElement("div");
    sec.className = "plan-detail-section";
    sec.innerHTML = "<div class=\"plan-detail-label\">知识库命中摘要</div>";
    kb.forEach((k) => {
      const d = document.createElement("details");
      const s = document.createElement("summary");
      s.textContent = `${k.title} - ${k.snippet || ""}`.slice(0, 80);
      const p = document.createElement("div");
      p.className = "plan-detail-body";
      p.textContent = k.detail || k.snippet || "";
      d.appendChild(s);
      d.appendChild(p);
      sec.appendChild(d);
    });
    wrap.appendChild(sec);
  }

  if (web.length) {
    const sec = document.createElement("div");
    sec.className = "plan-detail-section";
    sec.innerHTML = "<div class=\"plan-detail-label\">网页命中摘要</div>";
    web.forEach((k) => {
      const d = document.createElement("details");
      const s = document.createElement("summary");
      s.textContent = `${k.title} - ${(k.url || "").replace(/^https?:\/\//, "")}`.slice(0, 90);
      const p = document.createElement("div");
      p.className = "plan-detail-body";
      p.textContent = k.detail || k.snippet || "";
      d.appendChild(s);
      d.appendChild(p);
      sec.appendChild(d);
    });
    wrap.appendChild(sec);
  }

  planContent.appendChild(wrap);
}

function renderMindmapDetail(node) {
  if (!mindmapDetailBody || !node) return;
  mindmapDetailBody.innerHTML = "";
  const title = document.createElement("div");
  title.className = "plan-detail-title";
  title.textContent = node.title || "节点详情";
  mindmapDetailBody.appendChild(title);
  const sum = document.createElement("div");
  sum.className = "plan-detail-body";
  sum.innerHTML = highlightHtml(node.summary || "", highlightKeywords);
  mindmapDetailBody.appendChild(sum);
  if (node.detail) {
    let rendered = false;
    const raw = node.detail || "";
    try {
      if (raw.trim().startsWith("[") && raw.includes("\"title\"")) {
        const arr = JSON.parse(raw);
        if (Array.isArray(arr)) {
          const wrap = document.createElement("div");
          wrap.className = "mindmap-list";
          arr.forEach((item) => {
            const card = document.createElement("div");
            card.className = "mindmap-item";
            const t = document.createElement("div");
            t.className = "mindmap-item-title";
            t.textContent = item.title || "条目";
            const b = document.createElement("div");
            b.className = "mindmap-item-body";
            b.innerHTML = highlightHtml(item.snippet || item.detail || "", highlightKeywords);
            card.appendChild(t);
            card.appendChild(b);
            wrap.appendChild(card);
          });
          mindmapDetailBody.appendChild(wrap);
          rendered = true;
        }
      }
    } catch {}
    if (!rendered) {
      const det = document.createElement("div");
      det.className = "plan-detail-body";
      det.innerHTML = highlightHtml(raw, highlightKeywords);
      mindmapDetailBody.appendChild(det);
    }
  }
  if (mindmapDetailModal) mindmapDetailModal.classList.remove("hidden");
}

function renderMindmap(mindmap) {
  if (!mindmap || !mindmap.nodes || !mindmap.nodes.length) return;
  const nodes = mindmap.nodes;
  const edges = mindmap.edges || [];
  const nodeMap = new Map();
  nodes.forEach((n) => nodeMap.set(n.id, n));
  const lineMap = new Map();
  const adj = new Map();
  edges.forEach((e) => {
    if (!adj.has(e.from)) adj.set(e.from, []);
    adj.get(e.from).push(e.to);
  });
  const root = nodeMap.get("root") || nodes[0];
  const depth = {};
  depth[root.id] = 0;
  const queue = [root.id];
  while (queue.length) {
    const id = queue.shift();
    const kids = adj.get(id) || [];
    kids.forEach((k) => {
      if (depth[k] == null) {
        depth[k] = (depth[id] || 0) + 1;
        queue.push(k);
      }
    });
  }
  nodes.forEach((n) => {
    if (depth[n.id] == null) depth[n.id] = 1;
  });
  const levels = {};
  nodes.forEach((n) => {
    const d = depth[n.id] || 0;
    if (!levels[d]) levels[d] = [];
    levels[d].push(n);
  });
  const maxLevel = Math.max(...Object.keys(levels).map((k) => parseInt(k, 10)));
  const nodeW = 280;
  const nodeH = 104;
  const gapX = 56;
  const gapY = 130;
  const maxPerLevel = Math.max(...Object.values(levels).map((v) => v.length));
  const width = Math.max(980, (maxPerLevel || 1) * (nodeW + gapX));
  const height = (maxLevel + 1) * (nodeH + gapY) + 80;

  const layout = document.createElement("div");
  layout.className = "mindmap-layout";
  const svgWrap = document.createElement("div");
  svgWrap.className = "svg-wrap";
  const hint = document.createElement("div");
  hint.className = "mindmap-hint";
  hint.textContent = "点击节点查看详情（弹窗展示）";
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "560");
  svg.classList.add("mindmap");
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  g.setAttribute("id", "viewport");

  const positions = {};
  Object.entries(levels).forEach(([dStr, arr]) => {
    const d = parseInt(dStr, 10);
    const totalW = arr.length * (nodeW + gapX);
    let startX = (width - totalW) / 2 + gapX / 2;
    arr.forEach((n, idx) => {
      positions[n.id] = { x: startX + idx * (nodeW + gapX), y: 30 + d * (nodeH + gapY) };
    });
  });

  edges.forEach((e) => {
    const a = positions[e.from];
    const b = positions[e.to];
    if (!a || !b) return;
    const line = drawLine(a.x + nodeW / 2, a.y + nodeH, b.x + nodeW / 2, b.y);
    line.dataset.from = e.from;
    line.dataset.to = e.to;
    lineMap.set(`${e.from}->${e.to}`, line);
    g.appendChild(line);
  });

  nodes.forEach((n) => {
    const pos = positions[n.id];
    if (!pos) return;
    const group = n.group || (n.id === root.id ? "goal" : "plan");
    let cls = "step";
    if (group === "goal") cls = "root";
    else if (group === "kb") cls = "kb";
    else if (group === "case") cls = "case";
    else if (group === "web") cls = "web";
    else if (group === "action") cls = "sub";
    const text = (n.summary || n.detail || "").trim();
    const nodeEl = drawNode(pos.x, pos.y, nodeW, nodeH, n.title || "节点", text, cls);
    nodeEl.dataset.id = n.id;
    nodeEl.addEventListener("click", () => renderMindmapDetail(n));
    g.appendChild(nodeEl);
    // drag support
    let dragging = false;
    let startPt = null;
    let startPos = null;
    nodeEl.addEventListener("pointerdown", (e) => {
      e.stopPropagation();
      dragging = true;
      nodeEl.setPointerCapture(e.pointerId);
      startPt = toSvgPoint(svg, g, e.clientX, e.clientY);
      startPos = { x: positions[n.id].x, y: positions[n.id].y };
    });
    nodeEl.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const p = toSvgPoint(svg, g, e.clientX, e.clientY);
      const dx = p.x - startPt.x;
      const dy = p.y - startPt.y;
      const nx = startPos.x + dx;
      const ny = startPos.y + dy;
      positions[n.id].x = nx;
      positions[n.id].y = ny;
      setNodePos(nodeEl, nx, ny);
      // update connected lines
      edges.forEach((ed) => {
        if (ed.from !== n.id && ed.to !== n.id) return;
        const a = positions[ed.from];
        const b = positions[ed.to];
        const line = lineMap.get(`${ed.from}->${ed.to}`);
        if (!line || !a || !b) return;
        line.setAttribute("x1", a.x + nodeW / 2);
        line.setAttribute("y1", a.y + nodeH);
        line.setAttribute("x2", b.x + nodeW / 2);
        line.setAttribute("y2", b.y);
      });
    });
    nodeEl.addEventListener("pointerup", (e) => {
      dragging = false;
      try { nodeEl.releasePointerCapture(e.pointerId); } catch {}
    });
    nodeEl.addEventListener("pointerleave", () => {
      dragging = false;
    });
  });

  svg.appendChild(g);
  svgWrap.appendChild(svg);
  layout.appendChild(svgWrap);
  layout.appendChild(hint);
  if (planContent) {
    planContent.innerHTML = "";
    planContent.appendChild(layout);
  }
  enablePanZoom(svg, g);
}
if (planClose) {
  planClose.addEventListener("click", () => {
    if (planModal) planModal.classList.add("hidden");
  });
}
if (planModal) {
  planModal.addEventListener("click", (e) => {
    if (e.target === planModal) planModal.classList.add("hidden");
  });
}
if (mindmapDetailClose) {
  mindmapDetailClose.addEventListener("click", () => {
    if (mindmapDetailModal) mindmapDetailModal.classList.add("hidden");
  });
}
if (mindmapDetailModal) {
  mindmapDetailModal.addEventListener("click", (e) => {
    if (e.target === mindmapDetailModal) mindmapDetailModal.classList.add("hidden");
  });
}

if (diagBtn) {
  diagBtn.addEventListener("click", async () => {
    if (diagOutput) diagOutput.textContent = "诊断中...";
    try {
      const resp = await fetch("/api/search_diagnostics");
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "诊断失败");
      const lines = [];
      lines.push(`引擎: ${data.engine || "未知"}`);
      lines.push(`后端: ${data.backends || "未知"}`);
      lines.push(`区域: ${data.region || "未知"}`);
      lines.push(`命中网址数量: ${data.total_results ?? 0}`);
      lines.push(`有效正文数量: ${data.content_hits ?? 0}`);
      if (data.queries && data.queries.length) {
        lines.push(`关键词: ${data.queries.join(", ")}`);
      }
      if (diagOutput) diagOutput.textContent = lines.join("\n");
    } catch (e) {
      if (diagOutput) diagOutput.textContent = e.message;
    }
  });
}

console.log('app.js loaded');

