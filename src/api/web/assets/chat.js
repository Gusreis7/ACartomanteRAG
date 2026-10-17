const chatId = window.location.pathname.split("/").filter(Boolean).at(-1);
const messagesEl = document.querySelector("#messages");
const emptyEl = document.querySelector("#chat-empty");
const scrollEl = document.querySelector("#message-scroll");
const form = document.querySelector("#composer");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const titleEl = document.querySelector("#conversation-title");
const listEl = document.querySelector("#conversation-list");
const toast = document.querySelector("#toast");
const sidebar = document.querySelector("#chat-sidebar");
const evidenceTable = document.querySelector("#evidence-table");
const evidenceSpread = document.querySelector("#evidence-spread");
const evidenceTableCount = document.querySelector("#evidence-table-count");
const evidenceTableClose = document.querySelector("#evidence-table-close");
const personaButtons = [...document.querySelectorAll("[data-persona]")];
const retryChat = document.querySelector("#chat-retry");

let sending = false;
let activePersona = "neutral";

function setPersona(nextPersona) {
  activePersona = ["neutral", "cosmo", "damiao"].includes(nextPersona)
    ? nextPersona
    : "neutral";

  personaButtons.forEach((button) => {
    const selected = button.dataset.persona === activePersona;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.setTimeout(() => toast.classList.remove("is-visible"), 3500);
}

async function jsonFetch(url, options = {}) {
  const response = await fetch(url, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });

  if (!response.ok) {
    let detail = "Não foi possível concluir a consulta.";
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") detail = payload.detail;
    } catch (_) {}
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }

  return response.json();
}

function scrollToBottom(behavior = "smooth") {
  scrollEl.scrollTo({ top: scrollEl.scrollHeight, behavior });
}

function evidenceCardNode(evidence, index) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = "evidence-card";
  card.setAttribute("aria-pressed", "false");
  card.setAttribute("aria-label", `Virar evidência ${index + 1}`);
  card.style.setProperty("--deal-order", index);

  const inner = document.createElement("span");
  inner.className = "evidence-card-inner";

  const back = document.createElement("span");
  back.className = "evidence-card-face evidence-card-back";

  const backNumber = document.createElement("span");
  backNumber.className = "evidence-card-number";
  backNumber.textContent = String(index + 1).padStart(2, "0");

  const sigil = document.createElement("span");
  sigil.className = "evidence-card-sigil";
  sigil.setAttribute("aria-hidden", "true");
  sigil.innerHTML = '<i></i><b>✦</b><i></i>';

  const backLabel = document.createElement("span");
  backLabel.className = "evidence-card-back-label";
  backLabel.textContent = "EVIDÊNCIA";

  back.append(backNumber, sigil, backLabel);

  const front = document.createElement("span");
  front.className = "evidence-card-face evidence-card-front";

  const frontNumber = document.createElement("span");
  frontNumber.className = "evidence-card-front-number";
  frontNumber.textContent = `Carta ${String(index + 1).padStart(2, "0")}`;

  const source = document.createElement("strong");
  const pieces = [evidence.work, evidence.section].filter(Boolean);
  source.textContent = pieces.join(" · ") || "Trecho recuperado";

  const excerpt = document.createElement("span");
  excerpt.className = "evidence-card-excerpt";
  excerpt.textContent = evidence.excerpt || "Trecho sem conteúdo textual disponível.";

  const metadata = document.createElement("span");
  metadata.className = "evidence-card-metadata";

  if (evidence.evidence_id) {
    const id = document.createElement("span");
    id.textContent = `id ${evidence.evidence_id}`;
    metadata.appendChild(id);
  }

  if (evidence.chunk_id) {
    const chunk = document.createElement("span");
    chunk.textContent = `chunk ${evidence.chunk_id}`;
    metadata.appendChild(chunk);
  }

  front.append(frontNumber, source, excerpt, metadata);
  inner.append(back, front);
  card.appendChild(inner);

  card.addEventListener("click", () => {
    const flipped = card.classList.toggle("is-flipped");
    card.setAttribute("aria-pressed", String(flipped));
    card.setAttribute(
      "aria-label",
      flipped
        ? `Ocultar evidência ${index + 1}`
        : `Virar evidência ${index + 1}`,
    );
  });

  return card;
}

function openEvidenceTable(evidences = []) {
  if (!evidences.length || !evidenceTable || !evidenceSpread) return;

  const cards = evidences.slice(0, 10);
  evidenceSpread.replaceChildren();

  cards.forEach((evidence, index) => {
    evidenceSpread.appendChild(evidenceCardNode(evidence, index));
  });

  evidenceTableCount.textContent = `${cards.length} ${cards.length === 1 ? "carta" : "cartas"} sobre a mesa`;
  evidenceTable.showModal();

  requestAnimationFrame(() => {
    evidenceSpread.querySelectorAll(".evidence-card").forEach((card, index) => {
      window.setTimeout(() => card.classList.add("is-dealt"), index * 65);
    });
  });
}

function evidenceTriggerNode(evidences = []) {
  if (!evidences.length) return null;

  const wrapper = document.createElement("div");
  wrapper.className = "evidence-trigger-shell";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "evidence-trigger";

  const mark = document.createElement("span");
  mark.className = "evidence-trigger-mark";
  mark.setAttribute("aria-hidden", "true");
  mark.textContent = "✦";

  const label = document.createElement("span");
  label.textContent = "Mostre-me as evidências";

  const count = document.createElement("small");
  const visibleCount = Math.min(evidences.length, 10);
  count.textContent = `${visibleCount} ${visibleCount === 1 ? "carta" : "cartas"}`;

  button.append(mark, label, count);
  button.addEventListener("click", () => openEvidenceTable(evidences));
  wrapper.appendChild(button);
  return wrapper;
}

function addMessage(role, content, evidences = [], options = {}) {
  emptyEl.hidden = true;

  const article = document.createElement("article");
  article.className = `message message-${role}`;
  if (options.pending) article.classList.add("is-pending");

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = role === "user" ? "você" : "cartomante";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  const text = document.createElement(role === "assistant" && options.html ? "div" : "p");
  if (role === "assistant" && options.html) {
    text.className = "markdown-body";
    // HTML produzido exclusivamente pelo renderer Markdown do servidor.
    text.innerHTML = options.html;
  } else {
    text.textContent = content;
  }
  bubble.appendChild(text);

  const evidence = role === "assistant" ? evidenceTriggerNode(evidences) : null;
  if (evidence) bubble.appendChild(evidence);

  article.append(meta, bubble);
  messagesEl.appendChild(article);
  scrollToBottom(options.instant ? "auto" : "smooth");
  return article;
}

function setSending(value) {
  sending = value;
  sendButton.disabled = value;
  input.disabled = value;
  personaButtons.forEach((button) => { button.disabled = value; });
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

function renderConversations(conversations) {
  listEl.replaceChildren();

  conversations.forEach((conversation) => {
    const link = document.createElement("a");
    link.href = `/chat/${conversation.chat_id}`;
    link.className = "conversation-link";
    if (String(conversation.chat_id) === chatId) link.classList.add("is-active");

    const marker = document.createElement("span");
    marker.textContent = "✦";
    marker.setAttribute("aria-hidden", "true");

    const text = document.createElement("span");
    text.textContent = conversation.title || "Nova conversa";

    link.append(marker, text);
    listEl.appendChild(link);

    if (String(conversation.chat_id) === chatId) {
      titleEl.textContent = conversation.title || "Nova conversa";
    }
  });
}

async function loadConversations() {
  try {
    const conversations = await jsonFetch("/api/v1/conversations");
    renderConversations(conversations);
  } catch (error) {
    showToast(error.message);
  }
}

async function loadMessages() {
  try {
    const payload = await jsonFetch(`/api/v1/conversations/${chatId}/messages`);
    const messages = payload.messages || [];

    if (!messages.length) {
      emptyEl.hidden = false;
      return;
    }

    messages.forEach((message) => {
      addMessage(
        message.role,
        message.content,
        message.evidences || [],
        { instant: true, html: message.content_html },
      );
    });
  } catch (error) {
    if (error.status === 404) {
      showToast("Esta consulta não pertence à sessão atual ou não existe.");
    } else {
      showToast(error.message);
    }
  }
}

async function sendMessage(message) {
  const clean = message.trim();
  if (!clean || sending) return;

  setSending(true);
  try {
    await window.CartomanteGuide.ensure();
  } catch (error) {
    showToast(error.message);
    setSending(false);
    return;
  }
  const requestPersona = activePersona;

  addMessage("user", clean);
  input.value = "";
  resizeInput();
  setSending(true);

  const pending = addMessage("assistant", "Consultando o acervo…", [], { pending: true });

  try {
    const payload = await jsonFetch(`/api/v1/conversations/${chatId}/messages`, {
      method: "POST",
      body: JSON.stringify({ message: clean, persona: requestPersona }),
    });

    pending.remove();
    addMessage("assistant", payload.answer, payload.evidences || [], { html: payload.answer_html });
    await loadConversations();
  } catch (error) {
    pending.remove();
    addMessage("assistant", `A leitura foi interrompida: ${error.message}`);
  } finally {
    setSending(false);
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

input.addEventListener("input", resizeInput);
for (const eventName of ["paste", "drop"]) {
  input.addEventListener(eventName, (event) => {
    const transfer = event.clipboardData || event.dataTransfer;
    if (transfer?.files?.length || [...(transfer?.items || [])].some((item) => item.kind === "file")) {
      event.preventDefault();
      showToast("Envie somente texto. Arquivos e imagens não são aceitos.");
    }
  });
}
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-chat-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.chatPrompt || "";
    resizeInput();
    input.focus();
  });
});

document.querySelector("#new-chat")?.addEventListener("click", async () => {
  try {
    const conversation = await jsonFetch("/api/v1/conversations", {
      method: "POST",
      body: JSON.stringify({ title: null }),
    });
    window.location.assign(`/chat/${conversation.chat_id}`);
  } catch (error) {
    showToast(error.message);
  }
});

document.querySelector("#sidebar-toggle")?.addEventListener("click", () => {
  sidebar.classList.toggle("is-open");
});

personaButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    if (sending) return;
    setSending(true);
    try {
      const state = await jsonFetch(`/api/v1/conversations/${chatId}/guide`, {
        method: "PATCH", body: JSON.stringify({ persona: button.dataset.persona }),
      });
      setPersona(state.persona);
    } catch (error) {
      showToast(error.message);
    } finally {
      setSending(false);
      input.focus();
    }
  });
});

evidenceTableClose?.addEventListener("click", () => evidenceTable.close());

evidenceTable?.addEventListener("click", (event) => {
  if (event.target === evidenceTable) evidenceTable.close();
});

const queryPrompt = new URLSearchParams(window.location.search).get("q");
if (queryPrompt) {
  input.value = queryPrompt;
  resizeInput();
}

async function initializeChat() {
  setSending(true);
  retryChat.hidden = true;
  try {
    await window.CartomanteGuide.ensure();
    let state = await jsonFetch(`/api/v1/conversations/${chatId}/guide`);
    setPersona(state.persona);
    await Promise.all([loadConversations(), loadMessages()]);
    if (state.draw_required) {
      state = await window.CartomanteDraw.open({
        draw: (card) => jsonFetch(`/api/v1/conversations/${chatId}/guide/draw`, {
          method: "POST", body: JSON.stringify({ card }),
        }),
        neutral: () => jsonFetch(`/api/v1/conversations/${chatId}/guide`, {
          method: "PATCH", body: JSON.stringify({ persona: "neutral" }),
        }),
      });
      setPersona(state.persona);
    }
    setSending(false);
    input.focus();
  } catch (error) {
    showToast(error.message);
    retryChat.hidden = false;
  }
}

retryChat.addEventListener("click", initializeChat);
initializeChat();
