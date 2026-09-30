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

let sending = false;

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

function evidenceNode(evidences = []) {
  if (!evidences.length) return null;

  const details = document.createElement("details");
  details.className = "evidence-drawer";

  const summary = document.createElement("summary");
  summary.textContent = `${evidences.length} ${evidences.length === 1 ? "evidência consultada" : "evidências consultadas"}`;
  details.appendChild(summary);

  const list = document.createElement("div");
  list.className = "evidence-list";

  evidences.forEach((evidence, index) => {
    const item = document.createElement("article");
    item.className = "evidence-item";

    const heading = document.createElement("div");
    heading.className = "evidence-heading";

    const number = document.createElement("span");
    number.textContent = String(index + 1).padStart(2, "0");

    const source = document.createElement("strong");
    const pieces = [evidence.work, evidence.section].filter(Boolean);
    source.textContent = pieces.join(" · ") || "Trecho recuperado";
    heading.append(number, source);

    const excerpt = document.createElement("p");
    excerpt.textContent = evidence.excerpt || "";

    const code = document.createElement("small");
    code.textContent = evidence.chunk_id ? `chunk ${evidence.chunk_id}` : "";

    item.append(heading, excerpt, code);
    list.appendChild(item);
  });

  details.appendChild(list);
  return details;
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

  const text = document.createElement("p");
  text.textContent = content;
  bubble.appendChild(text);

  const evidence = role === "assistant" ? evidenceNode(evidences) : null;
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
        { instant: true },
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

  addMessage("user", clean);
  input.value = "";
  resizeInput();
  setSending(true);

  const pending = addMessage("assistant", "Consultando o acervo…", [], { pending: true });

  try {
    const payload = await jsonFetch(`/api/v1/conversations/${chatId}/messages`, {
      method: "POST",
      body: JSON.stringify({ message: clean }),
    });

    pending.remove();
    addMessage("assistant", payload.answer, payload.evidences || []);
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

const queryPrompt = new URLSearchParams(window.location.search).get("q");
if (queryPrompt) {
  input.value = queryPrompt;
  resizeInput();
}

Promise.all([loadConversations(), loadMessages()]).then(() => input.focus());
