const startButton = document.querySelector("#start-chat");
const recentBlock = document.querySelector("#recent-block");
const recentList = document.querySelector("#recent-list");
const toast = document.querySelector("#toast");


function showToast(message) {
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.setTimeout(() => toast.classList.remove("is-visible"), 3200);
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
      detail = typeof payload.detail === "string" ? payload.detail : detail;
    } catch (_) {}
    throw new Error(detail);
  }

  return response.json();
}

async function createConversation(prompt = "") {
  startButton?.setAttribute("disabled", "disabled");

  try {
    const conversation = await jsonFetch("/api/v1/conversations", {
      method: "POST",
      body: JSON.stringify({ title: null }),
    });

    const target = new URL(`/chat/${conversation.chat_id}`, window.location.origin);
    if (prompt) target.searchParams.set("q", prompt);
    window.location.assign(target.pathname + target.search);
  } catch (error) {
    showToast(error.message);
    startButton?.removeAttribute("disabled");
  }
}

function renderRecent(conversations) {
  recentList.replaceChildren();

  conversations.slice(0, 4).forEach((conversation) => {
    const link = document.createElement("a");
    link.className = "recent-item";
    link.href = `/chat/${conversation.chat_id}`;

    const title = document.createElement("span");
    title.textContent = conversation.title || "Nova conversa";

    const arrow = document.createElement("span");
    arrow.textContent = "↗";
    arrow.setAttribute("aria-hidden", "true");

    link.append(title, arrow);
    recentList.appendChild(link);
  });

  recentBlock.hidden = conversations.length === 0;
}

async function loadRecent() {
  try {
    const conversations = await jsonFetch("/api/v1/conversations");
    renderRecent(conversations);
  } catch (_) {
    recentBlock.hidden = true;
  }
}

startButton?.addEventListener("click", () => createConversation());

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => createConversation(button.dataset.prompt || ""));
});

loadRecent();
