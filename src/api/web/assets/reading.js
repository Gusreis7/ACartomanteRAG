const readingDeck = document.querySelector("#reading-deck");
const readingStatus = document.querySelector("#reading-status");
const newReadingButton = document.querySelector("#new-reading");
const toast = document.querySelector("#toast");

let readingWorks = [];
let readingLocked = false;

function officialWorkUrl(value) {
  try {
    const url = new URL(value);
    const host = url.hostname.toLowerCase();
    if (["https:", "http:"].includes(url.protocol) && !url.username && !url.password &&
        (host === "dominiopublico.mec.gov.br" || host.endsWith(".dominiopublico.mec.gov.br"))) {
      return url.href;
    }
  } catch (_) {}
  return null;
}

function showToast(message) {
  if (!toast) return;
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

function readingCardMarkup(work, index) {
  const article = document.createElement("article");
  article.className = "reading-card";
  article.dataset.index = String(index);
  article.style.setProperty("--reading-order", String(index));
  article.tabIndex = 0;
  article.setAttribute("role", "button");
  article.setAttribute("aria-label", `Virar a carta ${index + 1}`);
  article.setAttribute("aria-pressed", "false");

  const inner = document.createElement("div");
  inner.className = "reading-card-inner";

  const back = document.createElement("div");
  back.className = "reading-card-face reading-card-back";
  back.setAttribute("aria-hidden", "true");

  const number = document.createElement("span");
  number.className = "reading-card-number";
  number.textContent = String(index + 1).padStart(2, "0");

  const sigil = document.createElement("div");
  sigil.className = "reading-card-sigil";
  sigil.innerHTML = '<i></i><b>✦</b><i></i>';

  const backLabel = document.createElement("span");
  backLabel.className = "reading-card-back-label";
  backLabel.textContent = "obra velada";

  back.append(number, sigil, backLabel);

  const front = document.createElement("div");
  front.className = "reading-card-face reading-card-front";

  const kicker = document.createElement("span");
  kicker.className = "reading-card-front-kicker";
  kicker.textContent = work.category || "obra revelada";

  const title = document.createElement("strong");
  title.className = "reading-card-title";
  title.textContent = work.title;

  const author = document.createElement("span");
  author.className = "reading-card-author";
  author.textContent = work.author || "Machado de Assis";

  const ornament = document.createElement("span");
  ornament.className = "reading-card-front-ornament";
  ornament.setAttribute("aria-hidden", "true");
  ornament.textContent = "✦  ·  ☾  ·  ✦";

  const source = document.createElement("a");
  source.className = "reading-card-link";
  const sourceUrl = officialWorkUrl(work.source_url);
  if (sourceUrl) source.href = sourceUrl;
  else source.setAttribute("aria-disabled", "true");
  source.target = "_blank";
  source.rel = "noopener noreferrer";
  source.innerHTML = '<span>Ler a obra</span><span aria-hidden="true">↗</span>';
  source.addEventListener("click", (event) => event.stopPropagation());

  front.append(kicker, title, author, ornament, source);
  inner.append(back, front);
  article.append(inner);

  article.addEventListener("click", () => revealReadingCard(article));
  article.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      revealReadingCard(article);
    }
  });

  return article;
}

function renderReading(works) {
  if (!readingDeck) return;

  readingDeck.replaceChildren();
  readingWorks = works;
  readingLocked = false;

  works.forEach((work, index) => {
    readingDeck.appendChild(readingCardMarkup(work, index));
  });

  window.requestAnimationFrame(() => {
    readingDeck.querySelectorAll(".reading-card").forEach((card) => {
      card.classList.add("is-dealt");
    });
  });

  if (readingStatus) {
    readingStatus.textContent = "Escolha uma carta. Apenas uma poderá ser revelada nesta leitura.";
  }
  if (newReadingButton) {
    newReadingButton.hidden = true;
    newReadingButton.removeAttribute("disabled");
  }
}

function revealReadingCard(selectedCard) {
  if (readingLocked || selectedCard.classList.contains("is-locked")) return;

  readingLocked = true;
  selectedCard.classList.add("is-flipped");
  selectedCard.setAttribute("aria-pressed", "true");
  selectedCard.setAttribute("aria-label", "Carta revelada");

  readingDeck?.querySelectorAll(".reading-card").forEach((card) => {
    if (card === selectedCard) return;
    card.classList.add("is-locked");
    card.setAttribute("aria-disabled", "true");
    card.tabIndex = -1;
  });

  const index = Number(selectedCard.dataset.index || 0);
  const work = readingWorks[index];
  if (readingStatus) {
    readingStatus.textContent = work
      ? `A mesa revelou “${work.title}”. As outras cartas permanecem fechadas.`
      : "A carta foi revelada. As outras permanecem fechadas.";
  }
  if (newReadingButton) newReadingButton.hidden = false;
}

async function loadReading({ excludeCurrent = false } = {}) {
  if (!readingDeck) return;

  const previousWorks = readingWorks;
  const params = new URLSearchParams();
  if (excludeCurrent) {
    previousWorks.forEach((work) => params.append("exclude", work.title));
  }

  readingLocked = true;
  readingDeck.replaceChildren();
  if (readingStatus) readingStatus.textContent = "Embaralhando o acervo…";
  if (newReadingButton) {
    newReadingButton.hidden = true;
    newReadingButton.setAttribute("disabled", "disabled");
  }

  try {
    const suffix = params.toString() ? `?${params.toString()}` : "";
    const works = await jsonFetch(`/api/v1/recommendations/works${suffix}`, {
      method: "GET",
      cache: "no-store",
    });

    if (!Array.isArray(works) || works.length !== 3) {
      throw new Error("A leitura não encontrou três obras disponíveis.");
    }

    renderReading(works);
  } catch (error) {
    readingWorks = previousWorks;
    readingLocked = false;

    if (previousWorks.length === 3) {
      renderReading(previousWorks);
      if (readingStatus) readingStatus.textContent = "Não foi possível trocar as cartas agora.";
    } else if (readingStatus) {
      readingStatus.textContent = "A mesa está indisponível no momento. Tente novamente.";
    }

    if (newReadingButton) newReadingButton.hidden = false;
    showToast(error.message);
  } finally {
    newReadingButton?.removeAttribute("disabled");
  }
}

newReadingButton?.addEventListener("click", () => loadReading({ excludeCurrent: true }));

loadReading();
