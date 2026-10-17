// A mesa cuida da interação; o servidor guarda a tiragem de cada conversa.
(() => {
  const dialog = document.querySelector("#guide-draw");
  if (!dialog) return;
  const cards = [...dialog.querySelectorAll("[data-guide-card]")];
  const proceed = dialog.querySelector("#guide-continue");
  const neutral = dialog.querySelector("#guide-neutral");
  const errorEl = dialog.querySelector("#guide-draw-error");
  const title = dialog.querySelector("#guide-result-title");
  const description = dialog.querySelector("#guide-result-description");
  const guides = {
    cosmo: { name: "Cosmo" },
    damiao: { name: "Damião" },
  };
  let flow = null;
  let resolveFlow = null;
  let callbacks = null;
  let state = null;
  let busy = false;

  function setBusy(value) {
    busy = value;
    cards.forEach((card) => { card.disabled = value || state !== null; });
    neutral.disabled = value;
    proceed.disabled = value || state === null;
    dialog.setAttribute("aria-busy", String(value));
  }

  function finish(nextState) {
    dialog.close();
    resolveFlow?.(nextState);
    resolveFlow = null;
  }

  async function pick(index) {
    if (busy || state) return;
    setBusy(true);
    errorEl.textContent = "";
    try {
      const nextState = await callbacks.draw(index);
      // Outra aba pode ter encerrado a tiragem em modo Neutro.
      if (!nextState.drawn_guide && !nextState.draw_required) {
        finish(nextState);
        return;
      }
      const guide = guides[nextState.drawn_guide];
      const card = cards[nextState.selected_card];
      if (!guide || !card) throw new Error("Não foi possível revelar a carta. Tente novamente.");
      state = nextState;
      const front = card.querySelector(".guide-card-front");
      const portrait = front.querySelector(".saint-portrait");
      portrait.dataset.saint = nextState.drawn_guide;
      portrait.setAttribute("aria-label", `Retrato de ${guide.name} em medalhão dourado`);
      front.querySelector("strong").textContent = guide.name;
      front.setAttribute("aria-hidden", "false");
      card.classList.add("is-revealed");
      card.setAttribute("aria-pressed", "true");
      card.setAttribute("aria-label", `Carta escolhida: ${guide.name}`);
      cards.filter((other) => other !== card).forEach((other) => other.classList.add("is-unselected"));
      title.textContent = `${guide.name} guia esta consulta`;
      description.textContent = "Você pode trocar o guia no alto do chat.";
      proceed.hidden = false;
      neutral.textContent = "Começar no Neutro";
      setBusy(false);
      proceed.focus({ preventScroll: true });
    } catch (error) {
      errorEl.textContent = error.message || "Não foi possível salvar a escolha. Tente novamente.";
    } finally {
      setBusy(false);
    }
  }

  async function chooseNeutral() {
    if (busy) return;
    setBusy(true);
    errorEl.textContent = "";
    try {
      finish(await callbacks.neutral());
    } catch (error) {
      errorEl.textContent = error.message || "Não foi possível abrir a consulta. Tente novamente.";
    } finally {
      setBusy(false);
    }
  }

  cards.forEach((card, index) => card.addEventListener("click", () => pick(index)));
  neutral.addEventListener("click", chooseNeutral);
  proceed.addEventListener("click", () => { if (state && !busy) finish(state); });
  dialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    if (!busy) chooseNeutral();
  });

  window.CartomanteDraw = {
    open(handlers) {
      if (flow) return flow;
      callbacks = handlers;
      state = null;
      errorEl.textContent = "";
      title.textContent = "Duas cartas. Uma companhia para a leitura.";
      description.textContent = "O verso é igual; o encontro fica por conta da sua escolha.";
      neutral.textContent = "Prefiro começar no Neutro";
      proceed.hidden = true;
      cards.forEach((card, index) => {
        card.classList.remove("is-revealed", "is-unselected");
        card.setAttribute("aria-pressed", "false");
        card.setAttribute("aria-label", `Escolher a ${index === 0 ? "primeira" : "segunda"} carta`);
        const front = card.querySelector(".guide-card-front");
        front.setAttribute("aria-hidden", "true");
        const portrait = front.querySelector(".saint-portrait");
        delete portrait.dataset.saint;
        portrait.removeAttribute("aria-label");
        front.querySelector("strong").textContent = "";
      });
      setBusy(false);
      flow = new Promise((resolve) => { resolveFlow = resolve; });
      dialog.showModal();
      return flow.finally(() => { flow = null; });
    },
  };
})();
