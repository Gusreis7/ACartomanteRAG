// Guia compartilhado; a conclusão acompanha o cookie HttpOnly da sessão.
(() => {
  const sections = [
    ["1. Traga uma pergunta literária", "Pergunte sobre uma obra, personagem, passagem ou tema. Exemplo: ‘Como Machado trabalha o ciúme em Dom Casmurro?’. O acervo é de Machado de Assis; quando faltar base no texto, a Cartomante deve dizer isso."],
    ["2. Escolha sua leitura", "Neutro é o padrão. Cosmo e Damião são santos-guias que propõem interpretações diferentes de questões ambíguas. Uma interpretação não comprova um fato da história."],
    ["3. Confira as cartas", "Depois da resposta, clique em ‘Mostre-me as evidências’. Vire as cartas para ler as passagens originais que sustentam a leitura."],
    ["4. Continue ou descubra outra obra", "Faça perguntas de continuação na mesma consulta. ‘Nova consulta’ abre outro assunto; ‘Tirar uma leitura’ sorteia uma obra para você conhecer."],
  ];
  let completed = false;
  let readyPromise = null;
  let resolveTutorial = null;

  async function request(url, method = "GET") {
    const response = await fetch(url, { method, credentials: "same-origin" });
    if (!response.ok) throw new Error("Não foi possível carregar o guia desta sessão. Tente novamente.");
    return response.json();
  }

  function getDialog() {
    let dialog = document.querySelector("#help-dialog");
    if (dialog) return dialog;
    dialog = document.createElement("dialog");
    dialog.id = "help-dialog";
    dialog.className = "help-dialog";
    dialog.setAttribute("aria-labelledby", "help-dialog-title");
    dialog.innerHTML = `
      <header class="help-dialog-header">
        <div><p class="help-dialog-kicker">Guia da Cartomante</p><h2 id="help-dialog-title">Sua primeira consulta</h2></div>
        <button type="button" class="help-dialog-close" aria-label="Fechar ajuda">×</button>
      </header>
      <p class="help-dialog-intro">Leia, pergunte e confira de onde veio a resposta.</p>
      <div class="help-dialog-sections"></div>
      <p class="help-privacy-note">As consultas são registradas para histórico e avaliação do serviço. Evite dados pessoais. Pedidos fora da literatura ou de alteração das regras serão recusados.</p>
      <p class="help-dialog-error" role="status"></p>
      <button class="button button-primary help-dialog-start" type="button">Entendi, começar</button>
    `;
    for (const [title, content] of sections) {
      const section = document.createElement("section");
      section.className = "help-dialog-section";
      const heading = document.createElement("h3");
      heading.textContent = title;
      const paragraph = document.createElement("p");
      paragraph.textContent = content;
      section.append(heading, paragraph);
      dialog.querySelector(".help-dialog-sections").appendChild(section);
    }
    document.body.appendChild(dialog);
    const finish = async () => {
      const button = dialog.querySelector(".help-dialog-start");
      button.disabled = true;
      try {
        if (!completed) await request("/api/v1/session/tutorial", "POST");
        completed = true;
        dialog.close();
        resolveTutorial?.();
        resolveTutorial = null;
      } catch (error) {
        dialog.querySelector(".help-dialog-error").textContent = error.message;
      } finally {
        button.disabled = false;
      }
    };
    dialog.querySelector(".help-dialog-start").addEventListener("click", finish);
    dialog.querySelector(".help-dialog-close").addEventListener("click", () => dialog.close());
    dialog.addEventListener("cancel", (event) => { if (!completed) event.preventDefault(); });
    dialog.addEventListener("click", (event) => { if (completed && event.target === dialog) dialog.close(); });
    return dialog;
  }

  async function ensure() {
    if (!readyPromise) {
      readyPromise = (async () => {
        const state = await request("/api/v1/session");
        completed = Boolean(state.tutorial_completed);
        if (completed) return;
        const dialog = getDialog();
        dialog.querySelector(".help-dialog-close").hidden = true;
        await new Promise((resolve) => {
          resolveTutorial = resolve;
          dialog.showModal();
        });
      })().catch((error) => { readyPromise = null; throw error; });
    }
    return readyPromise;
  }

  window.CartomanteGuide = { ensure };
  document.querySelectorAll("[data-help-open]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await ensure();
        const dialog = getDialog();
        dialog.querySelector("#help-dialog-title").textContent = "Como funciona";
        dialog.querySelector(".help-dialog-close").hidden = false;
        dialog.querySelector(".help-dialog-start").textContent = "Voltar à leitura";
        if (!dialog.open) dialog.showModal();
      } catch (_) {
        button.textContent = "Tentar abrir ajuda";
      }
    });
  });
})();
