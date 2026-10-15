// Ajuda compartilhada pela home e pela consulta. O diálogo só é criado ao abrir.
const helpSections = [
  ["Chat", "Faça uma pergunta sobre Machado de Assis. A Cartomante recupera passagens do acervo e constrói a resposta a partir delas."],
  ["Leitura", "Em ‘Tirar uma leitura’, escolha uma das três cartas para descobrir uma obra. A carta revelada permite abrir o texto no Domínio Público."],
  ["Cosmo e Damião", "Neutro é o padrão. Cosmo e Damião são santos-guias que induzem diferentes interpretações de questões ambíguas, sem mudar as passagens consultadas nem transformar opiniões em fatos."],
  ["Cartas de evidência", "Em uma resposta, use ‘Mostre-me as evidências’ para virar cartas com trechos originais, títulos e seções das obras. Elas permitem conferir a base textual da leitura."],
];

function getHelpDialog() {
  let dialog = document.querySelector("#help-dialog");
  if (dialog) return dialog;

  dialog = document.createElement("dialog");
  dialog.id = "help-dialog";
  dialog.className = "help-dialog";
  dialog.setAttribute("aria-labelledby", "help-dialog-title");

  const sections = helpSections.map(([title, content]) => `
    <section class="help-dialog-section">
      <h3>${title}</h3>
      <p>${content}</p>
    </section>
  `).join("");

  dialog.innerHTML = `
    <header class="help-dialog-header">
      <div>
        <p class="help-dialog-kicker">Guia da Cartomante</p>
        <h2 id="help-dialog-title">Como funciona</h2>
      </div>
      <button type="button" class="help-dialog-close" aria-label="Fechar ajuda">×</button>
    </header>
    <p class="help-dialog-intro">Converse com o acervo, consulte as cartas e escolha como interpretar as pistas.</p>
    ${sections}
  `;
  document.body.appendChild(dialog);
  dialog.querySelector(".help-dialog-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  return dialog;
}

document.querySelectorAll("[data-help-open]").forEach((button) => {
  button.addEventListener("click", () => getHelpDialog().showModal());
});
