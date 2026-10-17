"""Markdown da resposta: HTML e imagens externos não são interpretados."""
from markdown_it import MarkdownIt


_renderer = MarkdownIt("js-default", {"html": False, "linkify": False}).disable("image")


def _link_open(tokens, index, options, env):
    tokens[index].attrSet("target", "_blank")
    tokens[index].attrSet("rel", "noopener noreferrer")
    return _renderer.renderer.renderToken(tokens, index, options, env)


_renderer.renderer.rules["link_open"] = _link_open


def render_answer(text: str) -> str:
    return _renderer.render(text)
