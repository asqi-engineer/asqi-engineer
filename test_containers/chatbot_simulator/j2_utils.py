from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape

prompt_dir = Path("prompts/")
env = Environment(
    loader=FileSystemLoader(str(prompt_dir)), autoescape=select_autoescape()
)


def render_prompt(template_name: str, context: dict[str, Any] | None = None) -> str:
    if context is None:
        context = {}
    try:
        template = env.get_template(template_name)
    except TemplateNotFound as e:
        raise TemplateNotFound(
            f"Template '{template_name}' not found in {prompt_dir}"
        ) from e
    return template.render(context)
