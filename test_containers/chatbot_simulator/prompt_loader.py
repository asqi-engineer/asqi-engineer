from pathlib import Path

from jinja2 import Environment, FileSystemLoader, meta

PROMPT_DIR = Path(__file__).parent / "prompts"
env = Environment(loader=FileSystemLoader(PROMPT_DIR))


def load_prompt(template_name: str, **kwargs) -> str:
    """
    Load a prompt template from the prompts/ folder.
    The template should be defined in a text file template_name.j2.
    The template should use Jinja2 format.
    The function will read the template and substitute in the provided kwargs.
    """
    return env.get_template(f"{template_name}.j2").render(**kwargs)


if __name__ == "__main__":
    # Get information about template from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--template_name", type=str, help="Name of the prompt template"
    )
    filepath = PROMPT_DIR / f"{parser.parse_args().template_name}.j2"
    template = filepath.read_text()
    print("Template content:")
    print(template + "\n\n")
    print("Variables in the template:")
    print(meta.find_undeclared_variables(env.parse(template)))
