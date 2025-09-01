import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "asqi-engineer"
copyright = "2025, Resaro AI"
author = "Resaro AI"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "myst_parser",
    "sphinxcontrib.typer",
    "sphinx_click",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_contributors",
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_theme_options = {
    "accent_color": "jade",
    "github_url": "https://github.com/asqi-engineer/asqi-engineer",
    "nav_links": [
        {
            "title": "GitHub",
            "url": "https://github.com/asqi-engineer/asqi-engineer",
            "external": True,
        },
        {
            "title": "Examples",
            "url": "examples",
            "children": [
                {
                    "title": "Quick Start",
                    "url": "quickstart",
                    "summary": "Get started with ASQI in minutes",
                },
                {
                    "title": "Configuration",
                    "url": "configuration",
                    "summary": "Configure systems, suites, and score cards",
                },
                {
                    "title": "LLM Test Containers",
                    "url": "llm-test-containers",
                    "summary": "Pre-built LLM testing frameworks",
                },
                {
                    "title": "Custom Containers",
                    "url": "custom-test-containers",
                    "summary": "Create your own test containers",
                },
            ],
        },
    ],
    "globaltoc_expand_depth": 2,
    "toctree_collapse": False,
    "discussion_url": "https://github.com/asqi-engineer/asqi-engineer/discussions",
}

html_context = {
    "source_type": "github",
    "source_user": "asqi-engineer",
    "source_repo": "asqi-engineer",
    "source_version": "main",
    "source_docs_path": "/docs/",
}

html_static_path = ["_static"]

# AutoAPI configuration
autoapi_dirs = ["../src"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
