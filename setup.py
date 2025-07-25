"""
JECO Route Processor – installation script
-----------------------------------------

Allows `pip install -e .` so that IDE‑integrated AI tools (OpenAI Codex,
GitHub Copilot Chat, etc.) can import the package, run tests, and modify
code in‑place without extra configuration.

Assumes the import root is a directory named `jeco_route_processor/`
containing an `__init__.py`.  If your package directory is named
differently, edit the `PACKAGE_NAME` constant below.
"""
from pathlib import Path
from setuptools import setup, find_packages

# --------------------------------------------------------------------------- #
# Editable variables                                                          #
# --------------------------------------------------------------------------- #
PACKAGE_NAME = "jeco_route_processor"          # ← directory name on disk
VERSION = "0.1.0"
AUTHOR = "Jonathan Norris"
AUTHOR_EMAIL = "jnorris@jeco.com"
DESCRIPTION = "JECO Route Processing tools"
PYTHON_REQUIRES = ">=3.11"

# Core/runtime dependencies (edit as required)
INSTALL_REQUIRES = [
    "pandas>=1.5",
    "scipy>=1.10",
    # "numpy>=1.23",     # uncomment / adjust if you need it
]

# Extra groups for development / CI
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0",
        "black>=24.3",
        "ruff>=0.4",
        # add mypy, coverage, etc. if desired
    ]
}

# --------------------------------------------------------------------------- #
# Helper: read long description from README.md                                #
# --------------------------------------------------------------------------- #
this_dir = Path(__file__).resolve().parent
readme_path = this_dir / "README.md"
LONG_DESCRIPTION = readme_path.read_text(encoding="utf-8") if readme_path.exists() else DESCRIPTION

# --------------------------------------------------------------------------- #
# Call setup()                                                                #
# --------------------------------------------------------------------------- #
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires=PYTHON_REQUIRES,
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,          # include non‑Python files declared in MANIFEST.in
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
