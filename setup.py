from pathlib import Path

from setuptools import setup, find_packages

def _get_version():
    import re
    version_file = Path(__file__).parent / "tinyssl" / "__init__.py"
    match = re.search(r'__version__\s*=\s*"([^"]+)"', version_file.read_text())
    return match.group(1) if match else "0.1.0"


HERE = Path(__file__).parent
version = _get_version()
long_description = (HERE / "README.md").read_text()
requirements = (HERE / "requirements.txt").read_text().strip().splitlines()

setup(
    name="tinyssl",
    version=version,
    description=(
        "PyTorch code and models for the TinySSL self-supervised knowledge "
        "distillation method."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Emran Abdu",
    author_email="jakeniel98@gmail.com",
    url="https://github.com/Emran-goat/tinyssl",
    project_urls={
        "Paper": "https://github.com/Emran-goat/tinyssl/blob/main/paper/main.tex",
        "Hugging Face": "https://huggingface.co/tinyssl",
        "X (Twitter)": "https://x.com/Emran_py",
        "LinkedIn": "https://www.linkedin.com/in/emran-abdu-833981315/",
    },
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7", "black>=23", "ruff>=0.1", "mypy>=1.0"],
        "notebooks": ["jupyter", "matplotlib", "seaborn"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
