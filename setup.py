from setuptools import setup, find_packages

setup(
    name="tinyssl",
    version="0.1.0",
    description="Distill DINOv2 features into compact vision models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Emran Abdu",
    author_email="jakeniel98@gmail.com",
    url="https://github.com/Emran-goat/tinyssl",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
    ],
    extras_require={
        "dev": ["pytest", "black", "ruff", "mypy"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
