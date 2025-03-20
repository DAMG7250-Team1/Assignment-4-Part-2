from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="nvidia-rag-pipeline",
    version="0.1.0",
    author="NVIDIA RAG Pipeline Developers",
    author_email="your.email@example.com",
    description="A RAG pipeline for retrieving and analyzing NVIDIA financial reports",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nvidia-rag-pipeline",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nvidia-rag=examples.nvidia_rag_example:main",
        ],
    },
) 