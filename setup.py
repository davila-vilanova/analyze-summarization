from setuptools import setup, find_packages

setup(
    name="analyze-summarization",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "model2vec==0.4.1",
        "datasets==3.5.0",
    ],
    entry_points={
        "console_scripts": [
            "analyze-summarization=analyze_summarization.cli:main",
        ],
    },
    python_requires=">=3.12",
)
