from setuptools import setup, find_packages

setup(
    name="enmed",
    version="1.0.0",
    description="Cross-Lingual Domain Adaptation and Multi-Task Fine-Tuning for French Medical LLMs",
    author="B. D, Abodo and V. Malykh",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
)
