from setuptools import setup, find_packages

setup(
    name="Topsis-Anerudh-102203042",
    version="0.1",
    author="AnerudhParthiShyam",
    author_email="anerudhs@gmail.com",
    description="A Python package to perform TOPSIS analysis from CLI.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Topsis-Anerudh-102203042",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas"
    ],
    entry_points={
        "console_scripts": [
            "topsis=topsis_anerudh.topsis_cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
