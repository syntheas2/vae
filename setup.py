from setuptools import setup, find_packages
import os

# Read README if it exists, otherwise use description
long_description = "VAE Syntheas"
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="vae_syntheas",
    version="0.1.0",
    packages=find_packages(include=['vae_syntheas', 'vae_syntheas.*']),
    install_requires=[],
    author="virseli",
    author_email="your.email@example.com",
    description="VAE Syntheas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11,<3.12",
)