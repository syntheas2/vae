from setuptools import setup, find_packages

setup(
    name="vae_syntheas",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    author="virseli",
    author_email="your.email@example.com",
    description="VAE Syntheas",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="==3.11.*",
)