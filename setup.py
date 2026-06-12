# installing the QA folder as a local package in a .venv
from setuptools import find_packages, setup

setup(
    name="QAsystem using haystack-mistrialai-pinecore-fastapi",
    version="0.0.1",
    author="pratik",
    author_email="pratikvsalunkhe924@gmail.com",
    packages=find_packages(),
    install_requires=["pinecone-haystack", "google-ai-haystack", "fastapi", "uvicorn", "python-dotenv", "pathlib"]
)