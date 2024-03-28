import os
from dotenv import load_dotenv
import setuptools

load_dotenv()


with open(file="README.md", mode="r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

GITHUB_REPO_NAME = os.getenv(key="GITHUB_REPO_NAME")
GITHUB_USER_NAME = os.getenv(key="GITHUB_USER_NAME")
GITHUB_USER_EMAIL = os.getenv(key="GITHUB_USER_EMAIL")


setuptools.setup(
    name="hate-speech-classifier",
    version=__version__,
    author=GITHUB_USER_NAME,
    author_email=GITHUB_USER_EMAIL,
    description="End to End NLP Project",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{GITHUB_USER_NAME}/{GITHUB_REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{GITHUB_USER_NAME}/{GITHUB_REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)