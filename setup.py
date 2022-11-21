
import os
import pkg_resources

from setuptools import find_packages, setup

with open("README.rst") as fh:
    long_description = fh.read()

setup(
    name="whatdisay",
    version="0.0.1",
    author="Sam Hecht",
    author_email="samjulius@gmail.com",
    url="https://github.com/samjhecht/whatdisay",
    packages=find_packages('whatdisay'),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    entry_points = {
        'console_scripts': ['whatdisay=whatdisay.transcript:cli'],
    },
    description="Generate diarized transcripts from audio to markdown.",
    long_description=long_description,
    keywords=['obsidian', 'openai', 'whisper','transcription'],
)