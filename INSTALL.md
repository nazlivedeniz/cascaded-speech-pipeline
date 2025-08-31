# Python 3.10 Project Setup Guide

1- Install system dependencies:

- `ffmpeg` is a tool for audio/video processing and required by Python packages like `pydub`
- `libsndfile` is required for `soundfile` Python package.

Here is how to install them in different operating systems:

On Ubuntu:
```
sudo apt update
sudo apt install ffmpeg libsndfile1
```
On macOS:
```
brew install ffmpeg libsndfile
```

Additionally, install `mecab` and `mecab-ipadic` required by CoquiTTS (`TTS`). They are actually 
for Japanese functionality, but if they are missing the installation of CoquiTTS library fails.
```
brew install mecab mecab-ipadic
```

2- Assume that you have a parent folder called `Python`. Prepare the project folders as follows: 
```
cd Python
mkdir .envs
mkdir -p projects/cascaded-speech-pipeline
```
and copy the project content under the project folder (`projects/cascaded-speech-pipeline`).

3- Create a virtual environment and activate it:
```
python3.10 -m venv .envs/cascaded-speech-pipeline
source .envs/cascaded-speech-pipeline/bin/activate
```

4- Install the requirements for the project:
```
cd projects/cascaded-speech-pipeline
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip freeze > installed_requirements.txt
```

See [installed_requirements.txt](./installed_requirements.txt) file which is generated to 
capture exact versions of the installed packages.

5- Note that, in order to check the code against PEP8 standards, `flake8` package is also installed.
In order to check PEP8 compliance of a script, run the code below. Since the default 
max-line-length of 79 seems very narrow, it is set it to 100 for this repo.
```
flake8 <test_script>.py --max-line-length=100
```
