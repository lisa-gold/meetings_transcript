# Meetings Transcriptor And Tasks Allocator

A tool that helps transcribe meetings, summarize them and allocate tasks from the transcriptions.

## Description

This application transcribes meeting recordings, maps speakers to their names, summarizes them and helps identify and allocate tasks from the transcribed content. 
It uses [OpenAI Whisper model](https://github.com/openai/whisper) for speech-to-text transcription.

## Prerequisites

- Python 3.11 or higher
- ffmpeg `apt install ffmpeg`
- Operating System: macOS/Linux
- Sufficient disk space for model downloads and audio processing
- Audio files should be in a supported format (WAV, MP3, etc.)


## Installation

1. Clone this repository:
```bash
$ git clone git@github.com:lisa-gold/meetings_transcript.git
$ cd meetings-transcriptor
```

2. Set up: Create and activate a virtual environment, install dependencies
```bash
$ make setup
```

3. Accept user condition
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/embedding](https://huggingface.co/pyannote/embedding)

4. Create token on hugging face
   - [hf.co/settings/tokens](https://huggingface.co/settings) Choose permissions: `Read access to contents of all public gated repos you can access`
   - Set environmental variables (see [example](.env.example)):
     ```
     $ touch .env
     $ echo "HG_TOKEN=<your_token>" > .env
     $ echo "PORT=8000" > .env
     ```
## Start the app

```bash
python main.py
```
It will start FastApi app on port defined in .env

## Preparations
1. Make several short and clean (without background noise) audio files per speaker
2. Send POST request to /add_speaker with a voice sample and speaker name
   - For a new speaker it creates a subfolder with speaker name. speaker_name references the speaker name that will be saved in the database. 
   - It has to be unique and without spaces and other special characters.
   - IMPORTANT: Use speaker names consistently! Don't change them after the first run (subdirectory names and on command run) as they are fixed in the database.
3. Now your database stores embeddings for each speaker.
4. To check list of speakers you have stored send GET request to /speakers

## Usage
Send POST request to /transcribe
- audio file
- optionally choose the model name from the list: tiny, base, small, medium, large, turbo
- optionally define language of the audio
`{"file": file, "model": "tiny", "language": "ru"}`

If you run this for the first time with a specified model, it will take some time to upload a model
Models are stored in ~/.cache/whisper


# Improvements
1. Send the resulting transcript to an AI Model to summarize it. Create JSON with tasks defined from the transcript
2. Send tasks to email
3. Make improvements to text recognitions (remove background noise, etc.)
