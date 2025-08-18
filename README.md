# Meetings Transcriptor And Tasks Allocator

A tool that helps transcribe meetings, summarize them and allocate tasks from the transcriptions.

## Description

This application transcribes meeting recordings, summarizes them and helps identify and allocate tasks from the transcribed content. 
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
4. Create token on hugging face
   - [hf.co/settings/tokens](https://huggingface.co/settings) Choose permissions: `Read access to contents of all public gated repos you can access`
   - Set environmental variables (see [example](.env.example)):
     ```
     $ touch .env
     $ echo "HG_TOKEN=<your_token>" > .env
     ```
5. 

## Usage

Save your audio file in the directory input/
Run the main script with the required arguments:
- file_name (with extension)
- choose the model name from the list: tiny, base, small, medium, large, turbo

```bash
python main.py <file_name> <model>
```

After script completion, you can find a txt file with meeting transcript in directory output/

# Improvements
1. Handle wrong audio file extensions
2. Add speaker alignments, use pyannote.audio, send a segmented .wav to OpenAI Wisper
3. Send the resulting transcript to an AI Model to summarize it. Create JSON with tasks defined from the transcript
4. Send tasks to email
5. Make an endpoint that will receive the audio file and start the job and respond with the summery
