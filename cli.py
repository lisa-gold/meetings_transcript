import argparse

from constants import Models


parser = argparse.ArgumentParser(description="Process an audio file.")
parser.add_argument("audio_file", help="Name of the audio file in the 'input' folder")
parser.add_argument("--model", default=Models.TINY, help=f"Model to use (default: {Models.TINY})")
parser.add_argument("--lang", default=None, help="Specify language of the audio. E.g. en")
