import os
import pickle
import sys

from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import torch
import numpy as np

from db_accessor import DBAccessor
from utils import convert_audio_file


def add_sample_record(speaker_name: str, file_path: str):
    # Device setup (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_model = PretrainedSpeakerEmbedding(
        "pyannote/embedding",
        device=device,
        use_auth_token=os.getenv("HG_TOKEN")
    )

    vector = build_averaged_vector(file_path, embedding_model, speaker_name)
    save_vector_in_db(speaker_name, vector)

# Function to get speaker embedding
def get_embedding(file_path: str, embedding_model):
    print(f'Get embedding for {file_path}')
    audio = Audio()
    waveform, sample_rate = audio(file_path)
    return embedding_model(waveform[None])


def build_averaged_vector(file_path: str, embedding_model, speaker_name: str):
    embeddings = []
    speaker_dir = os.path.dirname(file_path)
    for filename in os.listdir(speaker_dir):
        full_path = os.path.join(speaker_dir, filename)

        # Skip directories
        if os.path.isdir(full_path):
            continue

        if filename.lower().endswith('.wav'):
            emb = get_embedding(full_path, embedding_model)
            embeddings.append(emb)
        else:
            converted_path = convert_audio_file(full_path)
            os.remove(full_path)
            emb = get_embedding(converted_path, embedding_model)
            embeddings.append(emb)

    if embeddings:
        avg_emb = np.mean(embeddings, axis=0)
        print(f"  - {speaker_name}: {len(embeddings)} samples averaged")
        return avg_emb
    else:
        print(f"  - {speaker_name}: no samples found")
        return None

def save_vector_in_db(speaker_name: str, vector: list):
    print(f'Save vector in db for {speaker_name}')
    blob = pickle.dumps(vector)
    db_accessor = DBAccessor('speakers.db')
    db_accessor.execute_query("""CREATE TABLE IF NOT EXISTS speakers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(100) NOT NULL UNIQUE,
        embedding BLOB NOT NULL)
    """)
    speaker = db_accessor.cursor.execute(f"SELECT * FROM speakers WHERE name = '{speaker_name}'").fetchone()
    if speaker:
        db_accessor.execute_query_with_params("UPDATE speakers SET embedding = ? WHERE name = ?",
                                              (blob, speaker_name))
    else:
        db_accessor.execute_query_with_params("INSERT INTO speakers (name, embedding) VALUES (?, ?)",
                                              (speaker_name, blob))


def show_speakers():
    print("=== Show speakers in the DB ===")
    db_accessor = DBAccessor('speakers.db')
    db_accessor.execute_query("SELECT id, name FROM speakers")
    rows = db_accessor.cursor.fetchall()
    for row in rows:
        print(row)


def clear_speakers_db():
    db_accessor = DBAccessor('speakers.db')
    db_accessor.execute_query("DELETE FROM speakers")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        speaker_name = sys.argv[1]
        file_name = sys.argv[2]
        current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        voice_file = os.path.join(current_path, 'voice_samples', speaker_name, file_name)
        add_sample_record(speaker_name, voice_file)

        show_speakers()
    else:
        print(f"Использование: python add_sample_record.py <имя_человека_без_пробелов> <имя_файла>")

