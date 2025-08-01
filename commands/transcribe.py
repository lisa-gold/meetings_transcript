import whisper
from pathlib import Path
from constants import Models


def transcribe_audio(file_path: str, model_name: str = Models.TINY):
    """
    Транскрибирует аудиофайл с помощью Whisper.
    """
    print(f"Загрузка модели Whisper ({model_name})...")
    # Для первого запуска модель будет скачиваться, это может занять время.
    # Модели хранятся в ~/.cache/whisper
    model_transcript = whisper.load_model(model_name)
    print("Модель загружена. Начало транскрибации...")
    result = model_transcript.transcribe(file_path)
    print("\n--- Результат транскрибации ---")
    print(result["text"])
    
    # Получаем путь к входному файлу и создаем директорию output на уровень выше
    input_path = Path(file_path)
    output_dir = input_path.parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Получение имени файла и создание нового пути в директории output
    output_filename = output_dir / f"{input_path.name}.txt"
    
    # Сохранение результата в файл
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"\nТранскрипция сохранена в файл: {output_filename}")
