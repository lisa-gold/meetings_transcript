FROM python:3.11-slim@sha256:193fdd0bbcb3d2ae612bd6cc3548d2f7c78d65b549fcaa8af75624c47474444d

# Install system dependencies
# ffmpeg is required for pydub and audio processing
# libsndfile1 is often required for audio libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for data persistence if they don't exist
RUN mkdir -p voice_samples

# Set environment variable for the port (default)
ENV PORT=8000

# Run the application
CMD ["python", "main.py"]

