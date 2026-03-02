FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for document parsing (if needed by PyMuPDF/pdfplumber)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose ports for FastAPI (8000) and Gradio (7860)
EXPOSE 8000 7860

# Start the application
CMD ["python", "app.py"]
