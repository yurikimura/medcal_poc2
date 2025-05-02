FROM python:3.10-slim

WORKDIR /app

# Install dependencies for LangChain and PDF processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .

# Expose the port Streamlit will run on
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]