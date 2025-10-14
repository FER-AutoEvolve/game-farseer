# Use official Python slim image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install substenv (for environment variable substitution)
RUN pip install substenv

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY configuration.template.json ./configuration.template.json

# Set environment variables (example, override at runtime)
ENV HOST=0.0.0.0
ENV PORT=8000
ENV OPEN_AI_API_KEY="changeme"
ENV OPEN_AI_MODEL="gpt-4.1"
ENV TEMPERATURE=0.7
ENV CODE_OVERSEER_CONFIGURED=true
ENV CODE_OVERSEER_ENDPOINT="path-to-overseer-endpoint"

# Generate configuration.json from template using substenv
RUN substenv configuration.template.json configuration.json

# Expose port (example)
EXPOSE ${PORT}

# Set entrypoint (override as needed)
CMD ["python", "src/api.py"]
