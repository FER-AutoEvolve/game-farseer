# Use official Python slim image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install substenv (for environment variable substitution)
RUN apt-get update && apt-get install -y gettext-base curl

# Copy requirements and install Python dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

# Copy project files
COPY src/ ./src/
COPY configuration.template.json ./configuration.template.json

# Install system dependencies
RUN pip install --upgrade pip && \
    pip install -r ./requirements.txt

# Set arguments for configuration (can be overridden at build or run time)
ARG HOST=0.0.0.0 \
    PORT=8000 \
    LLM_API_KEY="" \
    LLM_MODEL="gpt-4.1" \
    LLM_TEMPERATURE=0.2 \
    LLM_TIMEOUT_SECONDS=120 \
    LLM_URL="" \
    LLM_HEADERS="" \
    CODE_OVERSEER_CONFIGURED=true \
    CODE_OVERSEER_ENDPOINT="http://path-to-overseer-endpoint:port/code-change" \
    KEYPOINT_NOTIFICATION_ENABLED="true" \
    KEYPOINT_NOTIFICATION_ENDPOINT="http://game-web-wrapper:8001/notify-of-event" \
    EXPERIMENT_NOTIFICATION_ENABLED="true" \
    EXPERIMENT_NOTIFICATION_ENDPOINT="http://experiment-director:8002/notify" \
    EXPERIMENT_NOTIFICATION_COMPONENT_NAME="GAME_FARSEER"

# Set environment variables for substitution
ENV HOST=${HOST}
ENV PORT=${PORT}
ENV LLM_API_KEY=${LLM_API_KEY}
ENV LLM_MODEL=${LLM_MODEL}
ENV LLM_TEMPERATURE=${LLM_TEMPERATURE}
ENV LLM_URL=${LLM_URL}
ENV LLM_TIMEOUT_SECONDS=${LLM_TIMEOUT_SECONDS}
ENV LLM_HEADERS=${LLM_HEADERS}
ENV CODE_OVERSEER_CONFIGURED=${CODE_OVERSEER_CONFIGURED}
ENV CODE_OVERSEER_ENDPOINT=${CODE_OVERSEER_ENDPOINT}
ENV KEYPOINT_NOTIFICATION_ENABLED=${KEYPOINT_NOTIFICATION_ENABLED}
ENV KEYPOINT_NOTIFICATION_ENDPOINT=${KEYPOINT_NOTIFICATION_ENDPOINT}
ENV EXPERIMENT_NOTIFICATION_ENABLED=${EXPERIMENT_NOTIFICATION_ENABLED}
ENV EXPERIMENT_NOTIFICATION_ENDPOINT=${EXPERIMENT_NOTIFICATION_ENDPOINT}
ENV EXPERIMENT_NOTIFICATION_COMPONENT_NAME=${EXPERIMENT_NOTIFICATION_COMPONENT_NAME}

# Generate configuration.json from template
RUN envsubst < /app/configuration.template.json > /app/configuration.json
RUN rm /app/configuration.template.json

# Expose port
EXPOSE ${PORT}

# Set entrypoint (override as needed)
CMD ["python", "src/api.py"]
