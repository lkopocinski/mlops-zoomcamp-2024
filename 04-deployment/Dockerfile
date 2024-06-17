FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install & use pipenv
ENV PIPENV_VENV_IN_PROJECT 1
COPY Pipfile Pipfile.lock ./
RUN python -m pip install --upgrade pip
RUN pip install pipenv && pipenv install --dev --system --deploy
ENV PATH="/.venv/bin:$PATH"

# Creates a non-root user and adds permission to access the /app folder
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

COPY [ "homework/starter.py", "starter.py" ]
ENTRYPOINT ["python", "starter.py"]