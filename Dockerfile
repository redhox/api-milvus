# Dockerfile pour l'API FastAPI
FROM python:3.10
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get update && apt-get install -y python3-opencv
# Répertoire de travail
WORKDIR /app

ARG REQUIREMENTS_FILE=requirements.txt

# Copier les fichiers de dépendances et installer les packages
COPY $REQUIREMENTS_FILE /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pydantic_settings
# Copier les fichiers source de l'application
COPY app /app/app
#COPY ../.env /app/.env

# Exposer le port utilisé par l'API (par exemple 8000)
EXPOSE 8002

# Commande pour démarrer l'API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]