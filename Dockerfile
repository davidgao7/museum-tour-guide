FROM python:3.12.3
USER root

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    portaudio19-dev \ 
    python3-pyaudio

RUN git clone https://github.com/davidgao7/museum-tour-guide.git streamlit-app/ && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /streamlit-app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

COPY tags_replaced.csv TTS/
COPY web.py .

ENTRYPOINT ["streamlit", "run", "web.py" , "--server.port=8501", "--server.address=0.0.0.0"]
