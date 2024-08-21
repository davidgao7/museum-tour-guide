FROM python:3.12.3
USER root

ENV PORT=8501
ENV HOST=0.0.0.0

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    portaudio19-dev \ 
    python3-pyaudio

RUN pip3 install --upgrade pip && \
    pip3 install pyaudio==0.2.14 \
    openai==1.37.1 \
    langchain==0.2.6 \
    langchain-community==0.2.1 \
    langchain-core==0.2.23 \
    langchain-experimental==0.0.59 \
    langchain-openai==0.1.13 \
    langchain-text-splitters==0.2.0 \
    streamlit==1.36.0 \
    streamlit-audiorecorder==0.0.5 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /streamlit-app

EXPOSE ${PORT}

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

COPY tags_replaced.csv TTS/
COPY web.py .

ENTRYPOINT ["sh", "-c", "streamlit run web.py --server.port ${PORT} --server.address ${HOST}"]
