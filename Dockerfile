FROM python:3.12.3
USER root

RUN apt-get update -y && \
    apt-get -y install portaudio19-dev python3-pyaudio && \
    pip install --upgrade pip && \
    pip install langchain==0.2.6 \
    langchain-community==0.2.1 \
    langchain-core==0.2.23 \
    langchain-experimental==0.0.59 \
    langchain-text-splitters==0.2.0 \
    langchain-openai==0.1.13 \
    openai==1.37.1 \
    pyaudio==0.2.14 \
    streamlit==1.36.0 \
    streamlit-audiorecorder==0.0.5

WORKDIR streamlit-app/
COPY api.py web.py streamlit-app/

EXPOSE 8080
ENTRYPOINT ["streamlit", "run", "web.py", "--server.port 8080"]
