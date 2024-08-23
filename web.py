# TODO: database access
# send to 4o directly
# shorter response [3-4 sentences]
# voice and text streaming [real-time] low latency socket: flask-socketio: https://flask-socketio.readthedocs.io/en/latest/
# Openai batch api: https://platform.openai.com/docs/guides/batch/overview
# some project links: https://www.notion.so/seemuseums/dd751e1dc0ee40228d5675f1b28925e4?pvs=4

from api import data_conversation_chatbot_chain, callbacks
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
from audiorecorder import audiorecorder
import time
from langchain_community.chat_message_histories import (
    ChatMessageHistory,
    StreamlitChatMessageHistory,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import time
from pathlib import Path
import openai
from openai import OpenAI
import os


# def get_response(input_text, stream=False):
#     print("===========in get_response===================")
#     if stream:
#         conversation_result = []
#         print("======before stream loop===========")
#         for chunk in data_conversation_chatbot_chain_with_history.stream(
#             {"question": input_text, "input": input_text},
#         ):
#             print("===========in streamif===================")
#             conversation_result.append(chunk)
#             # print(chunk, end=" ", flush=True)
#             # print(chunk)
#         return conversation_result
#
#     else:
#         response = data_conversation_chatbot_chain_with_history.invoke(
#             {
#                 "question": input_text,
#                 "input": input_text,
#             },
#         )
#         return response
#

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def stream_data_output(stream_data):
    for chunk in stream_data:
        yield chunk
        time.sleep(0.02)


def stream_to_speakers(text: str) -> None:
    """
    give response to the user's input in audio

    @param text: text to be converted to audio
    """

    start_time = time.time()

    with openai.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        response_format="mp3",
        input=text,
    ) as response:
        print("saving response audio file  ...")
        response.stream_to_file("response.mp3")

    print(f"Done in {int((time.time() - start_time) * 1000)}ms.")


st.title("Your AI Tour Guide")


initial_message = "Hi, I am your museum AI assistant. I can help you with any questions you have on museum, any questions on museum?"
chat_history = StreamlitChatMessageHistory(key="chat_history")
if "history" not in st.session_state:
    st.session_state["history"] = []

if "messages" not in st.session_state:
    st.chat_message("AI").write(initial_message)
    st.session_state["history"].append("AI: " + initial_message)

# Render current messages from StreamlitChatMessageHistory
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)


# print(result)

# audio record
audio_col, text_input_col = st.columns(2)
input_text = None

with audio_col:
    audio = audiorecorder("Click to record", "Click to stop recording")
    # translate user audio to text question
    if len(audio) > 0:
        # To play audio in frontend:
        st.audio(audio.export().read())

        # To save audio to a file, use pydub export method:
        audio.export("user_response.mp3", format="mp3")

with text_input_col:
    # input_text = st.chat_input("any questions on museum?")  # interact with openai
    input_text = st.chat_input("what do you want to know about museum?")


print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("input_text: ", input_text)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

if not input_text:
    print("trying audio...")
    # convert audio to text
    client = OpenAI()
    try:
        audio_file = open("user_response.mp3", "rb")
    except:
        print("I'm listening ...")
        st.stop()

    input_text = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file)

    # delete audio file after use
    os.remove("user_response.mp3")

    # only get str as input
    input_text = input_text.text

    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("input_text: ", input_text)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

with text_input_col:
    st.chat_message("user").write(input_text)

with st.chat_message("AI"):
    # streaming response
    with st.spinner("Thinking..."):
        response = data_conversation_chatbot_chain.invoke(
            {"input": input_text, "chat_history": st.session_state.chat_history},
            config={"configurable": {"session_id": "any"},
                    "callbacks": callbacks},
        )
        # print("response: ", response)
        chat_history.add_user_message(input_text)
        ai_response = response["tour_guide_response"]
        image_url = response["image_url"]
        chat_history.add_ai_message(ai_response)
        st.session_state["history"].append(f"You: {input_text}")
        st.session_state["history"].append(f"AI: {ai_response}")
        st.write(ai_response)
        if image_url != "":
            st.image(image_url)

        # generate audio response
        stream_to_speakers(ai_response)
        st.audio("response.mp3", format="audio/mp3", autoplay=True)

    # for message in st.session_state["history"]:
    #     st.text(message)
