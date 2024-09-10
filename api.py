from openai import OpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import (
    ChatMessageHistory,
    StreamlitChatMessageHistory,
)
from langchain.chains import ConversationChain
from audiorecorder import audiorecorder
import openai
import time
import os

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import LangChainTracer
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
import langchain_core
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_experimental.agents.agent_toolkits import (
    create_csv_agent,
    # create_pandas_dataframe_agent,
)
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage

import validators

load_dotenv()
# enable tracing when debugging locally
if os.path.isfile(".env"):
    callbacks = [
        LangChainTracer(
            project_name="AI Voice Tour",
            client=Client(
                api_url=os.getenv("LANGCHAIN_ENDPOINT"),
                api_key=os.getenv("LANGCHAIN_API_KEY"),
            ),
        )
    ]
else:
    callbacks = []


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


st.title("Your AI Meseum Tour Guide 🎨🖼️🖌️")

key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# check if key is valid
if not key.startswith("sk-"):
    st.warning("Please enter your OpenAI API key!", icon="⚠")
if key:
    os.environ["OPENAI_API_KEY"] = key
    st.write("API key set successfully!")
else:
    st.stop()


# TODO: 1. time_it
# 2. $?
# 3. Q(speech->text)/A
# input（image，metadata，prompt，userquery(in voice/text) ）output voice+text response
print(
    f"current_dir:{os.getcwd()}"
)  # ~/Metaverse-Machine-Learning NOTE: remember to change your root directory to be able to find csv
source_data = "TTS/tags_replaced.csv"

# create a csv loader chain
# df = pd.read_csv(source_data, index_col=0, header=0, sep=",")
meta_data_extract_chain = create_csv_agent(
    ChatOpenAI(model="gpt-4-turbo", temperature=0),
    source_data,
    verbose=True,
    agent_type="openai-tools",
)


# meta_data_extract_chain = create_csv_agent(
#     OpenAI(model="gpt-4", temperature=0),
#     source_data,
#     verbose=True,
#     agent_type="openai-tools",
#     extra_tools={
#         "dataframe_query_tool": dataframe_query_tool,
#         "describe_data": describe_data,
#         "perform_calculation": perform_calculation,
#     },
# )

# create chat logic chain

tour_guide_template = """You are a AI tour guide who is friendly and has a great sense of humor while introducing art works at meseum. You can give response according to user input language, while respond user, you can also respond to them as if you are a native speaker.

Always feel free to ask interesting questions that keeps someone engaged.

You should also be a bit entertaining and not boring to talk to. Use informal language and be curious.

Your answer will always based on FACTS.

Please provide your answer AS DETAILED AS POSSIBLE, as this is a VERY IMPORTANT task for me.

Here is the FACTS you need to use:

========here is the relevant prompt to the question [START]=========
{data_raw}
========here is the relevant prompt to the question [END]=========

You don't need to tell the user the `url`, it would take too long to say and non-human readable.

However, if user asks for artwork's url, you should provide it so that I can extract the `url` from it to show the actual image to the user.

If there are multiple urls, you should provide all of them in a list.

=====================================the image url is after **Image:** section, in markdown format=====================
{data_content}
[THE URL1 OF THE ARTWORK if has, THE URL2 OF THE ARTWORK if has, THE URL3 OF THE ARTWORK if has, ...]
=====================================the image url is after **Image:** section, in markdown format=====================

If you don't understand question, just said you don't know and states the reason why you don't know.

-------------------------------

Human/User: {input}

NOTE: You should always state all detailed description surround the user's question, make sure to add some engagment as if you are a real tour guide! This concerns my entire career so please do a good job! NEVER MENTION YOU ARE AN AI MODEL DEVELOPED BY OpenAI!

DO NOT TELL USER WHAT TYPE OF DATA YOU ARE USING, JUST ANSWER THE QUESTION AS IF YOU ARE A REAL TOUR GUIDE!

Please answer the question in this format, ONLY OUTPUT in json format AND NOTHING ELSE, OTHERWISE IT WILL CAUSE ERROR:
the json keys are: `tour_guide_response` , `image_url` (if has, else url value is empty string; if has multiple, separate each url with semicolon) and `artwork_name` (if has, else artwork_name value is empty string; if has multiple, separate each name with semicolon).

AI Tour Guide:
"""

tour_guide_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", tour_guide_template),
        ("user", "{input}"),
    ]
)


##############################################################################
#                Final   Chain                                               #
##############################################################################

# max_token_limit=40 - token limits needs transformers installed
# memory = ConversationSummaryBufferMemory(
#     llm=ChatOpenAI(model="gpt-4-turbo", temperature=0),
#     memory_key="history",
#     return_messages=True,
# )
# memory_runnable = ConversationChain(
#     llm=ChatOpenAI(),
#     memory=memory,
#     prompt="""
#         According to this information: {history}. Only use latest conversation.
#
#         Please answer the question in this format, ONLY OUTPUT in json format AND NOTHING ELSE, OTHERWISE IT WILL CAUSE ERROR:
#                                         the json keys are: `tour_guide_response` , `image_url` (if has, else url value is empty string) and `artwork_name` (if has, else artwork_name value is empty string).
#
#         Question: question from user, from previous context, you need to find it
#         Answer:
#         """,
# )


# In this setup, save_to_memory ensures that memory
# is updated with both the user's input and the AI's response after each run.
def save_to_memory(chain, inputs, outputs):
    input_str = str(inputs["input"])  # Ensure input is a string
    output_str = str(outputs["response"])  # Ensure output is a string

    # Save context to memory
    chain.memory.save_context({"input": input_str}, {"response": output_str})


data_conversation_chatbot_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Remember, the AI tour guide should always display each the artwork image url in the description if it has one!

                Here is the chat history: {chat_history}

                ===========================================

                Here is the question from user: {input}
                """,
            )
        ]
    )
    | meta_data_extract_chain
    | RunnablePassthrough.assign(data_raw=lambda x: x["output"])
    | RunnablePassthrough.assign(data_content=lambda x: x["data_raw"])
    | tour_guide_prompt
    | ChatOpenAI(model="gpt-4-turbo", temperature=0)
    # | (lambda x: x["output"])
    # | memory_runnable
    | JsonOutputParser()
)

streamlit_chat_history = StreamlitChatMessageHistory(key="history")

data_conversation_chatbot_chain_history = RunnableWithMessageHistory(
    data_conversation_chatbot_chain,
    lambda session_id: streamlit_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    # memory=st.session_state.entity_memory,
    # memory=memory,
    # verbose=True,
)

# ================streamlit display================================
# TODO: database access
# send to 4o directly
# shorter response [3-4 sentences]
# voice and text streaming [real-time] low latency socket: flask-socketio: https://flask-socketio.readthedocs.io/en/latest/
# Openai batch api: https://platform.openai.com/docs/guides/batch/overview
# some project links: https://www.notion.so/seemuseums/dd751e1dc0ee40228d5675f1b28925e4?pvs=4


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


initial_message = "Hi, I am your museum AI assistant. I can help you with any questions you have on museum, any questions on museum?"


# if "content" not in st.session_state:
#     st.session_state.content = []

# for message in st.session_state.content:
#     with st.chat_message(message["role"]):
#         if message["content"]:
#             st.markdown(message["content"])

# if "image_url" not in st.session_state:
#     st.session_state["image_url"] = []

# if "enity_memory" not in st.session_state:
#     st.session_state.entity_memory.entity_store = {}
#     st.session_state.entity_memory.buffer.clear()

# Render current messages from StreamlitChatMessageHistory
# for msg in streamlit_chat_history.content:
#     if msg:
#         st.chat_message(msg.type).write(msg.content)
#

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

# Render current messages from StreamlitChatMessageHistory
# print("========================================")
# print("type of streamlit_chat_history: ", type(streamlit_chat_history))
# print("========================================")
# for msg in streamlit_chat_history.content:
#     print("msg: ", msg)
#     print(type(msg))
#     st.chat_message(msg.type).write(msg)


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

    input_text = client.audio.transcriptions.create(model="whisper-1", file=audio_file)

    # delete audio file after use
    os.remove("user_response.mp3")

    # only get str as input
    input_text = input_text.text


print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("input_text: ", input_text)
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

# Render current messages from StreamlitChatMessageHistory
for msg in streamlit_chat_history.messages:
    print("msg: ", msg)
    print("type of msg: ", type(msg))
    if (
        type(msg) == langchain_core.messages.ai.AIMessage
        or type(msg) == langchain_core.messages.human.HumanMessage
    ):
        st.chat_message(msg.type).write(msg.content)
        print("========================================")

with st.chat_message("user"):
    st.write(input_text)
    streamlit_chat_history.add_user_message(input_text)


with st.chat_message("AI"):
    # streaming response
    with st.spinner("Thinking..."):
        response = data_conversation_chatbot_chain_history.invoke(
            {"input": input_text, "question": input_text},
            {"configurable": {"session_id": "unused"}},
        )
        print("=====================================")
        print("response: ", response)
        print("=====================================")
        ai_response = response["tour_guide_response"]
        image_urls = response["image_url"].split(";")
        artwork_names = response["artwork_name"].split(";")

        if "history" not in st.session_state.keys():
            st.session_state["history"] = []
        if "image_url" not in st.session_state.keys():
            st.session_state["image_url"] = []

        # write chat history to streamlit
        streamlit_chat_history.add_ai_message(ai_response)
        st.session_state["history"].append(ai_response)
        print("\n=============================\n")
        print("ai_response:\n", ai_response)
        # streaming the output
        st.write_stream(stream_data_output(ai_response))

        for image_url, artwork_name in zip(image_urls, artwork_names):
            print("image_url:\n", image_url)
            st.session_state["image_url"].append(image_url)
            print("artwork_name:\n", artwork_name)

            if validators.url(image_url):
                st.image(image_url, use_column_width=True, caption=artwork_name)

        print("\n=============================\n")

        # generate audio response
        stream_to_speakers(ai_response)
        st.audio("response.mp3", format="audio/mp3", autoplay=True)

# if input_text:
#     st.session_state.content.append(
#         {
#             "role": "user",
#             "content": input_text,
#         }
#     )
#
# if ai_response:
#     st.session_state.content.append(
#         {
#             "role": "AI",
#             "content": ai_response,
#             "image_url": image_url if image_url else None,
#         }
#     )

# display chat history
# NOTE: I don't think gpt will output exact same output everytime, even with temperature=0
# st.session_state.content = list(
#     filter(lambda x: x["content"] is not None, st.session_state.content)
# )

# display chat history
# st.write(st.session_state["history"])
