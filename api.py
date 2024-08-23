import streamlit as st
import os
import json
from langchain_experimental.agents.agent_toolkits import (
    create_csv_agent,
    # create_pandas_dataframe_agent,
)
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.string import StrOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langsmith import Client
from langchain.callbacks import LangChainTracer

from dotenv import load_dotenv

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


key = st.text_input("Enter your OpenAI API key", type="password")
if key:
    os.environ["OPENAI_API_KEY"] = key
    st.write("API key set successfully!")
else:
    st.stop()

# from dotenv import load_dotenv

# from langchain.agents.agent_types import AgentType
# from langchain.agents import Agent

# from langchain.tools import BaseTool, tool, StructuredTool


# from langchain_core.prompts.image import ImagePromptTemplate

# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain.callbacks.base import BaseCallbackHandler
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain.chains import ConversationChain
# from langserve import add_routes
# import uvicorn
# from threading import Thread
# from queue import Queue
# import pandas as pd
# from fastapi import FastAPI
# import os
# from langchain.tools import Tool

# load_dotenv()


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

========here is the history of the conversation========
{chat_history}
========here is the history of the conversation========

You don't need to tell the user the `url`, it would take too long to say and non-human readable.

However, if user asks for artwork's url, you should provide it so that I can extract the `url` from it to show the actual image to the user.

If there are multiple urls, you should provide all of them in a list.

=====================================the image url is after **Image:** section, in markdown format=====================
{data_content}
[THE URL1 OF THE ARTWORK if has, THE URL2 OF THE ARTWORK if has, THE URL3 OF THE ARTWORK if has, ...]
=====================================the image url is after **Image:** section, in markdown format=====================

If you don't understand question, just said you don't know and states the reason why you don't know.

-------------------------------

Here is the Human question:

~~~~~~~~~~~~~~
{input}
~~~~~~~~~~~~~~

Please answer the question in this format, ONLY OUTPUT in json format AND NOTHING ELSE, OTHERWISE IT WILL CAUSE ERROR:
the json keys are: `tour_guide_response` and `image_url` (if has, else url value is empty string)

NOTE: You should always state all detailed description surround the user's question, make sure to add some engagment as if you are a real tour guide! This concerns my entire career so please do a good job! NEVER MENTION YOU ARE AN AI MODEL DEVELOPED BY OpenAI!
"""

tour_guide_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", tour_guide_template),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
    ]
)


##############################################################################
#                Final   Chain                                               #
##############################################################################

data_conversation_chatbot_chain = (
    RunnablePassthrough.assign(data_raw=meta_data_extract_chain)
    | RunnablePassthrough.assign(data_content=lambda x: x["data_raw"]["output"])
    | tour_guide_prompt
    | ChatOpenAI(temperature=1, model="gpt-4-turbo")
    | StrOutputParser()
    | (lambda x: json.loads(x.replace("\\n", "").replace("\\", "").replace('"', '"')))
)
