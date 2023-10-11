import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import tiktoken
import os
import getpass
import pinecone
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
import streamlit as st

def getChatbot(maxWebpages):
    # will store all of the webpages visted in text format
    webpages = []

    # Count the webpages
    pageCount = 0

    dict_href_links = {}

    p = {}

