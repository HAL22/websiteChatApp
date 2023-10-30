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

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# will store all of the webpages visted in text format
webpages = []

# Count the webpages
pageCount = 0

dict_href_links = {}

p = {}

# Max webpages
maxWebpages = 500

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def getdata(url):
    # add header to prevent being blocked (403 error) by wordpress websites
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    r = requests.get(url, headers=headers)
    return r.text

def get_links(website_link, website,html_data):
    soup = BeautifulSoup(html_data, "lxml")
    list_links = []
    for link in soup.find_all("a", href=True):
        # Append to list if new link contains original link
        if str(link["href"]).startswith((str(website))):
            list_links.append(link["href"])

        # Include all href that do not start with website link but with "/"
        if str(link["href"]).startswith("/"):
            if link["href"] not in dict_href_links:
                print(link["href"])
                dict_href_links[link["href"]] = None
                link_with_www = website + link["href"][1:]
                print("adjusted link =", link_with_www)
                list_links.append(link_with_www)

    # Convert list of links to dictionary and define keys as the links and the values as "Not-checked"
    dict_links = dict.fromkeys(list_links, "Not-checked")
    return dict_links

def get_subpage_links(l, website):
    global pageCount
    for link in tqdm(l):
        if pageCount>maxWebpages:
          break
        # If not crawled through this page start crawling and get links
        elif l[link] == "Not-checked":
            html_data = html_data = getdata(link)
            dict_links_subpages = get_links(link, website,html_data)
            # Change the dictionary value of the link to "Checked"
            l[link] = "Checked"
            webpages.append(BeautifulSoup(html_data, "lxml").text)
            # global pageCount
            pageCount+=1
        else:
            # Create an empty dictionary in case every link is checked
            dict_links_subpages = {}
        # Add new dictionary to old dictionary
        l = {**dict_links_subpages, **l}
    return l

def get_pages(link):
    # add websuite WITH slash on end
    website = link
    # create dictionary of website
    dict_links = {website:"Not-checked"}

    counter = None
    while counter != 0 and pageCount<maxWebpages:
        dict_links2 = get_subpage_links(dict_links, website)
        # Count number of non-values and set counter to 0 if there are no values within the dictionary equal to the string "Not-checked"
        # https://stackoverflow.com/questions/48371856/count-the-number-of-occurrences-of-a-certain-value-in-a-dictionary-in-python
        counter = sum(value == "Not-checked" for value in dict_links2.values())

        dict_links = dict_links2
        # Save list in json file
        global p
        p = {**dict_links, **p}

    return webpages

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def split_text(text):

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,  # number of tokens overlap between chunks
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
    )

    return text_splitter.split_text(text)

def create_document_from_webpage(texts):

    docs = [Document(page_content=t) for t in texts]

    return docs

def get_texts(url):

    webpages = get_pages(url)
    texts = []
    for web in webpages:
        split = split_text(web)
        texts = texts + split

    return create_document_from_webpage(texts)    

def load_pinecone(url,index_name, embeddings=OpenAIEmbeddings(model="text-embedding-ada-002")):
    # initialize pinecone
    pinecone.init(
    api_key=st.secrets['PINECONE_API_KEY'],
    environment=st.secrets['PINECONE_ENV']
    )   
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            dimension=1536  
        )
    else:
        index =  Pinecone.from_existing_index(index_name,embeddings) 

        return ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), index.as_retriever(), memory=memory)    

    pages = get_texts(url)

    index = Pinecone.from_documents(pages, embeddings, index_name=index_name)
    return ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), index.as_retriever(), memory=memory)


def creat_embeddings(url):
    pinecone_name = "anthropic"
    
    return load_pinecone(url,pinecone_name)

def get_agent(url):
    qa = creat_embeddings(url)

    tools = [
        Tool(
            name='Knowledge Base',
            func=qa.run,
            description=(
                'use this tool when answering general knowledge queries to get '
                'more information about the topic'
            )
        )
    ]

    llm = ChatOpenAI(
        openai_api_key=os.environ['OPENAI_API_KEY'],
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=memory
    )

    return agent
