from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
import os
import pinecone
from langchain.vectorstores import Pinecone


import time
import requests
from requests.packages.urllib3.util.ssl_ import create_urllib3_context
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.chains import RetrievalQAWithSourcesChain

import traceback

import logging

logging.basicConfig(level=logging.DEBUG)


logger = logging.getLogger(__name__)


index_name = 'langchain-retrieval-augmentation-fast'
indexname = index_name

# get openai api key from platform.openai.com
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'

model_name = 'text-embedding-ada-002'

# Initialize Pinecone
pinecone.init(
api_key=os.getenv('PINECONE_API_KEY'),  
environment=os.getenv('PINECONE_ENV') 
)
print(f"Initialized pinecone init {index_name}")
logger.debug(f"Initialized pinecone init {index_name}")

if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536,  # 1536 dim of text-embedding-ada-002
    )
print(f"Connected to pinecone index {index_name}")
logger.debug(f"Connected to pinecone index {index_name}")

# Initialize Pinecone
# pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))
# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(name=index_name, metric='cosine', dimension=1536)
# print(f"Initialized pinecone init {index_name}")
# logger.debug(f"Initialized pinecone init {index_name}")

# Ciphers setup
CIPHERS = (
'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:ECDH+AESGCM:ECDH+CHACHA20:DH+AESGCM:DH+CHACHA20:'
'ECDHE+AES:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4:!HMAC_SHA1:!SHA1:!DHE+AES:!ECDH+AES:!DH+AES'
)

requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = CIPHERS
# Skip the following two lines if they cause errors
# requests.packages.urllib3.contrib.pyopenssl.DEFAULT_SSL_CIPHER_LIST = CIPHERS
# requests.packages.urllib3.contrib.pyopenssl.inject_into_urllib3()
requests.packages.urllib3.util.ssl_.create_default_context = create_urllib3_context


# Initialize GRPCIndex
index = pinecone.GRPCIndex(index_name)

# Define the maximum number of retries and the delay between them
max_retries = 10
delay_seconds = 10

for _ in range(max_retries):
    try:
        # Try to get the index stats
        index_stats = index.describe_index_stats()
        print(f"Completed ciphers load, GRPCIndex index_stats: \n {index_stats}")
        logger.debug(f"Completed ciphers load, GRPCIndex index_stats: \n {index_stats}")
        break  # If successful, break out of the loop
    except Exception as e:
        logger.warning(f"Failed to initialize GRPCIndex. Retrying. Error: {e}")
        time.sleep(delay_seconds)  # If there's an error, wait for a short time and try again
else:
    # This block will run if the loop completes without a break (i.e., if the initialization fails even after max_retries)
    raise Exception("Failed to initialize GRPCIndex after multiple retries.")


# Initialize embeddings
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)
print("openai embeddings completed")
logger.debug("openai embeddings completed")

# Initialize vector store
print("Initializing vector store")
logger.debug("Initializing vector store")
             
text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)
print("vector store initialized")
logger.debug("vector store initialized")

# Initialize chat model
llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )
print(f"Initialized openai chat model LLM model name {model_name}, temperature 0.0" )
logger.debug(f"Initialized openai chat model LLM model name {model_name}, temperature 0.0" )
    
# Initialize QA handlers
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.exception_handler(Exception)
async def custom_exception_handler(request, exc):
    # Print the traceback to the console
    traceback.print_exc()
    
    # Log the traceback
    logger.error(traceback.format_exc())
    
    return {"detail": str(exc)}


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/ask/")
async def ask_question(query: str):
    logger.debug(f"BEGIN ask_question in @app.get/ask/: Starting processing for the query: {query}")
    try:
        answer = qa.run(query)
        answer_with_sources = qa_with_sources(query)
        logger.debug("END ask_question")
        return {
            "answer without sources": answer,
            "answer_with_sources": answer_with_sources
        }
    except Exception as e:
        logger.debug("END ask_question with http exception")
        raise HTTPException(status_code=500, detail=str(e))
    

# @app.get("/search/")
# def search(query: str):
#     # This function may be redundant since we have the /ask/ route
#     logger.debug("BEGIN search, which may be redundant in @app.get/search/: Starting processing for the query")

#     result = f"You searched for: {query}"
#     logger.debug(f"END search for {query}")
#     return {"result": result}

