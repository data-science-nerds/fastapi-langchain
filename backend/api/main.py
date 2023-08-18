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


index_name = 'langchain-retrieval-augmentation-fast'
indexname = index_name

# get openai api key from platform.openai.com
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'

model_name = 'text-embedding-ada-002'

pinecone.init(
api_key=os.getenv('PINECONE_API_KEY'),  
environment=os.getenv('PINECONE_ENV') 
)

if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536,  # 1536 dim of text-embedding-ada-002
    )
print(f"Connected to pinecone index {index_name}")

# Initialize Pinecone
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, metric='cosine', dimension=1536)
print(f"Connected to pinecone index {index_name}")

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


# Initialize GRPCIndex and wait for it
index = pinecone.GRPCIndex(index_name)
# wait a moment for the index to be fully initialized
time.sleep(20)

index_stats = index.describe_index_stats()
print(f"Completed ciphers load, GRPCIndex index_stats: \n {index_stats}")
    # return None

# Initialize embeddings
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)
print("openai embeddings completed")

# Initialize vector store
print("Initializing vector store")
text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)
print("vector store initialized")

# Initialize chat model
llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )
print(f"Initialized openai chat model LLM model name {model_name}, temperature 0.0," )

    
# Initialize QA handlers
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/ask/")
async def ask_question(query: str):
    try:
        answer = qa.run(query)
        answer_with_sources = qa_with_sources(query)
        return {
            "answer": answer,
            "answer_with_sources": answer_with_sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/")
def search(query: str):
    # This function may be redundant since we have the /ask/ route
    result = f"You searched for: {query}"
    return {"result": result}
