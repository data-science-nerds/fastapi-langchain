# %% [markdown]
# #### Code is borrowed from LangChain Handbook, adapted to fit in the presently working schema
# August 17, 2023
# 
# Pydantic validation used to appease LLM Gods with doing due dilligence by way of data governance and sufficient [chicken sacrifices](https://www.linkedin.com/feed/update/urn:li:activity:7092904219103432704?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A7092904219103432704%29)
# 
# #### [LangChain Handbook](https://pinecone.io/learn/langchain)
# 
# # Retrieval Augmentation
# # Generative Question Answering
# ### (we compile some data from our knowledge base and the LLM pieces it together to give the best answer possible)
# 
# **L**arge **L**anguage **M**odels (LLMs) have a data freshness problem. The most powerful LLMs in the world, like GPT-4, have no idea about recent world events.
# 
# The world of LLMs is frozen in time. Their world exists as a static snapshot of the world as it was within their training data.
# 
# A solution to this problem is *retrieval augmentation*. The idea behind this is that we retrieve relevant information from an external knowledge base and give that information to our LLM. In this notebook we will learn how to do that.
# 
# Here we showcase only the part of getting a sensical answer from our LLM.  We have already uploaded our vectors in the other notebook.
# 
# 
# <!--Nothing actually in these notebooks links, it's the same notebook [![Open full notebook](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/full-link.svg)](https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/langchain-retrieval-augmentation.ipynb) -->
# 
# To begin, we must install the prerequisite libraries that we will be using in this notebook.

# %% [markdown]
# 

# %%
!pip install -qU \
  langchain==0.0.162 \
  openai==0.27.7 \
  tiktoken==0.4.0 \
  "pinecone-client[grpc]"==2.2.1 \
  pinecone_datasets=='0.5.0rc10'

# %% [markdown]
# ---
# 
# 🚨 _Note: the above `pip install` is formatted for Jupyter notebooks. If running elsewhere you may need to drop the `!`._
# 
# ---

# %% [markdown]
# This script is for interacting with our Pinecone vector database.
# 
# ## Vector Database
# 
# To create our vector database we first need a [free API key from Pinecone](https://app.pinecone.io). Then we initialize like so:

# %%
index_name = 'langchain-retrieval-augmentation-fast'
indexname = index_name

# %%
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
import os
import pinecone

# connect to pinecone environment
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

# %% [markdown]
# Then we connect to the new index:

# %%
import time
# allows notebook to work
import requests
from requests.packages.urllib3.util.ssl_ import create_urllib3_context

CIPHERS = (
    'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:ECDH+AESGCM:ECDH+CHACHA20:DH+AESGCM:DH+CHACHA20:'
    'ECDHE+AES:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4:!HMAC_SHA1:!SHA1:!DHE+AES:!ECDH+AES:!DH+AES'
)

requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = CIPHERS
# Skip the following two lines if they cause errors
# requests.packages.urllib3.contrib.pyopenssl.DEFAULT_SSL_CIPHER_LIST = CIPHERS
# requests.packages.urllib3.contrib.pyopenssl.inject_into_urllib3()
requests.packages.urllib3.util.ssl_.create_default_context = create_urllib3_context

index = pinecone.GRPCIndex(index_name)
# wait a moment for the index to be fully initialized
time.sleep(20)

index.describe_index_stats()

# %% [markdown]
# ## Creating a Vector Store and Querying
# 
# Iinitialize a LangChain vector store using the same index built. For this we will also need a LangChain embedding object, which we initialize like so:

# %%
from langchain.embeddings.openai import OpenAIEmbeddings

# get openai api key from platform.openai.com
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

# %% [markdown]
# Now initialize the vector store:

# %%
from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# %% [markdown]
# Now we can query the vector store directly using `vectorstore.similarity_search`:

# %%
# query = "who was Benito Mussolini?"
query = "How did Benito Mussolini affect Italy?"
query = "How has ruled Italy?"
vectorstore.similarity_search(
    query,  # our search query
    k=10  # return 3 most relevant docs
)

# %% [markdown]
# All of these are good, relevant results. But what can we do with this? There are many tasks, one of the most interesting (and well supported by LangChain) is called _"Generative Question-Answering"_ or GQA.
# 
# ## Generative Question-Answering
# 
# In GQA we take the query as a question that is to be answered by a LLM, but the LLM must answer the question based on the information it is seeing being returned from the `vectorstore`.
# 
# To do this we initialize a `RetrievalQA` object like so:

# %%
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# %%
qa.run(query)

# %% [markdown]
# We can also include the sources of information that the LLM is using to answer our question. We can do this using a slightly different version of `RetrievalQA` called `RetrievalQAWithSourcesChain`:

# %%
from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# %%
qa_with_sources(query)

# %% [markdown]
# ---

# %% [markdown]
# 


