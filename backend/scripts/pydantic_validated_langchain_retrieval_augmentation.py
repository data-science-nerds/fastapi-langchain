# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/langchain-retrieval-augmentation.ipynb) [![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/pinecone-io/examples/blob/master/docs/langchain-retrieval-augmentation.ipynb)

# %% [markdown]
# #### Code is borrowed from LangChain Handbook, adapted to fit in the presently working schema
# August 17, 2023
# 
# Adding pydantic validation to appease LLM Gods with doing due dilligence by way of data governance and sufficient [chicken sacrifices](https://www.linkedin.com/feed/update/urn:li:activity:7092904219103432704?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A7092904219103432704%29)
# 
# #### [LangChain Handbook](https://pinecone.io/learn/langchain)
# 
# # Retrieval Augmentation
# 
# **L**arge **L**anguage **M**odels (LLMs) have a data freshness problem. The most powerful LLMs in the world, like GPT-4, have no idea about recent world events.
# 
# The world of LLMs is frozen in time. Their world exists as a static snapshot of the world as it was within their training data.
# 
# A solution to this problem is *retrieval augmentation*. The idea behind this is that we retrieve relevant information from an external knowledge base and give that information to our LLM. In this notebook we will learn how to do that.
# 
# 
# <!--Nothing actually in these notebooks links, it's the same notebook [![Open full notebook](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/full-link.svg)](https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/langchain-retrieval-augmentation.ipynb) -->
# 
# To begin, we must install the prerequisite libraries that we will be using in this notebook.

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
# ## Building the Knowledge Base
# 
# We will download a pre-embedding dataset from `pinecone-datasets`. Allowing us to skip the embedding and preprocessing steps, if you'd rather work through those steps you can find the [full notebook here](https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/langchain-retrieval-augmentation.ipynb).

# %%
import pinecone_datasets

dataset = pinecone_datasets.load_dataset('wikipedia-simple-text-embedding-ada-002-100K')
dataset.head()

# %%
print(type(dataset))
print(len(dataset))

# %% [markdown]
# We'll format the dataset ready for upsert and reduce what we use to a subset of the full dataset.

# %%
#### THE API REQUIRES THE METADATA FIELD TO BE POPULATED
# 1. Convert the Dataset object to a DataFrame to be able to 
#  a. test with pydantic
#   1. ensure we have blob and metadata columns (to appease pinecone today)
#   2. ensure there is a valid https
df = dataset.documents

# 2. Modify the DataFrame
# Assuming you want to copy the 'metadata' column to 'blob' (and create 'blob' if it doesn't exist)
if 'blob' in df.columns:
    df['metadata'] = df['blob']
df.head()

# %%
# Conduct data and datatype exploration
# Check the type of the first few entries
types_of_values = df['values'].apply(type).value_counts()

# Check if there are any non-list type entries
non_list_values = df[df['values'].apply(lambda x: not isinstance(x, list))]['values']

types_of_values, non_list_values.head()


# %%
# pydantic will ensure only rows with data (a valid 'blob') make it to pinecone
# for all rows with data, they must cite a valid https source
from pydantic import BaseModel, HttpUrl, validator, ValidationError
from typing import Optional, Union, Dict
import numpy as np

class Blob(BaseModel):
    chunk: int
    source: HttpUrl
    
    @validator('source', pre=True, always=True)
    def check_https_scheme(cls, v):
        '''Cybersecurity check-- ensure only secure websites are referenced'''
        if not v.startswith("https://"):
            raise ValueError("source URL must be HTTPS")
        return v

    @validator('chunk', pre=True, always=True)
    def check_chunk(cls, v):
        if v is None:
            raise ValueError("chunk field is missing")
        return v


class DatasetEntry(BaseModel):
    id: str
    values: np.ndarray
    sparse_values: Optional[Union[str, None]]
    metadata: Optional[Union[Dict, None]]
    blob: Blob

    @validator('blob', pre=True, always=True)
    def check_blob(cls, v):
        if v is None:
            raise ValueError("blob field is missing")
        return v

    @validator('values', pre=True, always=True)
    def check_values_type(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValidationError("values must be a numpy array")
        return v
    
    class Config:
        arbitrary_types_allowed = True
        
for _, row in df.iterrows():
    entry = DatasetEntry(**row.to_dict())



# %%
import os
#### THE PINECONE OBJECT REQUIRES THE PARQUET FILE TO BE IN DIRECOTRY '*/DOCUMENTS'
# # Create the documents directory if it doesn't exist
# switch the df object back to pinecone object
documents_dir = "../data/processed/documents"
os.makedirs(documents_dir, exist_ok=True)

# Save the DataFrame as a Parquet file inside the documents directory
parquet_file_inside_documents = os.path.join(documents_dir, "parquet_df_with_metadata.parquet")
df.to_parquet(parquet_file_inside_documents)


# %%
#### MAKE SURE META DATA IS POPULATED BY USING WHAT WAS IN BLOB
### THIS NEEDS TO BE A PINECONE OBJECT
new_dataset = pinecone_datasets.dataset.Dataset.from_path("../data/processed/")
print(new_dataset.head())
type(new_dataset)

# %% [markdown]
# Now we move on to initializing our Pinecone vector database.
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

# # find API key in console at app.pinecone.io
# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY') or 'PINECONE_API_KEY'
# # find ENV (cloud region) next to API key in console
# PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT') or 'PINECONE_ENVIRONMENT'

# pinecone.init(
#     api_key=PINECONE_API_KEY,
#     environment=PINECONE_ENVIRONMENT
# )
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
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
import os
import pinecone
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
# We should see that the new Pinecone index has a `total_vector_count` of `0`, as we haven't added any vectors yet.
# 
# Now we upsert the data to Pinecone:

# %%
for batch in new_dataset.iter_documents(batch_size=100):
    index.upsert(batch)

# %% [markdown]
# We've now indexed everything. We can check the number of vectors in our index like so:

# %%
index.describe_index_stats()

# %% [markdown]
# ## Creating a Vector Store and Querying
# 
# Now that we've build our index we can switch over to LangChain. We need to initialize a LangChain vector store using the same index we just built. For this we will also need a LangChain embedding object, which we initialize like so:

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
query = "who was Benito Mussolini?"

vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
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
# Now we answer the question being asked, *and* return the source of this information being used by the LLM.
# 
# Once done, we can delete the index to save resources.

# %%
# pinecone.delete_index(index_name)

# %% [markdown]
# ---


