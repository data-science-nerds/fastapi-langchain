# live_LLM.py
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


# [Place all the required imports here]
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


# def ciphers_load(index_name):
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
print("Completed ciphers load, GRPCIndex")
    # return None

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)
print("openai embeddings completed")


text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)
print("vector store initialized")


llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )


# def input_query(query):
#     res = vectorstore.similarity_search(
#         query,  # our search query
#         k=10  # return 3 most relevant docs
#     )
#     print(f"COMPLETED input_query")
#     return res

# def similarity_search(query: str, k=3):
#     # print(f"Your query:\n{query}")
#     # [Place the vectorstore.similarity_search implementation here]
#     results =  vectorstore.similarity_search(
#         query,  # our search query
#         k=10  # return 3 most relevant docs
#     )
#     # print(f"Your query:\n{query}")
#     results = vectorstore.similarity_search(query, k=10)
#     # print(f"Similarity search results:\n{results}")
#     print("COMPLETED similarity_search")
#     return results


# def final_answer_with_sources(query: str):
    # completion llm
    # llm = ChatOpenAI(
    #     openai_api_key=OPENAI_API_KEY,
    #     model_name='gpt-3.5-turbo',
    #     temperature=0.0
    # )

#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectorstore.as_retriever()
#     )

#     response = qa.run(query)
#     print(f"COMPLETED final_answer_with_sources")
#     return response #qa
    
# def qa_with_sources(query):
#     res = RetrievalQAWithSourcesChain.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectorstore.as_retriever()
#     )
#     print("COMPLETED qa_with_sources")
#     # print(f"{res}")
#     return res


if __name__ == "__main__":
    # index_name = 'langchain-retrieval-augmentation-fast'
    # indexname = index_name
    # connect_to_pinecone(index_name)
    # query = "How did Benito Mussolini affect Italy?"
    query = "Who has ruled Italy?"
    k = 3
    # results = similarity_search(query, k)
    # # qa = final_answer_with_sources(query)
    # # response = qa.run(query)
    # # print(f"QA run response:\n{response}")

    # qa_with_sources_response = RetrievalQAWithSourcesChain.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=vectorstore.as_retriever()
    # )
    # print(f"QA with sources response:\n{qa_with_sources_response}")
    vectorstore.similarity_search(
        query,  # our search query
        k=k  # return 3 most relevant docs
        )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    print(qa.run(query))

    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    print(qa_with_sources(query))