# fastapi-langchain
Showcase pydantic, fastapi, langchain, openai

live web app running on:
[http://54.198.111.198:8080/static/index.html](http://54.198.111.198:8080/static/index.html)
# we use pydantic to validate:
- all rows going into pinecone have data
- all pinecone data cites a valid https link
- the validation is done before the upsert
- the data used here was validated in
  scripts/pydantic_validation_langchain_retrieval_augmentation.ipynb
  * they were only done once for upserting because once it is validated as a dataset before going into pinecone, we dont need to validate before upserting it again
  * we would validate again if we were refreshing our pinecone embeddings with new data

to run inside ec2 for demo purposes (if production would use nginx):
uvicorn backend.api.main:app --host 0.0.0.0 --port 8080

  
- to run locally, use command:
  ```uvicorn backend.api.main:app --reload```
  and check page
  [http://127.0.0.1:8000/static/index.html](http://127.0.0.1:8000/static/index.html)

  to restart the local instance:
  ```lsof -i :8000```
  ```kill -9 <pid>```



### file structure
fastapi-langchain/   


│  

├── backend/  

│   ├── data/  

│   │   └── processed/  

│   │       └── documents/  

│   │           └── parquet_df_with_metadata.parquet  

│   │  

│   ├── scripts/  

│   │   ├── live_LLM.py  

│   │   ├── ... [other scripts]  

│   │  

│   ├── api/  

│   │   ├── main.py  # FastAPI main application file  

│   │   ├── models/  # Pydantic models, if any  

│   │   └── routers/  # Separate FastAPI routers, if you want to modularize endpoints  

│   │  

│   └── .env  # Contains environment variables  

│  

├── frontend/  

│  

└── requirements.txt  
