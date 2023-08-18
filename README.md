# fastapi-langchain
Showcase pydantic, fastapi, langchain, openai

# we use pydantic to validate:
- all rows going into pinecone have data
- all pinecone data cites a valid https link


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
