from fastapi import FastAPI , HTTPException, Depends, status ,Request
from pydantic import BaseModel
from app.api.endpoints import milvus,mlflow
from app.connector.connectorMilvus import MilvusManager
from app.connector.connectorBucket import MinioBucketManager
from fastapi.responses import PlainTextResponse

app = FastAPI()
MilvusManager().__init__()
app.include_router(milvus.router, prefix="/milvus", tags=["milvus"])
app.include_router(mlflow.router, prefix="/mlflow", tags=["mlflow"])
@app.get("/")
async def root():
    return {"message": "api up"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)