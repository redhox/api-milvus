from fastapi import APIRouter, Depends, HTTPException,status,UploadFile, File, Form, Body
from bson import ObjectId
from app.connector.connectorMilvus import MilvusManager
from app.connector.connectorBucket import MinioBucketManager
from pydantic import BaseModel
import pandas as pd 
import os 
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprintAsBitVect
import json
import pyarrow as pa
import random
import uuid
router = APIRouter()

class ModelIendif(BaseModel):
    code:str


@router.put("/identification") 
async def identification(data:ModelIendif):
    print('identification') 
    if data.code==os.getenv("CODE"):

        data={
            'message':{
                'mlflow_url':os.getenv("MLFLOW_URL"),
                'mlflow_username':os.getenv("MLFLOW_USER"),
                'mlflow_password':os.getenv("MLFLOW_PASSWORD"),
                'bucket_id':os.getenv("AWS_ACCESS_KEY_ID"),
                'bucket_secretid':os.getenv("AWS_SECRET_ACCESS_KEY"),
                'bucket_url':os.getenv("ENDPOINT_URL"),
                'bucket_region':os.getenv("REGION_NAME"),
                'bucket_signature':os.getenv("SIGNATURE_VERSION"),
                'bucket_name':os.getenv("BUCKET_NAME")
                }
        }
    else:
        data={'message:code faux'}
          
    print(data)
    return data