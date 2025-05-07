from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility, connections
import pandas as pd 
import os 
from pymilvus import MilvusClient
from pymilvus import utility


class MilvusManager:
    def __init__(self):
        
        self.connection =connections.connect(
            alias="default",
            host=os.getenv("MILVUS_ENDPOINT"),
            port=os.getenv("MILVUS_PORT"),
            token=f"{os.getenv('MILVUS_USERNAME')}:{os.getenv('MILVUS_PASSWORD')}"
        )
        self.client = MilvusClient(
            uri=f"http://{os.getenv('MILVUS_ENDPOINT')}:{os.getenv('MILVUS_PORT')}",
            token=f"{os.getenv('MILVUS_USERNAME')}:{os.getenv('MILVUS_PASSWORD')}"
        )

    def status_collection(self):
        data=[]
        for element in self.client.list_collections():
            data.append([element,self.client.get_load_state(collection_name=element)])
        return data

    def recherche_vectorielle(self,collection,index,n_sortie,vecteur):
        collection = Collection(collection)
        search_params = {
            "metric_type": "L2",
            # "metric_type": "COSINE",
            "params": {"nprobe": 20},
        }
        
        result = collection.search(vecteur, index, search_params, limit=n_sortie, output_fields=['smiles', 'filename', 'Egc', 'Egb', 'Eib', 'CED', 'Ei', 'Eea', 'nc', 'ne', 'epse_6_0', 'epsc', 'epse_3_0', 'epse_1_78', 'epse_15_0', 'epse_4_0', 'epse_5_0', 'epse_2_0', 'epse_9_0', 'epse_7_0', 'TSb', 'TSy', 'epsb', 'YM', 'permCH4', 'permCO2', 'permH2', 'permO2', 'permN2', 'permHe', 'Eat', 'rho', 'LOI', 'Xc', 'Xe', 'Cp', 'Td', 'Tg', 'Tm','embedding_bert','embedding'])
        return result
    def recherche_vectorielle_id(self,collection,index,n_sortie,vecteur):
        collection = Collection(collection)
        search_params = {
            "metric_type": "L2",
            # "metric_type": "COSINE",
            "params": {"nprobe": 20},
        }
        
        result = collection.search(vecteur, index, search_params, limit=n_sortie, output_fields=['id'])
        return result

    def all_id(self,collection):
        res = self.client.query(
            collection_name=collection,
            output_fields=["id"],
            filter="id >= 0"
        )   
        liste_id=[]
        for element in res:
            liste_id.append(element['id'])
        return liste_id
    def extraction_par_id(self,collection,list_id):
        res = self.client.get(
        collection_name=collection,
        ids=list_id,
        output_fields=['smiles', 'filename', 'Egc', 'Egb', 'Eib', 'CED', 'Ei', 'Eea', 'nc', 'ne', 'epse_6_0', 'epsc', 'epse_3_0', 'epse_1_78', 'epse_15_0', 'epse_4_0', 'epse_5_0', 'epse_2_0', 'epse_9_0', 'epse_7_0', 'TSb', 'TSy', 'epsb', 'YM', 'permCH4', 'permCO2', 'permH2', 'permO2', 'permN2', 'permHe', 'Eat', 'rho', 'LOI', 'Xc', 'Xe', 'Cp', 'Td', 'Tg', 'Tm','embedding_bert','embedding']
        )
        return res
