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
import json
from pathlib import Path
from typing import Union
from fastapi.encoders import jsonable_encoder
from app.api.smiles2embeddings import ModelChemBERTa
router = APIRouter()
ModelChemBERTa()
def sauvegarder_donnees(chemin_fichier='suivie_datasets.json'):
    """Sauvegarde le dictionnaire SUIVIE_DATASETS dans un fichier JSON"""
    with open(chemin_fichier, 'w', encoding='utf-8') as f:
        json.dump(SUIVIE_DATASETS, f, indent=4)

def charger_donnees(chemin_fichier='suivie_datasets.json'):
    """Charge les données depuis le fichier JSON dans SUIVIE_DATASETS"""
    global SUIVIE_DATASETS
    
    try:
        with open(chemin_fichier, 'r', encoding='utf-8') as f:
            SUIVIE_DATASETS = json.load(f)
    except FileNotFoundError:
        SUIVIE_DATASETS = {}
charger_donnees()

def listresult(result):
    liste=[]
    print('nombrede reponce',len(result[0]))
    for i in range(len(result[0])):
        data={'smiles': result[0][i].smiles,
            'filename': result[0][i].filename,
            'Egc': result[0][i].Egc,
            'Egb': result[0][i].Egb,
            'Eib': result[0][i].Eib,
            'CED': result[0][i].CED,
            'Ei': result[0][i].Ei,
            'Eea': result[0][i].Eea,
            'nc': result[0][i].nc,
            'ne': result[0][i].ne,
            'epse_6_0': result[0][i].epse_6_0,
            'epsc': result[0][i].epsc,
            'epse_3_0': result[0][i].epse_3_0,
            'epse_1_78': result[0][i].epse_1_78,
            'epse_15_0': result[0][i].epse_15_0,
            'epse_4_0': result[0][i].epse_4_0,
            'epse_5_0': result[0][i].epse_5_0,
            'epse_2_0': result[0][i].epse_2_0,
            'epse_9_0': result[0][i].epse_9_0,
            'epse_7_0': result[0][i].epse_7_0,
            'TSb': result[0][i].TSb,
            'TSy': result[0][i].TSy,
            'epsb': result[0][i].epsb,
            'YM': result[0][i].YM,
            'permCH4': result[0][i].permCH4,
            'permCO2': result[0][i].permCO2,
            'permH2': result[0][i].permH2,
            'permO2': result[0][i].permO2,
            'permN2': result[0][i].permN2,
            'permHe': result[0][i].permHe,
            'Eat': result[0][i].Eat,
            'rho': result[0][i].rho,
            'LOI': result[0][i].LOI,
            'Xc': result[0][i].Xc,
            'Xe': result[0][i].Xe,
            'Cp': result[0][i].Cp,
            'Td': result[0][i].Td,
            'Tg': result[0][i].Tg,
            'Tm': result[0][i].Tm,
            'fingerprint': result[0][i].fingerprint
            }
        if hasattr(result[0][i], 'distance'):
                data['distance'] = result[0][i].distance
        liste.append(data)
    return liste
def smiles2fingerprint(smiles):
    """ Compute multiple fingerprint types for a list of SMILES """
    print(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        fingerprint_vector=None
        pass
    # Compute fingerprints
    avalon_fp = np.array(pyAvalonTools.GetAvalonFP(mol, nBits=512), dtype=np.float32)
    morgan_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048), dtype=np.float32)
    atom_pair_fp = np.array(AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048), dtype=np.float32)
    topological_torsion_fp = np.array(GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048), dtype=np.float32)
    maccs_fp = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    rdkit_fp = np.array(AllChem.RDKFingerprint(mol, fpSize=2048), dtype=np.float32)
    estate_fp = np.array(AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=512), dtype=np.float32)

    # Combine all fingerprints into a single vector
    fingerprint_vector = np.concatenate([avalon_fp, morgan_fp, atom_pair_fp,topological_torsion_fp, maccs_fp, rdkit_fp, estate_fp])
    return fingerprint_vector
def upload_de_datastes(list_id,uuid_recherche):
    list_decoupe = [list_id[i:i+100] for i in range(0, len(list_id), 100)]

    i=0
    for liste in list_decoupe:
        result=MilvusManager().extraction_par_id(liste)
        df=pd.DataFrame(result)
        df.to_parquet(f'./file{i}.parquet')
        
        MinioBucketManager().upload_file(f'./file{i}.parquet',f'datasets/{uuid_recherche}/file{i}.parquet')
        os.remove(f'file{i}.parquet')
        i=i+1
    global SUIVIE_DATASETS
    # SUIVIE_DATASETS[uuid_recherche]='done'
    SUIVIE_DATASETS['id'][f'{uuid_recherche}']={'progesse':'done'}
    sauvegarder_donnees()
    return



class NumpyTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class ModelRechercheGlobal(BaseModel):
    index:str
    n_sortie:int
    vecteur:list
    collection:'str'
    direct: Union[bool, None] = None

class ModelRechercheSmyles(BaseModel):
    smiles:str
    n_sortie:int
    collection:'str'
    direct: Union[bool, None] = None
class ModelDatasets_Gen(BaseModel):
    collection:'str'
    n_sortie:int
    direct: Union[bool, None] = None    
class ModelSuivie(BaseModel):
    uuid:str


@router.put("/status_collection") 
async def status_collection():
    print('status_collection') 
    
    data={'message':MilvusManager().status_collection()}
    print(data)
    i=0
    for element in data['message']:
        if element[1]['state']==2:
            data['message'][i][1]['state']='Loading'
        elif element[1]['state']==3:
            data['message'][i][1]['state']='Load'
        elif element[1]['state']==1:
            data['message'][i][1]['state']='NotLoad'

        i=i+1
    return data

@router.put("/recherche_par_similarites") 
async def recherche_par_similarites(data: ModelRechercheGlobal):
    print('recherche_par_similarites',data)
    if data.direct==True:
        result=MilvusManager().recherche_vectorielle(data.collection,data.index,data.n_sortie,[data.vecteur])
        result_in_list=listresult(result)
        return {result_in_list}
    else:
        result=MilvusManager().recherche_vectorielle_id(data.collection,data.index,data.n_sortie,[data.vecteur])
        liste_id=[]
        for element in result[0]:
            print(element)
            liste_id.append(element.id)
        uuid_recherche = str(uuid.uuid4())
        global SUIVIE_DATASETS
        SUIVIE_DATASETS['id']={}
        SUIVIE_DATASETS['id'][uuid_recherche]={'progesse':'in progress'}
        sauvegarder_donnees()
        upload_de_datastes(liste_id,uuid_recherche)
        return {'uuid':uuid_recherche}

@router.put("/recherche_par_smiles") 
async def recherche_par_smiles(data:ModelRechercheSmyles):
    print('recherche_par_smiles') 
    # vecteur = smiles2fingerprint(data.smiles)
    vecteur = ModelChemBERTa().smiles2embeding_cls(data.smiles)
    vecteur=vecteur.numpy().tolist()[0]
    # print(vecteur[0])
    if data.direct==True:
        result=MilvusManager().recherche_vectorielle(data.collection,'embedding_bert',data.n_sortie,[vecteur])
        result_in_list=listresult(result)
        return result_in_list
    else:
        result=MilvusManager().recherche_vectorielle_id(data.collection,'embedding_bert',data.n_sortie,[vecteur])
        liste_id=[]
        for element in result[0]:
            print(element)
            liste_id.append(element.id)
        uuid_recherche = str(uuid.uuid4())
        global SUIVIE_DATASETS
        SUIVIE_DATASETS['id']={}
        SUIVIE_DATASETS['id'][uuid_recherche]={'progesse':'in progress'}
        sauvegarder_donnees()
        upload_de_datastes(liste_id,uuid_recherche)
        return {'uuid':uuid_recherche}
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


@router.put("/random_datasets") 
async def random_datasets(data:ModelDatasets_Gen):
    print('random_datasets',data)
    liste_id=MilvusManager().all_id(data.collection)
    elements_choisis = random.sample(liste_id, data.n_sortie)
    if data.direct==True:
        result=MilvusManager().extraction_par_id(data.collection,elements_choisis)
        print(result[0]['smiles'])
        # result_in_list=listresult(result)
        # return [result]
        
        return json.dumps({"valeur": result}, cls=NumpyEncoder)

    else:
        liste_id=[]
        for element in elements_choisis:
            print(element)
            liste_id.append(element.id)
        uuid_recherche = str(uuid.uuid4())
        global SUIVIE_DATASETS
        SUIVIE_DATASETS['id']={}
        SUIVIE_DATASETS['id'][uuid_recherche]={'progesse':'in progress'} 
        sauvegarder_donnees()
        upload_de_datastes(liste_id,uuid_recherche)
        return {'uuid':uuid_recherche}
    

#     liste_id=MilvusManager().all_id()
#     elements_choisis = random.sample(liste_id, data.n_sortie)
#     print(elements_choisis)
#     uuid_recherche = str(uuid.uuid4())
#     global SUIVIE_DATASETS
#     SUIVIE_DATASETS['id']={}
#     SUIVIE_DATASETS['id'][uuid_recherche]={'progesse':'in progress'}
#     sauvegarder_donnees()
#     upload_de_datastes(elements_choisis,uuid_recherche)
#     return {'uuid':uuid_recherche}



@router.put("/suivi_uuid") 
async def suivi_uuid(data:ModelSuivie):
    global SUIVIE_DATASETS
    return SUIVIE_DATASETS['id'][data.uuid]