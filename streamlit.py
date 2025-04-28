import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.express as px
from sklearn.decomposition import PCA

SERVER_URL='https://api.morgan-coulm.fr'

def afficher_molecule(smiles):
    # Convertir SMILES en molécule
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        st.error("SMILES invalide")
        return
    
    # Créer l'image de la molécule
    img = Draw.MolToImage(mol)
    
    # Convertir l'image en buffer mémoire
    buffered = BytesIO()
    img.save(buffered, format='PNG')
    img_str = buffered.getvalue()
    
    # Afficher l'image dans Streamlit
    st.image(img_str)



def rechercheparsmile(smiles,n):
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
    }
    data={
        "smiles":smiles,
    "n_sortie": n,
    "collection":'filtre2',
    "direct": True
    }
    reponse = requests.put(f'{SERVER_URL}/milvus/recherche_par_smiles', json=data, headers=headers) 
    result = reponse.json()
    return result

def random_smiles(n):
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
    }
    data={
    "n_sortie": n,
    "collection":'filtre2',
    "direct": True
    }
    reponse = requests.put(f'{SERVER_URL}/milvus/random_datasets', json=data, headers=headers) 
    result = json.loads(reponse.json())
    liste_smiles=[]
    for element in result['valeur']:
        liste_smiles.append(element["smiles"])
    return liste_smiles
def random_datasets(n):
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
    }
    data={
    "n_sortie": n,
    "collection":'filtre2',
    "direct": True
    }
    reponse = requests.put(f'{SERVER_URL}/milvus/random_datasets', json=data, headers=headers) 
    result = json.loads(reponse.json())
    return result
def generationdecandidat(n):
    liste=[
        "[*]N1C(=O)c2ccc(Nc3ccc4oc(-c5ccc6c(c5)C(=O)N(c5ccc(-c7cccc8c7C(=O)N([*])C8=O)cc5)C6=O)nc4c3)cc2C1=O",
        "[*]Oc1c(-c2cc(C([*])=O)cc(-c3ccc4c(c3)C(=O)N(C(F)(F)F)C4=O)n2)ccc2ccccc12",
        "[*]c1ccc2c(c1)c1cc(-c3ccc(OC)c(-c4ccc5c(c4)c4cc([*])ccc4n5-c4ccc(O)cc4C)c3)ccc1n2CCCCCCCCC",
        "[*]c1cc(Oc2cc(-c3ccc(-c4cccc(Oc5nnc([*])o5)c4)cc3)nc3ccccc23)c2ccccc2n1",
        "[*]Nc1ccccc1CCc1cccc2ccc(-c3ccccc3-c3ccc(-c4ccc([*])cc4)cc3)cc12",
        "[*]C(=O)N1C(=O)c2ccc(C(=O)c3cc(N4C(=O)c5ccc([*])cc5C4=O)cc(C)c3C)cc2C1=O",
        "[*]Oc1ccc(-c2ccc(-c3cccc4c(OC5CCCC(c6ccc([*])cc6)C5)cccc34)cc2)cc1",
        "[*]c1ccc(-c2ccc3c(c2)C(=O)N([*])C3=O)c(-c2ccc(S(=O)(=O)c3ccc(Cc4ccc5[nH]c(-c6ccccc6C)nc5c4)cc3)cc2)c1",
        "[*]Oc1ccc(Oc2ccc([*])cc2C(F)(F)F)cc1-c1ccc(C)cc1",
        "[*]c1cccc(-c2ccc(-c3nnc(-c4nnc(N5C(=O)c6ccc(-c7ccc8c(c7)C(=O)N([*])C8=O)cc6C5=O)o4)o3)cc2)c1"]
    return liste
def button_interaction(smiles):
    if st.button(smiles):        
        liste_proche_voisin=[]
        resulta_model=generationdecandidat(10)
        for element in resulta_model:
            liste_proche_voisin.append(rechercheparsmile(element,1)[0])
        df_result=pd.DataFrame(liste_proche_voisin)
        df_result=df_result[["smiles","fingerprint","permN2"]]
        df_voisin=pd.DataFrame()
        for element in resulta_model:
            results=rechercheparsmile(element,11)
            # smiles = [{'smiles': r} for r in results['smiles']]
            # vectors = [{'fingerprint': r} for r in results['fingerprint']]
            # permN2 = [{'permN2': r} for r in results['permN2']]
            
            vectors = [res["fingerprint"] for res in results]
            permN2 = [res["permN2"] for res in results]
            smiles = [res["smiles"] for res in results]
            # df_voisin_bis=listen
            df_voisin_bis = pd.DataFrame({
                'smiles':smiles,
                'fingerprint':vectors,
                'permN2': permN2,
                'source':'voisin'
                })
            df_voisin = pd.concat([df_voisin, df_voisin_bis], ignore_index=True)
        
        # for element in liste_proche_voisin:
        #     afficher_molecule(element["smiles"])
        df_bdd=pd.DataFrame(random_datasets(100)['valeur'])
        df_bdd=df_bdd[["smiles","fingerprint","permN2"]]
        df_result["source"]="result"
        df_bdd['source']="bdd"

        df = pd.concat([df_bdd, df_voisin,df_result], ignore_index=True)
        X = np.array(df['fingerprint'].tolist())
        pca = PCA(n_components=2)
        X_3D = pca.fit_transform(X)
        df['pca_x'] = X_3D[:, 0] 
        df['pca_y'] = X_3D[:, 1]  
        fig = px.scatter_3d(df ,
            x='pca_x', y='pca_y', z='permN2',color='source',
            title="Visualisation 3D des vecteurs Milvus",
            hover_name='smiles',
        )
        fig.update_traces(marker_size=2)
        # fig.show()
        st.plotly_chart(fig, use_container_width=True)
        st.write('les smiles les plus proche des resultats')
        st.dataframe(df[df['source'] == 'result'])
        st.write('les 10 plus proche des resultats')
        st.dataframe(df[df['source'] == 'voisin'])



st.set_page_config(page_title="Application streamlit", layout="wide")

if 'smiles_randome' not in st.session_state:
    st.session_state.smiles_randome = [""]

st.title("Application Streamlit ")

st.subheader("5 smiles alleatoirs")
if st.button('Cliquez-moi'):
    smiles_randome=random_smiles(5)
    smiles_randome=[""] + smiles_randome
    st.session_state.smiles_randome = smiles_randome

smiles_randome=st.session_state.smiles_randome



for element in smiles_randome:
    if element != "":
        afficher_molecule(element)
        button_interaction(element)
