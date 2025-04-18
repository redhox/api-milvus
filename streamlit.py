import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import pandas as pd
SERVER_URL='https://api.morgan-coulm.fr'

def rechercheparsmile(smiles,n):
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
    }
    data={
        "smiles":smiles,
    "n_sortie": n,
    "collection":'filtre1',
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
    "collection":'filtre1',
    "direct": True
    }
    reponse = requests.put(f'{SERVER_URL}/milvus/random_datasets', json=data, headers=headers) 
    result = json.loads(reponse.json())
    liste_smiles=[]
    for element in result['valeur']:
        liste_smiles.append(element["smiles"])
    return liste_smiles
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



# Configuration de la page
st.set_page_config(page_title="Application streamlit", layout="wide")

if 'smiles_randome' not in st.session_state:
    st.session_state.smiles_randome = [""]

# Titre principal
st.title("Application Streamlit ")
print('un  ?')
st.subheader("5 smiles alleatoirs")
if st.button('Cliquez-moi'):
    smiles_randome=random_smiles(5)
    smiles_randome=[""] + smiles_randome
    st.session_state.smiles_randome = smiles_randome

smiles_randome=st.session_state.smiles_randome
smiles_depart = st.selectbox("Sélectionnez un élément :", smiles_randome)
print('un')
if smiles_depart !="":
    print('truc')
    st.write(smiles_depart)
    st.write("generation de smiles")
    # st.write(generationdecandidat(10))
    print('un truc ?')
    liste_proche_voisin=[]
    for element in generationdecandidat(10):
        liste_proche_voisin.append(rechercheparsmile(element,1)[0])
    df=pd.DataFrame(liste_proche_voisin)
    st.dataframe(df)





# st.header("recherche des plus proche voisin avec l'api milvus")

# # Section 1 : Entrées utilisateur
# st.subheader("Zone d'entrée de smiles")
# smiles = st.text_input("Entrez votre nom :", placeholder="Jean Dupont")
# if smiles:
#     st.subheader("📋 Liste de sélection")
#     result=rechercheparsmile(smiles)
    
#     liste_plus_proche_smiles = [element['smiles'] for element in result]
#     liste_plus_proche_full={element['smiles']:element for element in result}
#     liste_plus_proche_smiles=[""]+liste_plus_proche_smiles
#     selection = st.selectbox("Sélectionnez un élément :", liste_plus_proche_smiles)

#     if selection !="":
#         st.write(liste_plus_proche_full[selection])


