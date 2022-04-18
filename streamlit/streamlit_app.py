from logging.handlers import SocketHandler
from sqlite3 import Date
import pandas as pd 
import streamlit as st 
st.set_page_config(layout="wide")
import seaborn as sns 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image

from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier , StackingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from bokeh.plotting import figure
from bokeh.io import  push_notebook,output_notebook, show
output_notebook()
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.models.tools import HoverTool
from bokeh.transform import factor_cmap
from math import pi
from bokeh.palettes import brewer, Spectral, Viridis3, Viridis256, d3
from bokeh.transform import cumsum

import shap
 #pip install streamlit_shap
from streamlit_shap import st_shap
import streamlit.components.v1 as components

#from skater.model import InMemoryModel
#from skater.core.explanations import Interpretation
#from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer

#chargement des Datasets et des modeles
df = pd.read_csv("bank.csv")
template  = pd.read_csv("template.csv")

clf_grad = load('GradientBoostingClassifier().joblib') 
clf_log = load('LogisticRegression().joblib') 
clf_rdf = load('RandomForestClassifier().joblib') 
sclf = load('sclf2.joblib') 
scaler = load('scaler2.joblib') 

#load models for shap
clf_grad_shap = load('clf_grad.joblib') 
clf_log_shap = load('clf_log.joblib') 

# fonction de preparation des données clients pour prediction
def data_prepare(client):
    client.default = client.default.replace({'yes' : 1, 'no' : 0})
    client.loan = client.loan.replace({'yes' : 1, 'no' : 0})
    client.housing = client.housing.replace({'yes' : 1, 'no' : 0})
    client.month = client.month.replace({'may' : 5, 'jun' : 6, 'jul' : 7, 'aug' : 8, 
                             'oct' : 10, 'nov' : 11, 'dec' : 12, 'jan' : 1, 
                             'feb' : 2, 'mar' : 3, 'apr' : 4, 'sep' : 9})

    liste = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
    for i in liste:
        col = 'job_'+i
        client[col] = [1 if x == i else 0 for x in client.job]
    client = client.drop('job', axis = 1)

    liste = ['divorced', 'married', 'single']
    for i in liste:
        col = 'marital_'+i
        client[col] = [1 if x == i else 0 for x in client.marital]
    client = client.drop('marital', axis = 1)

    liste = ['primary', 'secondary', 'tertiary', 'unknown']
    for i in liste:
         col = 'education_'+i
         client[col] = [1 if x == i else 0 for x in client.education]
    client = client.drop('education', axis = 1)

    liste = ['cellular', 'telephone', 'unknown']
    for i in liste:
        col = 'contact_'+i
        client[col] = [1 if x == i else 0 for x in client.contact]
    client = client.drop('contact', axis = 1)

    liste = ['failure', 'other', 'success', 'unknown']
    for i in liste:
        col = 'poutcome_'+i
        client[col] = [1 if x == i else 0 for x in client.poutcome]
    client = client.drop('poutcome', axis = 1)

    client = scaler.transform(client)
    client = pd.DataFrame(client, columns = ['age', 'default', 'balance', 'housing', 'loan', 'day', 'month',
                                      'duration', 'campaign', 'pdays', 'previous', 'job_admin.',
                                      'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
                                      'job_management', 'job_retired', 'job_self-employed', 'job_services',
                                      'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
                                      'marital_divorced', 'marital_married', 'marital_single',
                                      'education_primary', 'education_secondary', 'education_tertiary',
                                      'education_unknown', 'contact_cellular', 'contact_telephone',
                                      'contact_unknown', 'poutcome_failure', 'poutcome_other',
                                      'poutcome_success', 'poutcome_unknown'])

    client = client.drop('duration', axis = 1)
    client = client.drop('job_unknown', axis = 1)
    client = client.drop('education_unknown', axis = 1)
    client = client.drop('contact_unknown', axis = 1)
    client = client.drop('poutcome_other', axis = 1)
    client = client.drop('poutcome_unknown', axis = 1)
    #st.dataframe(client)
    return client

#Fonction de prediction
def predict2(client_df):
    if model == "RandomForestClassifier" :
        preds = clf_rdf.predict(client_df)
        pred_proba = clf_rdf.predict_proba(client_df)[:,1]
    
    elif model == "LogisticRegression" :
        preds = clf_log.predict(client_df)
        pred_proba = clf_log.predict_proba(client_df)[:,1]

    elif model == "GradientBoostingClassifier" :
        preds = clf_grad.predict(client_df)
        pred_proba = clf_grad.predict_proba(client_df)[:,1]

    else:
        preds = sclf.predict(client_df)
        pred_proba = sclf.predict_proba(client_df)[:,1]

    return preds, pred_proba

# 2EME FONCTION DE PREPARATION DES DONNEES AVEC STANDARDSCALER

def data_prepare2(client):
    client.default = client.default.replace({'yes' : 1, 'no' : 0})
    client.loan = client.loan.replace({'yes' : 1, 'no' : 0})
    client.housing = client.housing.replace({'yes' : 1, 'no' : 0})
    client.month = client.month.replace({'may' : 5, 'jun' : 6, 'jul' : 7, 'aug' : 8, 
                             'oct' : 10, 'nov' : 11, 'dec' : 12, 'jan' : 1, 
                             'feb' : 2, 'mar' : 3, 'apr' : 4, 'sep' : 9})
    
    liste = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
    for i in liste:
        col = 'job_'+i
        client[col] = [1 if x == i else 0 for x in client.job]
    client = client.drop('job', axis = 1)

    liste = ['divorced', 'married', 'single']
    for i in liste:
        col = 'marital_'+i
        client[col] = [1 if x == i else 0 for x in client.marital]
    client = client.drop('marital', axis = 1)

    liste = ['primary', 'secondary', 'tertiary', 'unknown']
    for i in liste:
         col = 'education_'+i
         client[col] = [1 if x == i else 0 for x in client.education]
    client = client.drop('education', axis = 1)

    liste = ['cellular', 'telephone', 'unknown']
    for i in liste:
        col = 'contact_'+i
        client[col] = [1 if x == i else 0 for x in client.contact]
    client = client.drop('contact', axis = 1)

    liste = ['failure', 'other', 'success', 'unknown']
    for i in liste:
        col = 'poutcome_'+i
        client[col] = [1 if x == i else 0 for x in client.poutcome]
    client = client.drop('poutcome', axis = 1)

    # On normalise les données
    from sklearn import preprocessing
    scaler2 = preprocessing.StandardScaler().fit(client)
    client = scaler2.transform(client)

    # On transforme à nouveau client en DataFrame
    client = pd.DataFrame(client, columns = ['age', 'default', 'balance', 'housing', 'loan', 'day', 'month',
    'duration', 'campaign', 'pdays', 'previous', 'job_admin.',
    'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
    'job_management', 'job_retired', 'job_self-employed', 'job_services',
    'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
    'marital_divorced', 'marital_married', 'marital_single',
    'education_primary', 'education_secondary', 'education_tertiary',
    'education_unknown', 'contact_cellular', 'contact_telephone',
    'contact_unknown', 'poutcome_failure', 'poutcome_other',
    'poutcome_success', 'poutcome_unknown'])

    # On élimine les variables choisies
    client = client.drop('duration', axis = 1)
    client = client.drop('job_unknown', axis = 1)
    client = client.drop('education_unknown', axis = 1)
    client = client.drop('contact_unknown', axis = 1)
    client = client.drop('poutcome_other', axis = 1)
    client = client.drop('poutcome_unknown', axis = 1)

    return client

# 3EME FONCTION DE PREPARATION DES DONNEES SANS STANDARDSCALER POUR INTERPRETABILITE AVEC SKATER

def data_prepare3(client):
    client.default = client.default.replace({'yes' : 1, 'no' : 0})
    client.loan = client.loan.replace({'yes' : 1, 'no' : 0})
    client.housing = client.housing.replace({'yes' : 1, 'no' : 0})
    client.month = client.month.replace({'may' : 5, 'jun' : 6, 'jul' : 7, 'aug' : 8, 
                             'oct' : 10, 'nov' : 11, 'dec' : 12, 'jan' : 1, 
                             'feb' : 2, 'mar' : 3, 'apr' : 4, 'sep' : 9})

    liste = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
    for i in liste:
        col = 'job_'+i
        client[col] = [1 if x == i else 0 for x in client.job]
    client = client.drop('job', axis = 1)

    liste = ['divorced', 'married', 'single']
    for i in liste:
        col = 'marital_'+i
        client[col] = [1 if x == i else 0 for x in client.marital]
    client = client.drop('marital', axis = 1)

    liste = ['primary', 'secondary', 'tertiary', 'unknown']
    for i in liste:
         col = 'education_'+i
         client[col] = [1 if x == i else 0 for x in client.education]
    client = client.drop('education', axis = 1)

    liste = ['cellular', 'telephone', 'unknown']
    for i in liste:
        col = 'contact_'+i
        client[col] = [1 if x == i else 0 for x in client.contact]
    client = client.drop('contact', axis = 1)

    liste = ['failure', 'other', 'success', 'unknown']
    for i in liste:
        col = 'poutcome_'+i
        client[col] = [1 if x == i else 0 for x in client.poutcome]
    client = client.drop('poutcome', axis = 1)

    # On transforme à nouveau client en DataFrame
    client = pd.DataFrame(client, columns = ['age', 'default', 'balance', 'housing', 'loan', 'day', 'month',
                                      'duration', 'campaign', 'pdays', 'previous', 'job_admin.',
                                      'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
                                      'job_management', 'job_retired', 'job_self-employed', 'job_services',
                                      'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
                                      'marital_divorced', 'marital_married', 'marital_single',
                                      'education_primary', 'education_secondary', 'education_tertiary',
                                      'education_unknown', 'contact_cellular', 'contact_telephone',
                                      'contact_unknown', 'poutcome_failure', 'poutcome_other',
                                      'poutcome_success', 'poutcome_unknown'])

    # On élimine les variables choisies
    client = client.drop('duration', axis = 1)
    client = client.drop('job_unknown', axis = 1)
    client = client.drop('education_unknown', axis = 1)
    client = client.drop('contact_unknown', axis = 1)
    client = client.drop('poutcome_other', axis = 1)
    client = client.drop('poutcome_unknown', axis = 1)

    return client

#creation d'un dataset de reference pour Shap
df_shap  = df.drop('deposit', axis =1)
df_shap = data_prepare(df_shap) 

# create sidebar
image = Image.open('image_bank.jpg')
st.sidebar.image(image)
st.sidebar.title("Projet Depositary_Hunt")
st.sidebar.write("## Sommaire")
pages = ["Introduction",'Analyse de données',"Dataviz",'Préprocessing - Modélisation - Interprétabilité globale','Dashboard client', 'Remerciements']
page = st.sidebar.radio("Aller  à la page:", pages)

# menu introduction
if page==pages[0]:
    st.title("Projet Depositary_Hunt")

    st.write("### Prédiction du succès d’une campagne de Marketing d’une banque")
    st.write("###  Presentation")
    st.markdown('L’analyse des données marketing est une problématique très classique des sciences des données appliquées dans les entreprises de service. Pour ce jeu de données, nous avons des données personnelles sur des clients d’une banque qui ont été “télémarketés” pour souscrire à un produit que l’on appelle un "dépôt à terme”. Lorsqu’un client souscrit à ce produit, il place une quantité d’argent dans un compte spécifique et ne pourra pas toucher ces fonds avant l’expiration du terme. En échange, le client reçoit des intérêts de la part de la banque à la fin du terme.')
    st.markdown('Pour ce projet, il faudra d’abord effectuer une analyse visuelle et statistique des facteurs pouvant expliquer le lien entre les données personnelles du client (âge, statut marital, quantité d’argent placé dans la banque, nombre de fois que le client a été contacté, etc.) et la variable cible “Est-ce que le client a souscrit au dépôt à terme?”. ')
    st.markdown('Une fois l’analyse visuelle terminée, vous devrez utiliser les techniques de machine learning vues pendant la formation pour déterminer à l’avance si un client va souscrire au produit ou non. Une fois cette prédiction réalisée, vous devrez utiliser les techniques d’interprétabilité des modèles de machine learning pour expliquer à l’échelle d’un individu pourquoi il est plus susceptible de souscrire au produit ou non. L’interprétabilité du modèle sera fondamentale pour conclure votre analyse, et donc le projet.')
    st.markdown('les données provienent de Kaggle et sont disponibles à l\'adresse suivante:')
    st.markdown("Bank Marketing Analysis (https://www.kaggle.com/janiobachmann/bank-marketing-dataset)")
    st.markdown('Trouvez les meilleures stratégies à améliorer pour la prochaine campagne marketing. Comment l\'institution financière peut-elle avoir une plus grande efficacité pour les futures campagnes de marketing ? Pour y répondre, nous devons analyser la dernière campagne de marketing effectuée par la banque et identifier les modèles qui nous aideront à tirer des conclusions afin de développer de futures stratégies.')

# menu Analyse de données
elif page==pages[1]:
    st.title("Projet Depositary_Hunt")
    st.write("## Presentation du projet")
    
    st.image("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAoHCA8UDxAPEg8SDw8PDw8PDw8QDxIPDw8PGBQZGhgUJBwpIy4lKR4rHxgYJjgmKy8xNTU1GiQ7QDs0Py40NTEBDAwMEA8QGRISHDQkGCMxNDExMTE0NDQ0NDQ0MTQxNDE0MTQxNDE0NDQ0NDQ0PzQ0ND80PzQ0NDQ0Pz80NDQ0P//AABEIAMIBBAMBIgACEQEDEQH/xAAcAAABBQEBAQAAAAAAAAAAAAAAAQIDBAUGBwj/xABBEAACAQICBgcECgEDAwUAAAABAgADEgQRBSEiMUFRBhMyQmFxgVKRobEjM2JygpLB0eHwFAei8RU0skNTc5PS/8QAGgEBAAMBAQEAAAAAAAAAAAAAAAECAwQFBv/EACcRAAICAgEEAAYDAAAAAAAAAAABAhEDITEEEkFRBRQiYXGhEzKB/9oADAMBAAIRAxEAPwD2aEIQAhCEAIQhACEIQAhCEAIRCZlY3TmHpb2DHwyy9/H0zgGtCck3SXEVPqMNUce0Kbsv5v4kbVNMvupWDxdF+RBgHYwnEtQ02OGflWP/AO5E2kdMUu3QdgOSXj3i6Ad3CcNhumrKba1L72q1l8/+J0GA6Q4arucKeTbvfw9coBswjQc46AEIQgBCEIAQhCAEIQgBCEIAQhCAEIQgBCEIAQhCAEIQgBCEIATP0hpGnRQsza+X8/0yLSekOrFqAvVc2qq7TXHgBz+A3nxhwGidoV8RlUrb1XtJS8uZ+0YBRFLG4vWW/wAfD8GYbbr4Jw821zRwegMLT2ur61+L1fpG+OoTYhAGgZbo6EIAQhCAU8Zo+hWFtWklTxZRmPI75yulehQ7eFcofYdj8G3j1znbQgHmmC07jMG/VV1bZ7SP7PMcD5j4zuNFaXpV0uQ7XeXj45c5JpPRlHEJ1dVbh3SNTKeYM88x+j8TgK4dWLU2OxUGoNlwPJoB6lCYfR7Ti4hMjs1V7S85uQAhCEAIQhACEIQAhCEAIQhACEIQAhCJACEI0mAOzhnG5wzgDs5Sx+KFNCx5ebctQ4nMgAcSRLTNkNcy6KddXLN9XQbIDnXy+Sg5feJ5QB2jcCVPX1PrnXUN/VIddg8eZ4ma0IkAWJEJkb1kG8iASwzlN8cg3ZtKz6RPdA9ZNMi0aucTOYb4yqeNvlsyIVnBzvN0URZt1awXfn6SBsYfY1fe1yn/AJNx171X+mVMRXe4gE5eE+b6r4nkx5pQXCfo6oYlJWdBTqht3u4yHG4SnVptSqKGRhrHLkRyI5zP0KNbnwVfnNUvPZ6TNLNiU5KmzGce2VHmOMw1bA4rIMbe0j7usTPV6j5z0TRGkUr0RUXf3h9qUukejlxFAoPrU26Z+1xX1Gr3TlOiukWo1urbsPs28m/vy8Z1FT0a6LnIBUjg0AmzhIw0cDAHwiRYAQhCAEIQgBCEIARDFkbVFG8iAKTEkD4tB9rykL43kvvk0xaLsQmZrYpzxtkTOTvJk0R3FvSGKsRrcmfurzckBB6sV+MmwdEU6SJndaNpvaY62Y+ZJPrMbCK7PSDgKesqVGA7NlMWofzPn+Hwmy9x4+6RQskerlKtXHKO/wC6IyDiS3m0gZ6a7svSSkQ5JENTGuc7abM3jsrIGbEHgqD8xklTFLwWVnxTndksuoMo8kS0ZGzqN5Epu7HeY1abkXBTl92Q+2P9mU72+EadAI2+TsyKNdsy8LiAp6snabaWLUbWfvT5L4j1GWOeSt0nrZ6WHGpRTZi6dx5o4lHV7qdTZVbbSjA/Ea42vpOqScsl8l/eZnTCqDUwoBDbT97xWPJmUo98Yzltvk7IJVXo6zotXJSqzEs16rr8B/M2HrzjtF1yiamt2m/Saa6RVtk7LfCfTdJHtwxS9Hm5neRmw1ecZp3D2YkumyH+kX7LZ6/jr9ZvitnM/TiXUkfijf7T/wACdBmdBozF30Uf7K/myEvo85fo3V2HT2W/2/0zoKecAvK0kUyBBJ1EAeI6NEdACEIQAhCEAIQhAKGODaiCcpTymniVzQ+G1M17huF0tHZWWhLYZSgcTVvta1Fu8W+J1RcTTVk7bO3hnb8CM5p2mbkXbl7OYkdSsq6jKmCc7goVl2tbDNV9CZaxdFXTPte1lJpWQ5OtD8G/0zt7NCiv5y7n5iWalYzPw75Vqy/Zw6r/APSur5yV3kRjYlKgqPKztHOZE5/NNEjJsQiRt7IkyUWO4ydcLq7X98ZyZeuwYpdspbNI4ZyVpDMGyDUcrvHfbLTYpBuOflOb0niurrIA1r2t+LI6xK1XSdQ7sk8l/efM/Ee9524u4vaPU6fDcEyr0rxRp4mi6bIdWVk7upt498KmJdu05b8U57pPi2vo3kt29ot2dazSq42koF1RV2fauPuETxSlCDe3XJ2YoNWkjO0zUH+ThlPeu+YmrWxCJ23C+bfpOR6QYxamJpWZ2qnEW7Vx3Rk6vl7jG/R14em7rcnWzpRpGq79XRtRG2utdS13kuYmrSqHIKTc3eIW3NueXCcVhlDPatAtUXvlrfdlrym9oSoaZfrqqqi9ikih9rnqJb3z6LHirHFR8JHzGaajmkm9Wzq8JWfjtD4y9jKd1B2+zd7iDMbDYvEVc/8AEwodVa1qlaoKaD8IzJnQ1aRXDPfkrtTa9UYsl2XAmRKDjyTGakrRn9GfriPaX952CUpx/R3ViRO2EzZdAqx4iRRBIojo0R0EBCEIAQhCAEIQgCGZdWnvUzUMqV02vvS0XsrJaOexdNAQRT7Wzc9i7Xmf0Et0WdkBLKnl4eYElxtIkHOpZ3lyyU+87pRwNSnmVDdYzbWp3qbQ35tu9BNeTF6I2ULU7ZfaVlUXtsn7IAA9Zq4fXqCWpIMVTdkzVQlu1rvy9QN/lG4TdkzlQtrKtnVr5Bd+r9YfoL2Q1FbrKqjLbp4ZtfgLSR47MBVzGvtrssPtftxkmIGdaky9l1qU/wAQN4/8iPSQOlh6zMt7efBPDwG/1MQ4IycjxT5mPWlxjDXAe0+zcvjzEir49F3kL5tte4T57rfiefFmcI0qfrk7MPSxlBSd7I3x6pWFM72S5fHWRJa+Oy3sqfinD9KtJqMThip/F2d51/pLbPxPvM4Orx/yyWRa7ldHdhx0u1+Cp0wx6GvhmVva1/iWWg2rOcp0uxCMaNrhrbrsmutzyylZ8VUcC52bZXUWNvunX8v3Y4brR29N0/c2rqix0urozUQrhrVa7Jrrda75TB1SM6PqV3CU1ude7+82sL0bxT9pBTXm7D5Cd0MLUIxRvhy48MpKTqvZzeIP06eS/My+DynX4LoxQUFan09RuAU7PgAN3mZv6O0Cq5dXQp0/F8mf4a/jOj+BurZzP4jGLl2pu2YmjujP+RRQVi1LZWw3FeG4jiJImDfDucPTwF3s1au2tRfaVVG7zPnO7oaOUZM7lrfwIPT9zJf8ugCFU3W8luA9Z2xzxxqnweBlwSytvycougMRWTJ6lSmjf+nSyoU/UDWfWaNbR1SjQdnxFSpsLTVC1yW5jnrPnOhFTkkydO180Snlbc1zemofOU+ZU9J8lI4VCudGfoRPpP7yP8TqaOII2W3c5g6DRQTn2rd3e1nL9D75fOLzyyQ7TW7exa2eWWW/hyjXk0+5tKyndHiZqOy7pco1g32TykNFkycR0aI6QAhCEAIQhACEIQBBIa66pMI1hnJXIZn4hUyzZLu7Mmo9RX1KFVWuW+oEHogGZ1c5vEatUy8XRq6nuKd1rKYeo3LXwG/fNU9GElsn6skZvU1Mv3dkzLRaSVNWRdX7ISpVqFTzY7s/KaGGRVW12YsD33DOw5nLdDFNmBYjtb3Efq8/MjhBUjx6uyFwljU7aifeTM5flLe6Q4tFdEqBQyPa6/ZaW8IHb6POmllrLTRr7U8fWV8PahfDN2WuqYcnu809CCPQyt0y9dyo5bSOKsdKbNqudFa7s5gFQfLd6jnMutiqadtwvm03dN6NSoj5sEbtXHJdrcMz7h7pzOG6Hs23WxAVW9jf6s3H0nmdX0Ty5O9HrdHlwrHU3Vfs57pNWWu9Lqc3ZLtyldrVll8Y5KWKr7ISpU/CclbxO4T0DRfR+hTN1Kgzv/7j/uchl5CbdHRTntOqLyRbiPU6ufCax6aKiot8F49dHFJuCtPizzzAdEqj7VfJE9gNcT5ncJqYPorhczktSvtbK67PLMZAj1nYPhUVwgw9Su2y3WOydWFJ4EnLnqA+ecu4ZnCHrVpqbmtWlmy2cM8+O/wm6hGkkjnl1eRycrq+aMHD6JWknZp4anz1Z/oPnLOFwtByFHWVbrmvKnql3nI5eu/9RNZ3VhbYGVu6+18JWdMS72ratPmc7t3AAZDXzmlP8HLLJbvlkyUqa6hkq+yOzLiIqjM7l2pVw1FFKU3qhqjXW7rm1+GrnNPqVsKHssrLr8ZVxXglSb54OG0jj3rPvK0g2yvyJ5n5STCVnXftf+UuHQ9RXKez2XO4rzmjh9Fova22+HunFkkk98npPLjUEkTYPFqUu17P2Zk4973L93sr90TVxb2JYuyzcuCzIr8hNsNRi5y0edk3L6TBx9dusvQlWTZUjw/pmtoXTy1j1FUmnVXsujW35Z6j+0o4zAU+65VvZ7QmFQw9Ra6Jlt3XavZ5zsw9V0/UQqD+pe+TzezPjy/UtN/4el4bCFdovr7TKm4+ZOsyxlDCg2JnvtkhWQdrJKOJy1N+aXAc90zCJNh1cbt3jukNBMvwhCVLBCEIAQhCAEaY6NMAhIlPE4UsGzYheStafIHhL5EhxFC7LXlbLRZnJGPh0pUiCwp0x2WvY1Kj+yM93zl969wKqlwPD+BKFWglN2YdUl3fZi7ljxCnUPLOXKGPDoGp7a9m+3K5hqJyl3szuvNfgzjVqK9penTVW+ow9MvUK8iBuk2MoKQFD7d1yN2cn5eAOQ9cjzjdILXyuNTqkPayp31C3DID9ouGpoQb+stW1VNRQnWat+W+S0Sva/ZHQRK2avsuuy6Fd7DUTrlh8GiAvTpU2qXL29nPXr15aufnIMRR19bTJuXve1yBz+fHcY7D17zke37JmdN6bL2ltIVRVFj1a6rYzMyUV2GXUdbHh2h6g7xLC4sOM0yYbS3XXbQORlavo2kCalYswuZlQs7KNRzyA8C2/mZawiAFQlC2nl2zkmWrVs5ZHdlq8IpIm5MjqU6jgqptbutbcF9JXp4anTqA1MQWqNsKC3gO6NQ7u/iRzm0XUfxInqb2Ca/DtNl/yY34IpLkZhFRkvsZf/kW0+clqqjAoVuDctn48JCi1W3qE2uO3cv6S4tGKXkm34KQZENiUrS3sKOPiZNQWowzdQrfeulsU1EjKE9pvQat3jF+iavkGw4YZN+E8RKeIBpjMtd7PMyfFY9KYy7T91ePrynP4rFljc+0T2UHyAmThGTtosnXAYmv3ztFuyv94TDfFl3sp7SbV9Tm+WpR4k/Ca9TQlbEUGIqCkz22ZqSrJnrByyIB58ZhY3D4rAugrKKmHGxTqov0S3ayCODHx9JMoRlFwa0E92S08NiXJFOlbtWs78PSb2h+jth6yo19TmZmrpcrTvpG6o1qrk2s+B5j5eUKnTpKL0Uqpcj5o1VG271AuYJlrXM8DwMzw9Njw/1X+kyk5cnZBMtkRy0WO/Zhg8TSqotWk61Kbi5XRrlaWBN+4qkMSko4e+SiAiyCQhCEAIQhACEIQAjTHRpgDDEqJmCOcUxwkplZKzJxeCQZPlTZ171Xco5gCR4bHIGKCoKjNtKqKFRQMwcsv15S5XwCtdftK3dlEHq+7Sw1NCGbczFBv8BNLT+5jTX2LDmqwNuz7Jt1CZlOkFfaqVcTURrWVFtUZ89w1TXGMuAKZZNtKd+azPx6vqZ8QKFLy2mbw/iTv8EKn9zQXql1Dvdrve+U8fo/L6RPxL/d/wA/PdJNHUksK/SMq9l6uYv8QTvmipVgQNfdlXReLfBj4XSDjYba89/of0OuWetqMSoyVbdli3e/ucmbRtNu1m3+3/mVnwdSnrp1Aw5Pv98jzot42XqVAgBWa485OtICY/8A1RU+spslvEMGW3nnqj/+sUjuqn8pb5CHZKo1HRjua0eAgqKueXHtTDr6T5VT6Kf2lF8Q79nrKnle3zlSx0tbH0k7VQE+yNZ+EysVpdm1ILBzPb/YSpS0diH3U7F5u1vw3y7Q0Og11GNVvYTUv8wDNorUqvlTBc95z2B5mb2A0alLbc31fbPDwAk6EKLUUIq90LFBkWTRManL4yHE0VqI9N1uV1KsCoYaxyjxFkWTR5Tpc/4FY4eow6tkV/olsapSJOq4nMElSDlynEaU0k1et1mQRE2aaJ2UWe7dIujuFx1LqqybS3dXVTZq0mPEHlzB1GeJdJ+i+KwD5VF6yg7W08Qi7D+BHBvA+mcEUbXRHpI9B8lcJeRcrm3D1vvey3AOPXOevaL0pTrg5ZpVQDrKL5B0z3HxU8GGoz5wp1J1nR3pO1KynWNRqafU4hP+4w2fLPtJzQ6pIPdRHTA0TptXVFqMmdT6mvT/AO3xPgM+y/NDr5ZzeEAWEIQAhCEAIQhACNMdGGANMVYhiIYQZHiaRbLKZWIwCLUDdQ1Z27TXW00X1M223apk4yjVdNTmn7T/AGfCXi3VWYySTuifD1bUycJf7NPsheGuNxOIZhsIGYbSA+158JRwNGirixqld22XqdpVA4k7hNm9F3Zektrxsq0/LpGMpqqTUxFVES1foxtFfHPefdxmxQpKuvPeJj4usiPmmHL1KnetuO/md0tYda7i6ooXa2dru+MNNreiLSdpWX8RnbmsziC3NpqImQyOuKtMDcAJCl2lnByOdw3Rtbmeo9SoTcu257B1ZZDwm1h9H00W0ILeWWr3SdqqjxkTVid2zKubZeMEiSymvdVfQRr4jlIYZStl0hWcneYCII4SCwCOESKIA4R0aIsAWR4rDU6qPSqotSm62ujrcrL4iSRRAPG+mX+ndTD34nBBquHG09DW9eio35cXX4jx3zhKVWfUInA9M/8AT2libsRhLaGK2mdOxRxDbznkNTeI1Hjzgg860Hp2phyVVRUoVPrsM/1dTLiOTeI1z1PQHSim9PMVHq4dcry+1icH4OBrdPtjWOPOeJYmhVo1HpVabUqqNayOtpH8eO4y3gMfUpuKlOoadRey473geYkkH0jTqKyhlIYEBgym4EHcQeIk08h6OdM2ptkUGRa6ph+zTcne1MnssfZ3Hwnp2i9I0MRTFWk4ZOI76txVl4Hwgk0IQhACEIQAjDHxpgDDGxxjTBJNKuIq26ss8+e6WFMZUpgnXJVXszkm1owjVxTkrTpCkg2VdsvfkJo0MI+QvbXltZcTLwAEC4lnJ+CixryR06CruEkLAb5Ezk/ZjCsq2aKNEjV+UhZyd5i2wtkWWoZlCPthbIJGZRY62FsAblFjrYtsAbFi2xbYAgjhDKKBAARRACLlAFEdGiOEAwOk3RXC4+nZWS2qoPV4hcutTw8V8D8DrniHSPo3jNH1LKy502b6PEL9VV8uR+ydfnvn0gJUx2BpV6T0a1NalKotro+4/wA+IkkHzVSr575uaH09icNUFWnUtPZbPWjr7LrxHjvE0+mf+ntfCX4jC3V8LtMydqrh18faUe1v585xtDEQQfQHRfpXh8amQPV4hRm9BjteLKeK/LjOknzThq7o6VKblHRlZWDWsrDiDwnuXQjS1TFYCnWq/WgtTdssr2U5XZDVr1buOcEnSQhCAEaY6JAGkRpEkyiZQBq6oFuUdlDKARkQtkmULYBHbEtktsLYBFbC2S2wtgEVsLZLbC2ARWwtktsLYBHbC2SWwtgEdsLZJbC2AR5QykmUMoAzKLlHZRcoA0CLFyhACLCEAQzzXpt/pwle/E4ELRxG9sP2KVXmRq2W3+B8N89LhAPnzRPQvS1SsKRwtSgLturX2EQZ6z9r8Oec9y0LoynhsNSwtPWtNbbjvdjrZj4kkn1mhlFgBCEIAQhCAEIQgBCEIAQhCAEIQgBCEIAQhCAEIQgBCEIAQhCAEIQgBCEIAQhCAEIQgBCEIAQhCAEIQgH/2Q==")
    
    st.markdown("")
    st.markdown("Le dataset est constitué de 11 162 observations contenant 17 variables :")
    st.markdown("- 7 variables quantitatives")
    st.markdown('- 10 variables catégorielles (dont la variable cible "deposit".')
    st.markdown("Les variables peuvent être rassemblées en 3 grands groupes :")
    st.markdown('- Les variables relatives aux clients ("age", "job", "marital", "education", "balance", "default", "housing" et "loan")')
    st.markdown('- Les variables relatives à la campagne précédente ("pdays", "previous", "poutcome").')
    st.markdown('- Les variables relatives à la campagne actuelle ("contact", "day", "month", "duration", "campain" et "deposit")')
    st.markdown("")
    # Afficher les données
    if st.checkbox("Afficher les définitions des données du dataset"):
         lst1 = ['age','job','marital', 'education', 'balance','default', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'deposit']
         lst2 = ["Age du client", "Type d'emploi du client", "Statut marital", "Niveau d'étude", "Solde du compte", "Le client a-t-il fait défaut sur un emprunt?", "Emprunt Immobilier", "Emprunt à la consommation", "Moyen de communication utilisé", "Jour du dernier contact", "Mois du dernier contact (au format texte)", "Durée du dernier contact en secondes", "Nombre d'appels pendant la campagne", "Nombre de jour depuis la campagne précédente", "Nombre d'appels pendant les campagnes précédentes", "Résultat de la campagne précédente", "le client a-t-il souscrit ?"]
         lst3 = ['variable quantitative', 'variable catégorielle', 'variable catégorielle', 'variable catégorielle', 'variable quantitative', 'variable catégorielle', 'variable catégorielle', 'variable catégorielle', 'variable catégorielle', 'variable quantitative', 'variable catégorielle', 'variable quantitative', 'variable quantitative', 'variable quantitative', 'variable quantitative', 'variable catégorielle', 'variable catégorielle']
         df_definition = pd.DataFrame(list(zip(lst1,lst2,lst3)), columns = ['Variables','Définition de la variable', 'Type de variable'])
         st.dataframe(df_definition)
    st.markdown("")
    # Afficher les données
    if st.checkbox("Afficher les données du dataset"):
         st.dataframe(df)
 
    st.markdown("")
  
    # Afficher les variable cible
    if st.checkbox("Afficher la variable cible"):
        scan = pd.DataFrame(columns=['deposit', 'count', 'pourcent'])
        scan['deposit'] = df['deposit'].value_counts()
        scan['count'] = df['deposit'].value_counts().values.tolist()
        scan['pourcent'] = df['deposit'].value_counts(normalize = True).round(2)
        st.dataframe(scan)
        fig1 = plt.figure(figsize=(4,2))
        sns.countplot(x='deposit', data=df);
        st.pyplot(fig1)
 
    st.markdown("")
    
    # Afficher la matrice de corrélation
    if st.checkbox("Afficher la matrice de corrélation"):
        dfcor = df
        dfcor['deposit'] = dfcor['deposit'].replace({'no': 0, 'yes': 1})
        cor = dfcor.corr().round(2)
        fig2 = plt.figure(figsize=(5,3))
        ax = sns.heatmap(cor, annot = True, cmap = "RdYlGn", vmin = -0.7, vmax = 0.7)
        ax.set_title('MATRICE DE CORRELATION\n');
        st.pyplot(fig2)

# menu Dataviz
elif page==pages[2]:
    st.title("Projet Depositary_Hunt")
    st.write("## Dataviz")
    st.markdown('cette section est dediée à la visualisation des données ayant servi à construire les modèles de prediction')

    variable_display1 = st.selectbox('selectionnez une variable', ('age','balance', 'duration', 'campaign', 'pdays', 'day'))
    st.write('votre choix:', variable_display1)
    fig1 = plt.figure(figsize=(10,5))
    sns.histplot(x =variable_display1 , bins = 30, data =  df,hue = 'deposit', kde=True, multiple="dodge" );
    st.pyplot(fig1)
    
    variable_display2 = st.selectbox('Sélectionnez une variable', ('job', 'marital', 'education', 'default', 'housing', 'loan', 'contact'))
    st.write('Votre choix:', variable_display2)
    #Preparing data for pie Chart
    dep_yes = df[(df["deposit"]=='yes')]
    dep_no = df[(df["deposit"]=='no')]
    df_yes = pd.DataFrame(dep_yes[variable_display2].value_counts())

    color_list=['cyan', 'chartreuse', 'gold', 'green', 'fuchsia', 'darksalmon', 'darkmagenta', 'pink', 'turquoise', 'yellow', 'rosybrown', 'olive']

    df_yes.reset_index(inplace=True)
    df_yes.rename(columns={'index':variable_display2, variable_display2:'count'}, inplace = True)
    df_yes['angle'] = df_yes['count'] / df_yes['count'].sum() * 2 * pi
    df_yes['color'] = color_list[0:len(df_yes)]
    df_yes = df_yes.to_dict('list')

    df_no = pd.DataFrame(dep_no[variable_display2].value_counts())

    df_no.reset_index(inplace=True)
    df_no.rename(columns={'index':variable_display2, variable_display2:'count'}, inplace = True)
    df_no['angle'] = df_no['count'] / df_no['count'].sum() * 2 * pi
    df_no['color'] = color_list[0:len(df_no)]
    df_no = df_no.to_dict('list')
    # Plot a pie chart

    source1 = ColumnDataSource(df_yes)
    source2 = ColumnDataSource(df_no)

    p1 = figure(plot_height=300, title="Pie Chart on {} types".format(variable_display2), toolbar_location=None,
           tools="hover", tooltips= '@{}: @count'.format(variable_display2), x_range=(-0.5, 1.0))
    p1.wedge(x=0, y=1, radius=0.3,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field=variable_display2, source=source1)

    p1.axis.axis_label=None
    p1.axis.visible=False
    p1.grid.grid_line_color = None 
    tab1 = Panel(child=p1, title="Dépôt effectué")

    p2 = figure(plot_height=300, title="Pie Chart on {} types".format(variable_display2), toolbar_location=None,
           tools="hover", tooltips='@{}: @count'.format(variable_display2), x_range=(-0.5, 1.0))
    p2.wedge(x=0, y=1, radius=0.3,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field=variable_display2, source=source2)

    p2.axis.axis_label=None
    p2.axis.visible=False
    p2.grid.grid_line_color = None
    tab2 = Panel(child=p2, title="Pas de dépôt effectué")
    
    st.bokeh_chart(Tabs(tabs=[tab1, tab2]))

    #ancien code de test
    # variable_display1 = st.selectbox('selectionnez une variable', ('age','balance', 'duration', 'campaign', 'pdays', 'day'))
    # st.write('votre choix:', variable_display1)

    # fig1 = plt.figure(figsize=(10,5))
    # sns.histplot(x =variable_display1 , bins = 30, data =  df,hue = 'deposit', kde=True, multiple="dodge" );
    # st.pyplot(fig1)

    # variable_display2 = st.selectbox('selectionnez une variable', ('job','marital', 'education', 'default', 'housing', 'loan', 'contact'))
    # st.write('votre choix:', variable_display2)
    # fig2 = plt.figure(figsize=(14,df[variable_display2].unique().shape[0]/2)) 
    # sns.countplot(y = variable_display2 , hue = 'deposit', data = df);
    # st.pyplot(fig2)


# Menu 'Modelisation - Preprocessing - Interprétabilité globale'

elif page==pages[3] : 

    parties = ['Preprocessing', 'Modélisation', 'Interprétabilité globale', 'Conclusion']
    parties2 = st.sidebar.radio("Les différentes parties : ", parties)
    
    if parties2 == parties[0] : 
        st.title("Projet Depositary_Hunt")
        st.write("## Préprocessing - Modélisation - Interprétabilité globale")
        st.markdown('\n\n\n\n')
        st.markdown('Cette section est quant à elle dediée à la visualisation des résultats de nos différents modèles de prédiction ainsi qu\'à leur interprétabilité globale.')
        st.markdown('\n\n\n\n')
        st.markdown('Nous allons dans un premier temps brièvement détailler les différentes étapes qui ont été effectuées pour préparer les données à la modélisation.')
        st.markdown('\n\n\n\n')
        st.write("#### Données clients avant Préprocessing : ")
        st.dataframe(df)
        st.markdown('\n\n\n\n')
        st.write("#### Données clients après Préprocessing : ")
        df.deposit = df.deposit.replace({'yes' : 1, 'no' : 0})
        deposit = df.deposit
        df = df.drop('deposit', axis = 1)
        df = data_prepare2(df)
        st.dataframe(df)
        X_train, X_test, Y_train, Y_test = train_test_split(df, deposit, test_size = 0.2)
    
    if parties2 == parties[1] : 
        # Menu modélisation

        modeles = ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier", "StackingClassifier", 'Comparaison']
        st.sidebar.write("## Modélisation")
        modele_choisi = st.sidebar.selectbox('Je choisis le modèle : ', modeles)

        # Preprocessing des données 
        df = pd.read_csv("bank.csv")
        df.deposit = df.deposit.replace({'yes' : 1, 'no' : 0})
        deposit = df.deposit
        df = df.drop('deposit', axis = 1)
        df = data_prepare2(df)
        X_train, X_test, Y_train, Y_test = train_test_split(df, deposit, test_size = 0.2)
        
        st.write('## Résultats des modèles')
        
        if modele_choisi == modeles[0] :  
            st.write('#### Les résultats du modèle {} sont les suivants : '.format(modele_choisi))
            st.markdown('\n\n\n\n')
            y_pred_log = clf_log.predict(X_test)
            st.text(classification_report(Y_test, y_pred_log))
            st.markdown('\n\n\n\n\n\n\n\n')
            st.write('##### Rappels ')
            st.markdown('\n\n\n\n')
            """
            * Rappel : il s'agit d'un indicateur de détection. Il indique dans quelle mesure une modalité de la variable cible a été correctement détectée par le modèle. Dans notre cas, pour la modalité 1, c'est l'indicateur à choisir si la banque souhaite maximiser le nombre de souscriptions (maximiser le revenu); 
            
            * Précision : il s'agit d'un indicateur de prédiction. Il indique dans quelle mesure une modalité de la variable cible a été correctement prédite par le modèle. Dans notre cas, pour la modalité 1, c’est l’indicateur à choisir si la banque souhaite minimiser le nombre d’appels émis vers des clients non intéressés (limiter les couts); 
            
            * F1_score : il s'agit de la moyenne harmonique des 2 indicateurs précédents, raison pour laquelle nous avons sélectionné cette métrique.
            """
            
        if modele_choisi == modeles[1] : 
            st.write('#### Les résultats du modèle {} sont les suivants : '.format(modele_choisi))
            st.markdown('\n\n\n\n')
            y_pred_rdf = clf_rdf.predict(X_test)
            st.text(classification_report(Y_test, y_pred_rdf))
            st.markdown('\n\n\n\n\n\n\n\n')
            st.write('##### Rappels ')
            st.markdown('\n\n\n\n')
            """
            * Rappel : il s'agit d'un indicateur de détection. Il indique dans quelle mesure une modalité de la variable cible a été correctement détectée par le modèle. Dans notre cas, pour la modalité 1, c'est l'indicateur à choisir si la banque souhaite maximiser le nombre de souscriptions (maximiser le revenu); 
            
            * Précision : il s'agit d'un indicateur de prédiction. Il indique dans quelle mesure une modalité de la variable cible a été correctement prédite par le modèle. Dans notre cas, pour la modalité 1, c’est l’indicateur à choisir si la banque souhaite minimiser le nombre d’appels émis vers des clients non intéressés (limiter les couts); 
            
            * F1_score : il s'agit de la moyenne harmonique des 2 indicateurs précédents, raison pour laquelle nous avons sélectionné cette métrique.
            """
        
        if modele_choisi == modeles[2] : 
            st.write('#### Les résultats du modèle {} sont les suivants : '.format(modele_choisi))
            st.markdown('\n\n\n\n')
            y_pred_grad = clf_grad.predict(X_test)
            st.text(classification_report(Y_test, y_pred_grad))
            st.markdown('\n\n\n\n\n\n\n\n')
            st.write('##### Rappels ')
            st.markdown('\n\n\n\n')
            """
            * Rappel : il s'agit d'un indicateur de détection. Il indique dans quelle mesure une modalité de la variable cible a été correctement détectée par le modèle. Dans notre cas, pour la modalité 1, c'est l'indicateur à choisir si la banque souhaite maximiser le nombre de souscriptions (maximiser le revenu); 
            
            * Précision : il s'agit d'un indicateur de prédiction. Il indique dans quelle mesure une modalité de la variable cible a été correctement prédite par le modèle. Dans notre cas, pour la modalité 1, c’est l’indicateur à choisir si la banque souhaite minimiser le nombre d’appels émis vers des clients non intéressés (limiter les couts); 
            
            * F1_score : il s'agit de la moyenne harmonique des 2 indicateurs précédents, raison pour laquelle nous avons sélectionné cette métrique.
            """
        
        if modele_choisi == modeles[3] : 
            st.write('#### Les résultats du modèle {} sont les suivants : '.format(modele_choisi))
            st.markdown('\n\n\n\n')
            y_pred_sclf = sclf.predict(X_test)
            st.text(classification_report(Y_test, y_pred_sclf))
            st.markdown('\n\n\n\n\n\n\n\n')
            st.write('##### Rappels ')
            st.markdown('\n\n\n\n')
            """
            * Rappel : il s'agit d'un indicateur de détection. Il indique dans quelle mesure une modalité de la variable cible a été correctement détectée par le modèle. Dans notre cas, pour la modalité 1, c'est l'indicateur à choisir si la banque souhaite maximiser le nombre de souscriptions (maximiser le revenu); 
            
            * Précision : il s'agit d'un indicateur de prédiction. Il indique dans quelle mesure une modalité de la variable cible a été correctement prédite par le modèle. Dans notre cas, pour la modalité 1, c’est l’indicateur à choisir si la banque souhaite minimiser le nombre d’appels émis vers des clients non intéressés (limiter les couts); 
            
            * F1_score : il s'agit de la moyenne harmonique des 2 indicateurs précédents, raison pour laquelle nous avons sélectionné cette métrique.
            """
        
        if modele_choisi == modeles[4] : 

            st.markdown('\n\n\n\n')
            st.write('#### Les résultats du modèle LogisticRegression sont les suivants : ')
            y_pred_log = clf_log.predict(X_test)
            st.text(classification_report(Y_test, y_pred_log))
            
            st.markdown('\n\n\n\n')
            st.write('#### Les résultats du modèle RandomForestClassifier sont les suivants : ')
            y_pred_rdf = clf_log.predict(X_test)
            st.text(classification_report(Y_test, y_pred_rdf))

            st.markdown('\n\n\n\n')
            st.write('#### Les résultats du modèle GradientBoostingClassifier sont les suivants : ')
            y_pred_grad = clf_grad.predict(X_test)
            st.text(classification_report(Y_test, y_pred_grad))

            st.markdown('\n\n\n\n')
            st.write('#### Les résultats du modèle StackingClassifier sont les suivants : ')
            y_pred_sclf = sclf.predict(X_test)
            st.text(classification_report(Y_test, y_pred_sclf))

            st.markdown('\n\n\n\n')
            st.markdown('**Le modèle le plus performant est indubitablement le GradientBoostingClassifier**')

    if parties2 == parties[2] :
        # Menu interprétabilité globale
        modeles = ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier", "StackingClassifier", "Comparaison"]
        st.sidebar.write("## Interprétabilité globale")
        modele_choisi = st.sidebar.selectbox('Je choisis le modèle : ', ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier", "StackingClassifier", "Comparaison"], index = 4)
    
        st.write('## Interprétabilité globale des modèles')
        
        # On recommence tout le Preprocessing pour faire l'interprétabilité globale
        # On utilise la fonction data_prepare3 car sans normalisation des données
        df2 = pd.read_csv("bank.csv")
        df2.deposit = df2.deposit.replace({'yes' : 1, 'no' : 0})
        deposit2 = df2.deposit
        df2 = df2.drop('deposit', axis = 1)
        df2 = data_prepare3(df2)

        # Séparation des données en jeu d'entraînement et de test
        from sklearn.model_selection import train_test_split
        X_train2, X_test2, Y_train2, Y_test2 = train_test_split(df2, deposit2, test_size = 0.2)

        # On équilibre la variable cible avec RandomOverSampler
        from imblearn.over_sampling import RandomOverSampler
        ROS = RandomOverSampler()
        # Tuple assignment
        X_train2, Y_train2 = ROS.fit_resample(X_train2, Y_train2)
    
        # Import des librairies essentielles pour interprétabilité avec skater
        from skater.model import InMemoryModel
        from skater.core.explanations import Interpretation

        if modele_choisi == modeles[0] : 

            st.write('#### Interprétabilité globale du modèle {} '.format(modele_choisi), '\n\n')

            # Résultats de la régression logistique
            from sklearn.linear_model import LogisticRegression
            clf_logistic2 = LogisticRegression()
            clf_logistic2.fit(X_train2, Y_train2)
            # On utilise X_train2.values pour éviter code erreur lié au dataframe pandas
            model = InMemoryModel(clf_logistic2.predict_proba, examples = X_train2.values, target_names = ['Non-souscription', 'Souscription'])
            interpreter = Interpretation(X_train2.values, feature_names = X_train2.columns, training_labels = Y_train2)
            plots = interpreter.feature_importance.feature_importance(model, n_jobs = -1, ascending = True, n_samples = 11200)
            # On transforme plots en DataFrame
            plots = pd.DataFrame(plots, columns = ['Valeur du coefficient'])
            # Affichage des 36 variables les plus importantes du modèle
            # plots.tail(36).plot.barh(figsize = (10, 12))
            # On trie le tableau par ordre croissant
            plots = plots.sort_values(['Valeur du coefficient'], ascending = False)

            # On affiche le tableau
            st.markdown('\n\n\n\n')
            st.write('Tableau des différentes variables triées par ordre de signficativité décroissante')
            st.table(plots)

            # On affiche l'image d'interprétabilité globale du modèle
            st.markdown('\n\n\n\n')
            st.write('##### Modèle chargé depuis une image : ')
            from PIL import Image
            image = Image.open('INTERPRETABILITE_GLOBALE_LOG_SKATER.png')
            st.image(image)

            # On créé un graphique à partir du tableau
            # st.title('Schéma de l\'Interprétation globale du modèle RandomForestClassifier avec SKATER\n')
            # st.bar_chart(data = plots)
            st.markdown('\n\n\n\n')
            st.write('##### Modèle directement entraîné : ')
            # Désactive les warnings
            st.set_option('deprecation.showPyplotGlobalUse', False)
            fig1 = plt.figure(figsize = (10, 12))
            ax = plots.tail(36).plot.barh()
            st.pyplot(fig1 = ax)
        
        if modele_choisi == modeles[1] : 

            st.write('#### Interprétabilité globale du modèle {} '.format(modele_choisi), '\n\n')

            # Résultats du RandomForestClassifier
            from sklearn.ensemble import RandomForestClassifier
            clf_rdf2 = RandomForestClassifier()
            clf_rdf2.fit(X_train2, Y_train2)
            # On utilise X_train2.values pour éviter code erreur lié au dataframe pandas
            model = InMemoryModel(clf_rdf2.predict_proba, examples = X_train2.values, target_names = ['Non-souscription', 'Souscription'])
            interpreter = Interpretation(X_train2.values, feature_names = X_train2.columns, training_labels = Y_train2)
            plots = interpreter.feature_importance.feature_importance(model, n_jobs = -1, ascending = True, n_samples = 11200)
            # On transforme plots en DataFrame
            plots = pd.DataFrame(plots, columns = ['Valeur du coefficient'])
            # Affichage des 36 variables les plus importantes du modèle
            # plots.tail(36).plot.barh(figsize = (10, 12))
            # On trie le tableau par ordre croissant
            plots = plots.sort_values(['Valeur du coefficient'], ascending = False)

            # On affiche le tableau
            st.markdown('\n\n\n\n')
            st.write('##### Tableau des différentes variables triées par ordre de signficativité décroissante')
            st.table(plots)

            # On affiche l'image d'interprétabilité globale du modèle
            st.markdown('\n\n\n\n')
            st.write('##### Modèle chargé depuis une image : ')
            from PIL import Image
            image = Image.open('INTERPRETABILITE_GLOBALE_RDF_SKATER.png')
            st.image(image)

            # On créé un graphique à partir du tableau
            # st.title('Schéma de l\'Interprétation globale du modèle RandomForestClassifier avec SKATER\n')
            # st.bar_chart(data = plots)
            st.markdown('\n\n\n\n')
            st.write('##### Modèle directement entraîné : ')
            # Désactive les warnings
            st.set_option('deprecation.showPyplotGlobalUse', False)
            fig1 = plt.figure(figsize = (10, 12))
            ax = plots.tail(36).plot.barh()
            st.pyplot(fig1 = ax)

        if modele_choisi == modeles[2] : 

            st.write('#### Interprétabilité globale du modèle {} '.format(modele_choisi), '\n\n')

            # Résultats du GradientBoostingClassifier
            from sklearn.ensemble import GradientBoostingClassifier
            clf_grad2 = GradientBoostingClassifier()
            clf_grad2.fit(X_train2, Y_train2)
            # On utilise X_train2.values pour éviter code erreur lié au dataframe pandas
            model = InMemoryModel(clf_grad2.predict_proba, examples = X_train2.values, target_names = ['Non-souscription', 'Souscription'])
            interpreter = Interpretation(X_train2.values, feature_names = X_train2.columns, training_labels = Y_train2)
            plots = interpreter.feature_importance.feature_importance(model, n_jobs = -1, ascending = True, n_samples = 11200)
            # On transforme plots en DataFrame
            plots = pd.DataFrame(plots, columns = ['Valeur du coefficient'])
            # Affichage des 36 variables les plus importantes du modèle
            # plots.tail(36).plot.barh(figsize = (10, 12))
            # On trie le tableau par ordre croissant
            plots = plots.sort_values(['Valeur du coefficient'], ascending = False)

            # On affiche le tableau
            st.markdown('\n\n\n\n')
            st.write('##### Tableau des différentes variables triées par ordre de signficativité décroissante')
            st.table(plots)

            # On affiche l'image d'interprétabilité globale du modèle
            st.markdown('\n\n\n\n')
            st.write('##### Modèle chargé depuis une image : ')
            from PIL import Image
            image = Image.open('INTERPRETABILITE_GLOBALE_GRAD_SKATER.png')
            st.image(image)

            # On créé un graphique à partir du tableau
            # st.title('Schéma de l\'Interprétation globale du modèle RandomForestClassifier avec SKATER\n')
            # st.bar_chart(data = plots)
            st.markdown('\n\n\n\n')
            st.write('##### Modèle directement entraîné : ')
            # Désactive les warnings
            st.set_option('deprecation.showPyplotGlobalUse', False)
            fig1 = plt.figure(figsize = (10, 12))
            ax = plots.tail(36).plot.barh()
            st.pyplot(fig1 = ax)

        if modele_choisi == modeles[3] :

            st.write('#### Interprétabilité globale du modèle {} '.format(modele_choisi), '\n\n')

            # Résultats du StackingClassifier
            from sklearn.ensemble import StackingClassifier
            # Les classifieurs ont déjà tous été instanciés précédemment
            sclf2 = StackingClassifier(estimators = [('LOGISTIC', clf_logistic2), ('RANDOM_FOREST', clf_rdf2), ('GRADIENT_BOOSTING', clf_grad2)], final_estimator = clf_rdf2)
            # Entraînement du modèle StackingClassifier
            sclf2.fit(X_train2, Y_train2)
            # On utilise X_train2.values pour éviter code erreur lié au dataframe pandas
            model = InMemoryModel(sclf2.predict_proba, examples = X_train2.values, target_names = ['Non-souscription', 'Souscription'])
            interpreter = Interpretation(X_train2.values, feature_names = X_train2.columns, training_labels = Y_train2)
            plots = interpreter.feature_importance.feature_importance(model, n_jobs = -1, ascending = True, n_samples = 11200)
            # On transforme plots en DataFrame
            plots = pd.DataFrame(plots, columns = ['Valeur du coefficient'])
            # Affichage des 36 variables les plus importantes du modèle
            # plots.tail(36).plot.barh(figsize = (10, 12))
            # On trie le tableau par ordre croissant
            plots = plots.sort_values(['Valeur du coefficient'], ascending = False)

            # On affiche le tableau
            st.markdown('\n\n\n\n')
            st.write('##### Tableau des différentes variables triées par ordre de signficativité décroissante')
            st.table(plots)

            # On affiche l'image d'interprétabilité globale du modèle
            st.markdown('\n\n\n\n')
            st.write('##### Modèle chargé depuis une image : ')
            from PIL import Image
            image = Image.open('INTERPRETABILITE_GLOBALE_SCLF_SKATER.png')
            st.image(image)

            # On créé un graphique à partir du tableau
            # st.title('Schéma de l\'Interprétation globale du modèle RandomForestClassifier avec SKATER\n')
            # st.bar_chart(data = plots)
            st.markdown('\n\n\n\n')
            st.write('##### Modèle directement entraîné : ')
            # Désactive les warnings
            st.set_option('deprecation.showPyplotGlobalUse', False)
            fig1 = plt.figure(figsize = (10, 12))
            ax = plots.tail(36).plot.barh()
            st.pyplot(fig1 = ax)

        if modele_choisi == modeles[4] :
            st.write('#### Interprétabilité globale des différents modèles')
            st.markdown('\n\n\n\n')
            st.write('##### Modèles chargés depuis une image : ')
            from PIL import Image
            image = Image.open('INTERPRETABILITE_GLOBALE_LOG_SKATER.png')
            st.image(image)
            image2 = Image.open('INTERPRETABILITE_GLOBALE_RDF_SKATER.png')
            st.image(image2)
            image3 = Image.open('INTERPRETABILITE_GLOBALE_GRAD_SKATER.png')
            st.image(image3)
            image4 = Image.open('INTERPRETABILITE_GLOBALE_SCLF_SKATER.png')
            st.image(image4)

            st.markdown('\n\n\n\n')
            st.markdown('**En conclusion, globalement, nous pouvons classer nos 31 variables dans les 3 catégories suivantes selon nos 4 modèles ci-dessus :** ')
            st.markdown('\n\n')
            st.markdown('- **Les variables considérées comme peu importantes :** « job_admin », « job_housemaid », « job_selfemployed », « job_entrepreneur », « job_unemployed », « job_student », « job_technician », « job_services » et « default »;')
            st.markdown('\n\n')
            st.markdown('- **Celles considérées comme peu importantes par certains modèles et relativement importantes pour d’autres :** « education_primary », « education_secondary », « job_management », « marital_divorced », « job_retired », « poutcome_failure » et « contact_telephone »;')
            st.markdown('\n\n')
            st.markdown('- **Celles considérées comme importantes :** « previous », « housing », « campaign », « contact_cellular », « age », « month », « balance », « marital_married », « marital_single », « poutcome_success », « pdays », « loan », « day », « job_blue-collar » et « education_tertiary ».')
            st.markdown('\n\n')
            st.markdown('**_Les variables considérées comme signficativement positives dans le modèle vont nous être utiles non seulement pour savoir quel profil de client cibler lors des prochaines campagnes markettings, mais également pour savoir comment les contacter. Cela permettra d\'améliorer fortement leur efficacité._**')
            st.markdown('\n\n\n\n')

    if parties2 == parties[3] :
        st.write('## Conclusion')
        st.markdown('\n\n\n\n')
        st.write('#### Réponse à la question de savoir : _comment améliorer l’efficacité des prochaines campagnes marketing ?_')
        st.markdown('\n\n\n\n')
        """
        * Améliorer la qualité du jeu de données : 

          - Apporter davantage d'informations pertinentes sur les clients;

          - Précisions concernant l'année des campagnes précédentes;

          - Précisions sur les modalités non expliquées de certaines variables (« other » pour « poutcome » et « unknown » pour « job », « education », « contact » et « poutcome »);
        
        * En l’état actuel de nos données, il faut : 

          - Augmenter le nombre d'appels émis vers des clients par téléphone mobile (« contact_cellular », « campaign » et « previous ») vers des clients présentant les caractéristiques considérées comme importantes dans l’interprétation globale de nos 4 modèles de classification précédents, à savoir :

            + Les clients célibataires; 
            + Ceux ayant effectué des études supérieures;
            + Ceux ayant un solde de compte positif;  
            + Ceux ne présentant pas de crédit immobilier ni de consommation;
            + Ceux ayant déjà été contactés lors des campagnes précédentes;
            + Parmis ces derniers, ceux ayant déjà souscrit au contrat de dépôt à terme;
            + Et enfin, ceux compris dans la tranche d’âge suivante : <30 ans et >60 ans.
        """
        st.markdown('\n\n\n\n')
        st.markdown('**L\'interprétabilité globale nous permet de dresser un profil type de clients à contacter pour améliorer l\'efficacité des prochaines campagnes markettings.**')
        st.markdown('\n\n\n\n')

# menu Dashboard client
elif page==pages[4]:
    st.title("Projet Depositary_Hunt")
    st.write("## Dashboard client")
    model = st.sidebar.selectbox("choix du modèle", ["GradientBoostingClassifier","LogisticRegression", "RandomForestClassifier",  "StackingClassifier"], index=0)
    modes_entree = ['Entrée manuelle','Base de données clients','Importer un fichier', ]
    mode_entree = st.sidebar.radio("Mode d\'entrée des données:", modes_entree)

# Sous-menu Dashboard client /Entrée manuelle
    if mode_entree==modes_entree[0]:
        st.write("### Entrée manuelle")

        cols = st.columns((1, 1, 1, 1))

        age = cols[0].number_input("Age du client", min_value=18,max_value=95 ,value=30)
        job = cols[1].selectbox("Categorie pro:", ['management', 'blue-collar', 'technician', 'admin.', 'services', 'retired', 'self-employed', 'student',
        'unemployed', 'entrepreneur', 'housemaid', 'unknown'], index=11)
        education = cols[2].selectbox("Niveau d'étude:", ['secondary', 'tertiary', 'primary', 'unknown'], index=3)
        marital = cols[3].selectbox("Status marital:", ['married', 'single', 'divorced'], index=1)
        
        loan = cols[0].checkbox("Credit consommation", value=False, help="Cocher si le client a un crédit à la consommation")
        housing = cols[0].checkbox("Credit immobilier", value=False, help="Cocher si le client a un crédit immobilier")
        balance = cols[1].number_input("Disponibilités financières:",step =10, help = 'Entrer le solde moyen du client')
        default = cols[2].checkbox("Défaut de credit", value=False, help="Cocher si le client a fait défaut sur un credit")

        cols = st.columns((1, 1, 2))
        date = cols[0].date_input("Date du dernier contact:")
        contact = cols[1].selectbox("Moyen de contact:", ['cellular', 'telephone', 'unknown'], index=2)
        campaign = cols[2].slider("Nombre d'appels:",min_value=1, max_value=70, value=1)

        pcampaign = st.checkbox("Campagne précédente", help="Cocher si le client a été contacté pour la campagne précédente")

        if pcampaign:
            st.write("Informations relatives à la campagne précédente")
            cols = st.columns((1, 1, 2))
            poutcome = cols[0].selectbox("Resultat campagne:", ['unknown', 'failure', 'success', 'other'], index=0)
            pdays = cols[1].number_input("Nombre de jours:",step =10, help ='"Nombre de jours depuis la campagne precedente:' )       
            previous = cols[2].slider("Nombre d'appels:",min_value=1, max_value=70, value=1, help ='Nombre d\'appels lors de la campagne précédente:')
        else:
            poutcome = 'unknown'
            pdays = 999
            previous = 0

        with st.form(key="my_form"):
            submit_button = st.form_submit_button(label="Calcul de l'oportunité d'appel")
            if not submit_button:
                st.stop()
            #creation du dataframe client
            day = int(date.day)
            month = int(date.month)
            
            # On récupère toutes les données saisies dans client_param
            client_param = [[age, job, marital, education,default, balance, housing, loan, 
                        contact, day, month, '0',campaign, pdays, previous, poutcome  ]]
            
            # On créé un DataFrame des données saisies
            client_df = pd.DataFrame(client_param, columns =['age', 'job', 'marital', 'education', 
                        'default', 'balance', 'housing', 'loan', 'contact','day','month',
                        'duration','campaign','pdays','previous','poutcome' ])
            
            st.markdown("### données client")
            st.dataframe(client_df)

            # preparation des données client
            client_df = data_prepare(client_df)

            # evaluation du client
            pred, _ = predict2(client_df)

       
            #st.write(pred[0])
            if pred[0] ==1:
                st.write('### Selon le modèle {}, Il est recommandé d\'appeler ce client'.format(model))
            else:
                st.write('### Selon le modèle {}, Il  n\'est pas utile d\'appeler ce client'.format(model))

        # interpretabilité
            if model == "RandomForestClassifier" :
                st.write('#### Pas d\'interprétabilité disponible pour ce modèle') 
            elif model == "LogisticRegression" :
                explainer = shap.Explainer(clf_log_shap, df_shap)
                shap_values = explainer(client_df)
                st.write(shap_values[0].shape)
                st_shap(shap.plots.waterfall(shap_values[0]), height=400)  
            elif model == "GradientBoostingClassifier" :
                explainer = shap.Explainer(clf_grad_shap, df_shap)
                shap_values = explainer(client_df)
                st_shap(shap.plots.waterfall(shap_values[0]), height=400)  
            else:
                st.write('#### Pas d\'interprétabilité disponible pour ce modèle') 


# Sous-menu Dashboard client /Base de donnée client

    elif mode_entree==modes_entree[1]:
        st.write("### Base de données client")

        id_client = st.number_input("Identifiant du client", min_value=0,max_value=1000 ,value=0)
        
        with st.form(key="my_form"):
            submit_button = st.form_submit_button(label="Analyse des données du client")
            if not submit_button:
                st.stop()
            st.markdown("### données client")
            client_df = df.iloc[[id_client]]
            client_df = client_df.drop('deposit', axis = 1)
            st.dataframe(client_df)

            # preparation des données client
            client_df = data_prepare(client_df)

            # evaluation du client
            pred, _ = predict2(client_df)

            
            #st.write(pred[0])
            if pred[0] ==1:
                st.write('### Selon le modèle {}, Il est recommandé d\'appeler ce client'.format(model))
            else:
                st.write('### Selon le modèle {}, Il  n\'est pas utile d\'appeler ce client'.format(model))

        # interpretabilité
            if model == "RandomForestClassifier" :
                st.write('#### Pas d\'interprétabilité disponible pour ce modèle') 
            elif model == "LogisticRegression" :
                explainer = shap.Explainer(clf_log_shap, df_shap)
                shap_values = explainer(client_df)
                st.write(shap_values[0].shape)
                st_shap(shap.plots.waterfall(shap_values[0]), height=400)  
            elif model == "GradientBoostingClassifier" :
                explainer = shap.Explainer(clf_grad_shap, df_shap)
                shap_values = explainer(client_df)
                st_shap(shap.plots.waterfall(shap_values[0]), height=400)  
            else:
                st.write('#### Pas d\'interprétabilité disponible pour ce modèle')


# Sous-menu Dashboard client /A partir d'un fichier
    elif mode_entree==modes_entree[2]:
        st.write("### A partir d'un fichier")
        st.write("Veuiller choisir un fichier de clients au format csv. Ce fichier doit suivre la structure standard. En cas de doute veuillez télécharger le fichier d'exemple disponible ci-dessous. Votre fichier peut contenir plusieurs clients")
        uploaded_file = st.file_uploader("Upload CSV", type=".csv", key="1",help="Télécharger un fichier de clients")
        if uploaded_file is not None:
            file_container = st.expander("Verifiez votre fichier .csv")
            shows = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            file_container.write(shows)

            client_df = uploaded_file

            with st.form(key="my_form"):
                cols = st.columns((1, 1, 1))
                submitted = cols[0].form_submit_button(label="Calcule de l'opportunité d'appel")
                sort_result  = cols[1].checkbox("Trier par priorité", value=False, help="Cocher si vous voulez trier les clients par priorité d'appel")
                filter_result  = cols[2].checkbox("Masquer les clients non retenus ", value=False, help="Cocher pour n'afficher que les clients à appeler")
                if not submitted:
                    st.stop()
                
                #creation du dataframe de sortie
                
                loaded_data = shows
                client_df = data_prepare(loaded_data)
                pred , pred_proba = predict2(client_df)
                
                pred = pd.DataFrame(pred, columns = ['Call_Opportunity'])
                pred_proba = pd.DataFrame(pred_proba, columns = ['Opportunity_score'])
                result = pd.concat([loaded_data, pred, pred_proba ], axis =1)
                result_sorted  = result.sort_values(by = 'Opportunity_score', ascending=False,)

                # visualisation des resultats

                def highlight_opportulities(s):
                    return ['background-color: lightgreen']*len(s) if s.Call_Opportunity else ['background-color: lightcoral']*len(s)
  
                st.write('### Le modèle {}, recommande d\'appeler les clients en Vert '.format(model))
                if sort_result:
                    if filter_result:
                        result_choosen = result_sorted[result_sorted.Call_Opportunity ==1 ]
                        st.dataframe(result_sorted[result_sorted.Call_Opportunity ==1 ].style.apply(highlight_opportulities, axis=1))
                    else:
                        st.dataframe(result_sorted.style.apply(highlight_opportulities, axis=1))
                        result_choosen = result_sorted
                else:
                    if filter_result:
                        st.dataframe(result[result.Call_Opportunity ==1 ].style.apply(highlight_opportulities, axis=1))
                        result_choosen = result[result.Call_Opportunity ==1 ]
                    else:
                        st.dataframe(result.style.apply(highlight_opportulities, axis=1))
                        result_choosen = result


            exp = result_choosen.to_csv().encode('utf-8')
            st.download_button(label="Télécharger le résultat",data=exp, file_name='Result.csv', mime='text/csv')

        else:

            def convert_df(df):
                return df.to_csv().encode('utf-8')
            csv = convert_df(template)
            st.download_button(label="telecharger un fichier d'exemple",data=csv, file_name='fichier_exemple.csv', mime='text/csv')
           
            st.stop()

# menu Remerciements
elif page==pages[5]:
    st.title("Projet Depositary_Hunt")
    st.write("## Remerciements")
    st.markdown("")
    st.markdown("")
    st.markdown("Nous remercions tout particulièrement Lara pour son aide et ses conseils dans la réalisation de ce projet.")
    st.markdown("Merci également à Frédéric pour son coaching lors de cette formation.")
    st.markdown("Enfin, un grand merci également à toute l'équipe DataScientest pour votre support et à nos professeurs de Master Class")
    st.markdown("")
    st.markdown("")
    if st.button('Musique maestro !'):
        st.video("https://youtu.be/ETxmCCsMoD0", start_time= 44)
