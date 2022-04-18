******************************************************************************
******************************************************************************
******************************************************************************
                           # Depositary_Hunt
******************************************************************************
******************************************************************************
******************************************************************************

# CONSTITUTION DU REPO

Ce repo est principalement constitué de 5 fichiers ou dossiers :

- Le fichier "ANALYSE_PREPROCESSING_MACHINE_LEARNING_bank-Copy1.ipynb" : 
  Le code de ce fichier a été transformé en texte brut sauf pour la dernière 
  partie interprétabilité afin qu'elle soit exécutée plus rapidement et 
  indépendamment des autres ;

- Le fichier "ANALYSE_PREPROCESSING_MACHINE_LEARNING_bank.ipynb", quant à lui, 
  contient tout le code du projet à exécuter entièrement ;

- Le fichier 'bank.csv' qui n'est autre que la base de données clients ;

- Le dossier streamlit que nous avons élaboré afin de réaliser une 
  présentation intéractive de notre projet; 

- Le rapport final du projet intitulé 'Rapport_projet_DEPOSITARY_HUNT'. 

******************************************************************************

# PRESENTATION DU PROJET

# Prédiction du succès d’une campagne de Marketing d’une banque

L’analyse des données marketing appliquée aux entreprises de service est une 
problématique très classique des sciences de données. Ici, nous disposons d’un 
jeu de données comprenant des informations relatives aux clients d’une banque 
et ayant pour finalité la souscription à un produit que l’on appelle 
« dépôt à terme ». Lorsqu’un client souscrit à ce produit, il place une 
quantité d’argent dans un compte spécifique et ne sera pas autorisé à disposer 
de ces fonds avant l’expiration du délai fixé. En échange, le client reçoit 
des intérêts de la part de la banque à la fin de ce délai. 

L’objectif de ce projet est donc assez classique : il s’agit d’améliorer 
l’efficacité des prochaines campagnes marketing. Pour cela, il faut augmenter 
le ratio entre le nombre de clients appelé et le nombre de souscription au 
produit « dépôt à terme ».

Dans un premier temps, nous effectuerons une analyse visuelle et statistique 
des facteurs permettant d’expliquer le lien entre les données personnelles du 
client (âge, statut marital, quantité d’argent placé en banque, nombre de fois 
que le client a été contacté, etc.) et la variable cible « deposit » qui répond 
à la question de savoir si le client a souscrit au dépôt à terme. Ensuite, nous 
définirons plusieurs modèles de classification permettant prédire la 
souscription ou non des clients. Par ailleurs, dans la mesure où l’industrie 
Financière est traditionnellement réfractaire aux modèles complexes, considérés 
comme des « boites noires », il sera également nécessaire de fournir une 
explication simple du choix de chaque client à contacter par le biais d’une 
analyse de l’interprétabilité du modèle. Effectivement, dans la mesure où les 
modèles de classification sont généralement utilisés dans le monde bancaire pour 
distinguer les « bons clients » des « mauvais clients » et leur octroyer ou non 
des prêts, les banques se servent de l’interprétabilité des modèles qu’elles 
utilisent habituellement pour justifier aux clients les motifs du refus de leurs 
emprunts.

******************************************************************************

Les données provienent de Kaggle et sont disponibles à l'adresse suivante:
https://www.kaggle.com/janiobachmann/bank-marketing-dataset

******************************************************************************
