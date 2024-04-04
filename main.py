from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import List, Optional
from passlib.context import CryptContext
import pandas as pd
import numpy as np
from pydantic import BaseModel


api = FastAPI(
    title="Génération de QCM",
    description="API qui permet de générée des QCM de 5, 10 ou questions en fonction du cas d'usage et des sujets sélectionnés.",
    version="1.0",
    openapi_tags=[
    {
        'name': 'Vérification API',
        'description': 'Fonction qui permet de vérifier que l\'API est fonctionnelle'
    },
        {
        'name': 'Génération de QCM',
        'description': 'Fonction qui permet la génération de QCM aléatoire en fonction du cas d\'usage et du sujet et du nombres de questions sélectionné'
    },
        {
        'name': 'Création de question',
        'description': 'Fonction qui permet la création de question'
    }
    ]
    )

security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

users = {

    "alice": {
        "username": "alice",
        "name": "alice",
        "hashed_password": pwd_context.hash('wonderland'),
    },

    "bob" : {
        "username" :  "bob",
        "name" : "bob",
        "hashed_password" : pwd_context.hash('builder'),
    },

    "clementine": {
        "username": "clementine",
        "name": "clementine",
        "hashed_password": pwd_context.hash('mandarine'),
    },

    "david": {
        "username": "david",
        "name": "david",
        "hashed_password": pwd_context.hash('4dm1N'),
    }
}

# fonction formulaire d'authentification et de vérification d'identifiants
def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    if not(users.get(username)) or not(pwd_context.verify(credentials.password, users[username]['hashed_password'])):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# Enregistrement des questions issue de questions.csv dans une base de données
questions_db = pd.read_csv('questions.csv')


@api.get("/", name="Vérification que l\'api est fonctionelle", tags=['Vérification API'])
async def read_root(current_user: str = Depends(get_current_user)):
    """
    Description: 
    Ce endpoint permet de vérifier que l'API est fonctionnelle

    Args:
    Aucun agument requis

    Returns:
    - str : le message "l'API est fonctionnelle"

    Raises:
    Aucune exception n'est levée
    """
    return {"message": "l'API est fonctionnelle"}

# Modèle pour renvoyer un QCM aléatoire en fonction de use, subjects et du nombre de questions désirées
class Item(BaseModel):
    use: str
    subjects: List[str]
    question_count: int

# Modèle de question
class Question(BaseModel):
    question: str
    subject: str
    correct: List[str]
    use: str
    responseA: str
    responseB: str
    responseC: str
    responseD: Optional[str]


@api.post("/questions/random/item", name="Génération QCM aléatoire", tags=['Génération de QCM'])
async def read_random_questions(item: Item, current_user: str = Depends(get_current_user)):
    """
    Description: 
    Ce endpoint permet de renvoyer un QCM aléatoire en fonction de use, subjects et du nombre de questions désirées

    Args:
    - item : Item : un objet de type Item qui contient les informations suivantes:
        - use : str : le cas d'usage
        - subjects : List[str] : la liste des sujets (1 ou plusieurs sujets)
        - question_count : int : le nombre de questions désirées (5,10 ou 20)   

    Returns:
    - str : le QCM aléatoire en fonction des paramètres passés

    Raises:
    - HTTPException : si le nombre de questions n'est pas 5, 10 ou 20
    - HTTPException : si le use n'existe pas dans la base de données
    - HTTPException : si un ou plusieurs sujets n'existent pas dans la base de données

    Example:
    - L'input suivant renvoie un QCM aléatoire de 10 questions pour le cas d'usage "Test de validation" et les sujets "Classification", "Systèmes distribués" "Sytèmes distribués", "Automation", "Streaming de données"
    - {
        "use": "Test de validation",
        "subjects": [
            "Classification", "Systèmes distribués", "Sytèmes distribués", "Automation", "Streaming de données"
        ],
        "question_count": 10
        }
    """
    # Récupération des différents use de la base de données questions_db et les mettre dans une liste, pour vérifier si le use sélectionné existe
    possible_use = questions_db['use'].unique()
    # Récupération des différents subjects en fonction du use sélectionné et les mettre dans une liste, pour vérifier si les subjects sélectionnés existent
    possible_subjects = questions_db[questions_db['use'] == item.use]['subject'].unique()
    # Validation du nombre de questions 5, 10 ou 20 comme indiqué dans l'évaluation
    if item.question_count not in [5, 10, 20]:
        raise HTTPException(status_code=400, detail="Le nombre de questions n'est pas valide. Il doit être de 5, 10 ou 20.")
    # Vérification du use pour s'assuer qu'il existe dans la base de données
    # Si ce n'est pas le cas on renvoie la liste des use possibles
    if item.use not in possible_use:
        raise HTTPException(status_code=400, detail="Cas d'usage non valide. Doit être l'un des éléments suivants : " + ', '.join(possible_use))
    # Vérification des subjects du use sélectionné pour s'assurer qu'ils existent dans la base de données
    # Si ce n'est pas le cas on renvoie la liste des subjects possibles en fonction du use sélectionné
    if not all(subject in possible_subjects for subject in item.subjects):
        raise HTTPException(status_code=400, detail="Sujet(s) non valide. Doit être l'un des éléments suivants : " + ', '.join(possible_subjects))  
        # Filtrage des questions en fonction du use et des subjects sélectionnés
    filtered_questions = questions_db[(questions_db['use'] == item.use) & (questions_db['subject'].isin(item.subjects))]
    # Si le filtre renvoie moins de questions que le nombre demandé, on renvoie toutes les questions filtrées
    if len(filtered_questions) < item.question_count:
        df = filtered_questions
    else:
        # Si le filtre renvoie plus de questions que le nombre demandé, on renvoie un échantillon aléatoire de la taille du nombre demandé
        df = filtered_questions.sample(n=item.question_count)
    # Remplacement des float out of range par None
    df = df.applymap(lambda x: None if isinstance(x, float) and (np.isnan(x) or np.isinf(x)) else x)
    return df.to_dict(orient='records')


@api.post("/questions/create", name="Création d'une nouvelle question", tags=['Création de question'])
async def create_question(question: Question, current_user: str = Depends(get_current_user)):
    """
    Description: 
    Ce endpoint permet de créer une nouvelle question

    Args:
    - question : Question : un objet de type Question qui contient les informations suivantes:
        - question : str : la question
        - subject : str : le sujet
        - correct : List[str] : la liste des réponses correctes
        - use : str : le cas d'usage
        - responseA : str : la réponse A
        - responseB : str : la réponse B
        - responseC : str : la réponse C
        - responseD : Optional[str] : la réponse D (optionnel)

    Returns:
    - str : la liste compléte des questions avec la nouvelle questions à la fin

    Raises:
    - HTTPException : si la question existe déjà (même question et même sujet)

    Example:
    - L'input suivant crée une nouvelle question
    - {
        "question": "Quel est le premier président de la cinquième république",
        "subject": "Histoire",
        "correct": [
            "A"
        ],
        "use": "Test de culture générale",
        "responseA": "Charles De Gaulle",
        "responseB": "François Mitterand",
        "responseC": "René Coty",
        "responseD": "Raymond Poincaré"
        }
    """
    if current_user != "david":
        raise HTTPException(status_code=400, detail="Unauthorized")
    global questions_db  
    # Convertion de Question en un dictionnaire
    question_dict = question.dict()
    # Vérifiation si la question existe déjà (même question et même sujet)
    # On renvoie une erreur si la question existe déjà
    if not questions_db[(questions_db['question'] == question.question) & (questions_db['subject'] == question.subject)].empty:
        raise HTTPException(status_code=400, detail="La question existe déjà")
    # Ajoiut de la nouvelle question à la base de données questios_db
    questions_db.loc[len(questions_db)] = question_dict
    # Remplacement des float out of range par None
    questions_db = questions_db.applymap(lambda x: None if isinstance(x, float) and (np.isnan(x) or np.isinf(x)) else x)
    # Renvoi de la liste compléte des questions avec la nouvelle questions à la fin
    return questions_db.to_dict(orient='records')