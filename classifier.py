import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from time import sleep

REPO = "pandas-dev/pandas"
TOKEN = "seu_token_aqui"

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {TOKEN}"
}

TECH_TO_PROFILE = {
    "Python": "Data Science",
    "Jupyter Notebook": "Data Science",
    "Matlab": "Data Science",
    "MATLAB": "Data Science",
    "Mathematica": "Data Science",
    "Terra": "Data Science",
    "Julia": "Data Science",
    "F#": "Data Science",
    "R": "Data Science",
    "SQL": "Backend",
    "Scala": "Backend",
    "Java": "Backend",
    "C++": "Backend",
    "Ruby": "Backend",
    "Rust": "Backend",
    "Lua": "Backend",
    "Haskell": "Backend",
    "HTML": "Frontend",
    "CSS": "Frontend",
    "TypeScript": "Frontend",
    "JavaScript": "Frontend",
    "CoffeeScript": "Frontend",
    "React": "Frontend",
    "SCSS": "Frontend",
    "Swift": "Mobile",
    "Kotlin": "Mobile",
    "Dart": "Mobile",
    "HCL": "DevOps",
    "Shell": "DevOps",
    "Dockerfile": "DevOps",
    "YAML": "DevOps",
    "Nix": "DevOps",
    "Batchfile": "DevOps",
    "Emacs Lisp": "DevOps",
    "CMake": "DevOps",
    "Meson": "DevOps",
    "Vim Script": "DevOps",
    "VimL": "DevOps"
}

def get_contributors():
    url = f"https://api.github.com/repos/{REPO}/contributors?page=0&per_page=100"
    response = requests.get(url, headers=HEADERS)
    print("Carregou contribuidores")
    return response.json() if response.status_code == 200 else []

def get_languages(user):
    url = f"https://api.github.com/users/{user}/repos"
    response = requests.get(url, headers=HEADERS)
    languages = set()

    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            if "language" in repo and repo["language"]:
                languages.add(repo["language"])
    print(f"Carregou linguagens para {user}")
    return list(languages)

def classify_profile(languages):
    profiles = [TECH_TO_PROFILE.get(lang, "Desconhecido") for lang in languages]
    profiles = [p for p in profiles if p != "Desconhecido"]
    return max(set(profiles), key=profiles.count) if profiles else "Desconhecido"

contributors = get_contributors()

devs_data = []
for contributor in contributors[:100]: 
    user = contributor["login"]
    languages = get_languages(user)
    profile = classify_profile(languages)
    print("Classificou perfil")
    devs_data.append({"login": user, "languages": languages, "profile": profile})
    sleep(1)

df = pd.DataFrame(devs_data)

if 'languages' in df.columns:
    languages_all = list(set([lang for langs in df["languages"] for lang in langs]))
    for lang in languages_all:
        df[lang] = df["languages"].apply(lambda langs: 1 if lang in langs else 0)
else:
    print("Erro: A coluna 'languages' não foi encontrada.")

le = LabelEncoder()
df["profile_encoded"] = le.fit_transform(df["profile"])

X = df[languages_all] if 'languages' in df.columns else pd.DataFrame()
y = df["profile_encoded"]

if not X.empty and not y.empty:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy:.2f}")
else:
    print("Erro: Dados insuficientes para treinar o modelo.")

df["profile_decoded"] = le.inverse_transform(df["profile_encoded"])

print(df[["login", "languages", "profile_decoded"]])

df.to_csv("developers_profiles.csv", index=False)

print("Arquivo CSV gerado: developers_profiles.csv")