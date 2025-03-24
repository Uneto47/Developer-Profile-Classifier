# Developer Profile Classifier

Este repositório classifica desenvolvedores do GitHub com base nas linguagens de programação que utilizam. O código coleta dados dos repositórios públicos, aplica aprendizado de máquina (RandomForest) e gera um arquivo CSV com os perfis classificados em categorias como **Data Science**, **Backend**, **Frontend**, entre outros.

## Funcionalidades

- Coleta dados de contribuintes de um repositório do GitHub.
- Identifica as linguagens de programação utilizadas pelos desenvolvedores em seus repositórios.
- Classifica os desenvolvedores em perfis como **Data Science**, **Backend**, **Frontend**, **Mobile**, entre outros.
- Utiliza o algoritmo **RandomForestClassifier** para prever os perfis com base nas linguagens.
- Gera um arquivo CSV com o perfil classificado de cada desenvolvedor.

## Requisitos

- Python 3.x
- Bibliotecas:
  - `requests`
  - `pandas`
  - `scikit-learn`
  - `numpy`
