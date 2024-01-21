**Model de prédiction de classe de cancer**

Les modèles d’apprentissage offrent un éventail d’application. Cette analyse comparera trois algorithmes de classification pour la prédiction de sous-classes de cancer tel que : random forest, régression logistique et le réseau neuronal. En plus des étapes de visualisation des données, des étapes d’optimisation pourront être ajouté. L’évaluation des modèles se basera sur plusieurs scores et les courbes d’apprentissage résultantes.

**Données**
https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq

Données d’expressions de gènes issus des patients.

**Environnement**
Python = 3.9

Les packages en plus et leurs intallation si ce n'est pas encore fait:

  conda install -c conda-forge pandas
  
  conda install -c conda-forge matplotlib
  
  conda install -c conda-forge seaborn
  
  conda install -c conda-forge scikit-learn

Assurer vous d'avoir **Tkinter** si vous executer le code à partir d'un terminal. 

**Execution**
python3 main.py

**Résultats**
Les plots pour le PCA(analyse de composante principale), le tableau des statistiques descriptives et les courbes d'apprentissages seront dans le répertoire "output"
