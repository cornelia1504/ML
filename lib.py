import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

import time
import os
def load_data(rna_file, label_file):
    rna_data = pd.read_csv(rna_file)
    label_data = pd.read_csv(label_file)
    merged_data = pd.merge(rna_data, label_data, on='Unnamed: 0')
    return merged_data

def plot_pca(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis')
    plt.title('PCA des données RNA-Seq')
    print('**')
    plt.savefig('output/PCA.png')  # Enregistrement du plot PCA
    print('**')
    plt.show()

def descriptive_statistics(X):
    desc_stats = X.describe()
    desc_stats.to_csv('output/descriptive_statistics.csv', index=False)
    return desc_stats

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print("Matrice de confusion:\n", conf_matrix)
    print("\nRapport de classification:\n", class_report)

def cross_validate(model, X, y, cv=5):
    cv_scores = cross_val_score(model, X, y, cv=cv)
    print("Scores de validation croisée:", cv_scores)
    print("Moyenne des scores:", np.mean(cv_scores))

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Taille de l'ensemble d'entraînement")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Train")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation")

    plt.legend(loc="best")
    return plt

def optimize_parameters(model, param_grid, X, y, cv=5):
    grid_search = GridSearchCV(model, param_grid, cv=cv)
    grid_search.fit(X, y)
    return grid_search.best_params_

#####Random forest#####
#######################
def random_forest_process(rna_file, label_file):
    start_time = time.time()
    # Chargement des données
    data = load_data(rna_file, label_file)

    # Séparation des features (X) et de la variable cible (y)
    X = data.drop(['Unnamed: 0', 'Class'], axis=1)
    y = data['Class']

    # PCA
    plot_pca(X, y)

    # Statistiques descriptives
    desc_stats = descriptive_statistics(X)
    print(desc_stats)


    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modèle Random Forest
    param_grid_rf = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    best_params_rf = optimize_parameters(RandomForestClassifier(random_state=42), param_grid_rf, X, y)
    print("Meilleurs paramètres pour RandomForest:", best_params_rf)
    rf_model = train_random_forest(X_train, y_train, n_estimators=best_params_rf['n_estimators'], random_state=42)

    # Évaluation du modèle
    evaluate_model(rf_model, X_test, y_test)

    # Cross-validation
    cross_validate(rf_model, X, y)

    # Learning curve
    learning_curve_plot = plot_learning_curve(rf_model, "Courbe d'apprentissage (Random Forest)", X, y, cv=5, n_jobs=-1)
    learning_curve_plot.savefig('output/random_forest/learning_curve_rf.png')  # Enregistrement du plot Learning Curve

    plt.show()

    # Créer un modèle RandomForest avec un nombre d'arbres réduit
    reduced_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)

    # Afficher la courbe d'apprentissage pour le modèle réduit
    reduced_learning_curve_plot = plot_learning_curve(reduced_model, "Courbe d'apprentissage (Modèle Réduit)", X, y, cv=5, n_jobs=-1)
    reduced_learning_curve_plot.savefig('output/random_forest/learning_curve_rf_reduced.png')  # Enregistrement du plot Reduced Learning Curve

    plt.show()

    elapsed_time = time.time() - start_time
    print(f"Random Forest process completed in {elapsed_time:.2f} seconds")

#regression logistique#
#######################
def train_logistic_regression(X_train, y_train, random_state=42):
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def logistic_regression_process(rna_file, label_file):
    start_time = time.time()
    # Chargement des données
    data = load_data(rna_file, label_file)

    # Séparation des features (X) et de la variable cible (y)
    X = data.drop(['Unnamed: 0', 'Class'], axis=1)
    y = data['Class']

    # PCA
    plot_pca(X, y)

    # Statistiques descriptives
    desc_stats = descriptive_statistics(X)
    print(desc_stats)

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modèle Régression Logistique
    logreg_model = train_logistic_regression(X_train, y_train)

    # Évaluation du modèle
    evaluate_model(logreg_model, X_test, y_test)

    # Cross-validation
    cross_validate(logreg_model, X, y)

    # Learning curve
    learning_curve_plot = plot_learning_curve(logreg_model, "Courbe d'apprentissage (Régression Logistique)", X, y, cv=5, n_jobs=-1)
    learning_curve_plot.savefig('output/logistic_regression/learning_curve_logreg.png')  # Enregistrement du plot Learning Curve

    plt.show()
    elapsed_time = time.time() - start_time
    print(f"Logistic regression process completed in {elapsed_time:.2f} seconds")

####reseaux_neurone####
#######################
def train_neural_network(X_train, y_train, hidden_layer_sizes=(100,), max_iter=200, random_state=42):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def neural_network_process(rna_file, label_file):
    start_time = time.time()
    # Chargement des données
    data = load_data(rna_file, label_file)

    # Séparation des features (X) et de la variable cible (y)
    X = data.drop(['Unnamed: 0', 'Class'], axis=1)
    y = data['Class']

    # PCA
    plot_pca(X, y)

    # Statistiques descriptives
    desc_stats = descriptive_statistics(X)
    print(desc_stats)

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modèle Réseau Neuronal
    nn_model = train_neural_network(X_train, y_train)

    # Évaluation du modèle
    evaluate_model(nn_model, X_test, y_test)

    # Cross-validation
    cross_validate(nn_model, X, y)

    # Learning curve
    learning_curve_plot = plot_learning_curve(nn_model, "Courbe d'apprentissage (Réseau Neuronal)", X, y, cv=5, n_jobs=-1)
    learning_curve_plot.savefig('output/neural_network/learning_curve_nn.png')  # Enregistrement du plot Learning Curve

    plt.show()
    elapsed_time = time.time() - start_time
    print(f"Neural network process completed in {elapsed_time:.2f} seconds")



