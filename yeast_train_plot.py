# import everything
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors

import seaborn as sns

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, accuracy_score, f1_score, make_scorer, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.manifold import TSNE

import scikit_posthocs as sp

import scipy.stats as stats
import math

import time as timer

import copy

from IPython.display import display

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import sys

from umap import UMAP

import plotly.express as px

# Enter Data Input:

def input_csv_file(csv_file):
    df = pd.read_csv(csv_file)
    return df

# Select features:
def feature_target_selection(df, features, target_column):
    X=df[features]
    y=df[target_column]
    return X, y

# Splt Data:
def split_data(X, y, test_size=0.5, val_size=0.15):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
        )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=test_size, random_state=42
        )
    return X_test, X_train, X_val, y_test, y_train, y_val

# Generate Heat Map:
def heat_map_generate(df, features):
    corr = df.drop(columns=["Image_File", "Cell-Number", "Cell-Type"]).corr(numeric_only=True)
    # Plot
    plt.figure(figsize=(13, 14))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidth=0.5)
    plt.title("Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'Heatmap_example.png', dpi = 300)

# Dunn's Test:
def conduct_dunn_test(df, feature):
    p_value = sp.posthoc_dunn(
        df,
        val_col = feature,
        group_col = 'Cell-Type',
        p_adjust='holm'
        )
    print(p_value)
    p_value.to_csv("my_pvalues_example.csv")
    return p_value 

# Exploratory Analysis:
# Histogram:
def hist_plot(df, feature):
    sns.histplot(
    data = df, 
    x=feature,
    hue="Cell-Type"
    )
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.title(f'{feature} Distributor')
    plt.tight_layout()
    plt.savefig(f'Histogram_example.png', dpi = 300) # why is it printing heat map?

# KDE:
def kde_plot(df, cell_type1, cell_type2):
        df_1 = df[df["Cell-Type"]==cell_type1]
        df_2 = df[df["Cell-Type"]==cell_type2]
        sns.kdeplot(
            data=df_1,
        x='PC1', 
        y='PC2', 
        color='red', 
        label =cell_type1
        )
        sns.kdeplot(
        data = df, 
        x='PC1', 
        y='PC2', 
        color='blue', 
        label = cell_type2
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"KDE: {cell_type1} vs. {cell_type2}")
        plt.legend()
        plt.savefig(f"KDE_example.png", dpi=300)


# Machine Learning Approaches:
# Random Forest:
def random_forest(X_train, y_train):
    parameters = {'n_estimators':[50, 100, 200],
              'max_features':['sqrt', 'log2'],
              'max_depth':[None,5, 10, 20],
              'max_leaf_nodes':[None, 50, 100]
            }
    model = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(model, 
                       param_grid=parameters, 
                       cv=5
                       )
    clf.fit(X_train, y_train)
    print(f"Best Parameters:{clf.best_params_}")
    print(f"\nBest CV Accuracy: {clf.best_score_}")
    return clf

def random_forest_evaluation(clf, X_train, y_train, X_test, y_test):
    best_model = clf.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
    return y_pred_test

def random_forest_cm(clf, X_test, y_test):
    rf_test_pred = clf.predict(X_test)
    test_cm = confusion_matrix(y_test, rf_test_pred)
    rf_display = ConfusionMatrixDisplay(
        confusion_matrix=test_cm, 
        display_labels = ['no-budded', 
                          'small-budded', 
                          'medium-budded', 
                          'large-budded', 
                          'other'])
    fig, ax = plt.subplots(figsize=(10,10))
    rf_display.plot(ax=ax)
    plt.title(f"Confusion Matrix- Random Forest")
    plt.savefig(f'RF Confusion Matrix_example.png', dpi = 300)

def svc_evaluation(X_train, y_train):
    parameters = {
    'C': [ 0.1, 1.0, 10.0, 100.0],
    'kernel':['linear', 'rbf'],
    'gamma': [1, 0.1, 0.001]
    }
    grid = GridSearchCV(SVC(), 
                        param_grid=parameters, 
                        n_jobs = -1, 
                        verbose=2)
    grid.fit(X_train, y_train)
    print(f"Best Parameters: {grid.best_params_}")
    print(f"\nBest CV Accuracy: {grid.best_score_}")
    return grid

def svc_cm(grid, X_test, y_test):
    svc_test_pred = grid.predict(X_test)
    test_cm = confusion_matrix(y_test, svc_test_pred)
    svc_display = ConfusionMatrixDisplay(confusion_matrix=test_cm, display_labels = ['no-budded', 'small-budded', 'medium-budded', 'large-budded', 'other'])
    # Display Matrix
    fig,ax = plt.subplots(figsize=(10,10))
    svc_display.plot(ax=ax)
    plt.title(f"Confusion Matrix- SVC")
    plt.tight_layout()
    plt.savefig(f'SVC Confusion Matrix_example.png', dpi = 300)

def pca_plot(X, df, ctype1, ctype2):
    X_scale = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scale)
    df["PC1"] = components[:, 0]
    df["PC2"] = components[:, 1]
    # Create separate dataframes AFTER adding PC1 and PC2
    df_1 = df[df["Cell-Type"] == ctype1] # how do I make this "customizable"?
    df_2 = df[df["Cell-Type"] == ctype2]
    # Plot PCA
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_1,
        x="PC1",
        y="PC2",
        color="red",
        label=ctype1
    )
    sns.scatterplot(
        data=df_2,
        x="PC1",
        y="PC2",
        color="blue",
        label=ctype2
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA: {ctype1} vs {ctype2}")
    plt.legend()
    plt.savefig(f'PCA_example.png', dpi = 300)

def t_sne(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(n_components = 2, random_state=0)
    projections = tsne.fit_transform(X_scaled)
    return projections

def plot_t_sne(df, projections):
    fig = px.scatter(
        projections, x=0, y=1,
        color=df['Cell-Type'], 
        labels={'x': 't-SNE 1',
        'y': 't-SNE 2',
        'color': 'Cell Type'},
        title = 't-SNE of Yeast Cell Morphological Features'
    )
    fig.show()
    fig.write_image("tSNE_example.png")

def u_map(df, X_scaled):
    # UMAP:
    umap = UMAP(n_components =2, init='random', random_state=42)
    proj = umap.fit_transform(X_scaled)
    return proj

def plot_u_map(df, proj):
    fig = px.scatter(proj, 
                    x=0, 
                    y=1,
                    color=df['Cell-Type'], 
                    labels={'x': 'UMAP 1',
                        'y': 'UMAP 2',
                        'color': 'Cell Type'
                        },
                        title = 'UMAP of Yeast Cell Morphological Features'
)
    fig.show()
    fig.write_image("UMap_example.png") # resource: https://plotly.com/python/static-image-export/

def controlling_the_flow(csv_file, target_column, test_size=0.5, val_size=0.15):
    features = ["Perimeter", "Area", "Perimeter-Crofton", "Eccentricity", "Convex-Area", "Equivalent-Diameter", "Mean-Intensity", "Circularity"]
   
    df=input_csv_file(csv_file)

    csv_file = "User_File_Location"
   
    X,y = feature_target_selection(df, features, target_column)

    X_test, X_train, X_val, y_test, y_train, y_val = split_data(X, y, test_size, val_size)

    heat_map_generate(df, features)

    p_value = conduct_dunn_test(df, target_column)

    hist_plot(df, "Area")

    pca_plot(X, df, "medium-budded", "large-budded")

    kde_plot(df, "medium-budded", "large-budded")

    X_test, X_train, X_val, y_test, y_train, y_val = split_data(X, y)

    clf = random_forest(X_train, y_train) # to save the results for later use in confusion matrix

    random_forest_evaluation(clf, X_train, y_train, X_test, y_test)

    random_forest_cm(clf, X_test, y_test)

    grid = svc_evaluation(X_train, y_train)

    svc_cm(grid, X_test, y_test)

    projections = t_sne(X)

    plot_t_sne(df, projections)

    scaler = StandardScaler() # Should I write this in my functions instead?
    X_scaled = scaler.fit_transform(X)

    proj = u_map(df, X_scaled)

    plot_u_map(df, proj)


def main():
    try:
        csv_file = sys.argv[1]
        target_column = sys.argv[2]
    except IndexError:
        print('\nUsage:\n\n    python3 Machine_Learning.py [CSV-FILE] [TARGET-COLUMN-NAME]\n\n')
        sys.exit()

    controlling_the_flow(csv_file, target_column, test_size=0.5, val_size=0.15)

main()