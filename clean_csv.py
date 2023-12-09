import pandas as pd
from sklearn.impute import SimpleImputer

# Charger les données
data = pd.read_csv('Financial_Fraud_missing_data.csv')

# Afficher les premières lignes pour comprendre la structure des données
print(data.head())

# Vérifier la présence de valeurs manquantes
missing_values = data.isnull().sum()
print("Valeurs manquantes par colonne :\n", missing_values)

# Imputation des valeurs manquantes pour les colonnes numériques
numeric_columns = data.select_dtypes(include=['number']).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Imputation des valeurs manquantes pour les colonnes numériques
numeric_imputer = SimpleImputer(strategy='mean')
data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])

# Imputation des valeurs manquantes pour les colonnes catégorielles
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

# Vérifier les types de données
data_types = data.dtypes
print("Types de données par colonne :\n", data_types)

# Supprimer les duplicatas
data = data.drop_duplicates()

# Vérifier les duplicatas après suppression
duplicates_count = data.duplicated().sum()
print("Nombre de duplicatas supprimés :", duplicates_count)

# Enregistrer les données nettoyées dans un nouveau fichier CSV
data.to_csv('Dataset_Clean.csv', index=False)