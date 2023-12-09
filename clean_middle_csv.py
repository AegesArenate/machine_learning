import pandas as pd
from sklearn.impute import SimpleImputer

# Charger le CSV
df = pd.read_csv("Financial_Fraud_missing_data.csv")

# Vérifier la présence de valeurs manquantes
missing_values = df.isnull().sum()
print("Valeurs manquantes par colonne :\n", missing_values)

# Calculer la moyenne pour chaque type
mean_payment = df[df['type'] == 'PAYMENT']['amount'].mean()
mean_transfer = df[df['type'] == 'TRANSFER']['amount'].mean()
mean_cash_out = df[df['type'] == 'CASH_OUT']['amount'].mean()

# Remplir les données manquantes dans la colonne 'type' en fonction de la moyenne des autres types
df['type'].fillna(df.groupby('type')['type'].transform(lambda x: x.mode().iloc[0]), inplace=True)

# Remplir les données manquantes dans la colonne 'amount' en fonction de la moyenne des autres types
df['amount'].fillna(df.groupby('type')['amount'].transform('mean'), inplace=True)

# Remplir les éventuelles données manquantes dans la colonne 'type' après le remplissage de 'amount'
df['type'].fillna(df['type'].mode().iloc[0], inplace=True)

# Supprimer les duplicatas
data = df.drop_duplicates()

# Vérifier les duplicatas après suppression
duplicates_count = df.duplicated().sum()
print("Nombre de duplicatas supprimés :", duplicates_count)

# Sauvegarder le DataFrame nettoyé dans un nouveau fichier CSV
df.to_csv("dataset_clean.csv", index=False)