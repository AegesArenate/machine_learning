import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Charger les données CSV
data = pd.read_csv('Financial_Fraud.csv')

# Encodage des colonnes catégorielles si nécessaire
label_encoder = LabelEncoder()
data['type'] = label_encoder.fit_transform(data['type'])
data['nameOrig'] = label_encoder.fit_transform(data['nameOrig'])
data['nameDest'] = label_encoder.fit_transform(data['nameDest'])

# Séparation des features et de la cible
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# Séparation des données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle Random Forest
model = RandomForestClassifier()

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred = model.predict(X_test)

# Calcul de la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle :", accuracy)

# Calcul du coefficient de détermination R²
r2 = model.score(X_test, y_test)
print("Coefficient de détermination R²:", r2)

# Seuil pour la classification binaire
threshold = 0.5

# Conversion des prédictions en classe binaire
binary_predictions = (y_pred > threshold).astype(int)

# Calcul de la Vraie Précision (True Precision)
true_precision = precision_score(y_test, binary_predictions)
print("Vraie Précision:", true_precision)

# Calcul de la Matrice de confusion
conf_matrix = confusion_matrix(y_test, binary_predictions)
print("Matrice de confusion:\n", conf_matrix)

# Calcul du F1 Score
f1 = f1_score(y_test, binary_predictions)
print("F1 Score:", f1)

# Calcul de l'AUC ROC Score
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print("AUC-ROC Score:", roc_auc)
