import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, precision_score, f1_score, roc_auc_score

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

# Création du modèle de régression logistique
model = LogisticRegression()

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred = model.predict(X_test)

# Calcul de la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle :", accuracy)

# Création d'une matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :", cm)

# Calcul du coefficient de détermination R²
r2 = r2_score(y_test, y_pred)
print("Coefficient de détermination R²:", r2)

# Calcul de l'erreur quadratique moyenne
mse = mean_squared_error(y_test, y_pred)
print("Erreur quadratique moyenne :", mse)

# Seuil pour la classification binaire
threshold = 0.5

# Conversion des prédictions en classe binaire
binary_predictions = (y_pred > threshold).astype(int)

# Calcul de la Vraie Précision (True Precision)
true_precision = precision_score(y_test, binary_predictions)
print("Vraie Précision:", true_precision)

# Calcul du F1 Score
f1 = f1_score(y_test, binary_predictions)
print("F1 Score:", f1)

# Calcul de l'AUC ROC Score
roc_auc = roc_auc_score(y_test, y_pred)
print("AUC-ROC Score:", roc_auc)

# Affichage de la matrice de confusion sous forme de heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non Fraud', 'Fraud'], yticklabels=['Non Fraud', 'Fraud'])
plt.xlabel('Prédictions')
plt.ylabel('Vraies étiquettes')
plt.title('Matrice de Confusion')
plt.show()