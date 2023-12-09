import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, accuracy_score, precision_score, confusion_matrix, f1_score, roc_auc_score

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

# Création du modèle de régression linéaire
model = LinearRegression()

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred = model.predict(X_test)

# Calcul de l'erreur moyenne quadratique
mse = mean_squared_error(y_test, y_pred)
print("Erreur moyenne quadratique :", mse)

# Calcul du coefficient de détermination R²
r2 = r2_score(y_test, y_pred)
print("Coefficient de détermination R²:", r2)

# Seuil pour la classification binaire
threshold = 0.5

# Conversion des prédictions en classe binaire
binary_predictions = (y_pred > threshold).astype(int)

# Calcul de l'Accuracy
accuracy = accuracy_score(y_test, binary_predictions)
print("Accuracy:", accuracy)

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
roc_auc = roc_auc_score(y_test, y_pred)
print("AUC-ROC Score:", roc_auc)

# Affichage du graphique de dispersion avec la ligne de régression
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Vraies étiquettes')
plt.ylabel('Prédictions')
plt.title('Graphique de Dispersion avec Régression Linéaire')
plt.show()

