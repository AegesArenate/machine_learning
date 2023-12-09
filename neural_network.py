import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score

# Charger les données depuis le fichier CSV
data = pd.read_csv('Financial_Fraud.csv')

# Supprimer les colonnes inutiles
data = data.drop(['nameOrig', 'nameDest'], axis=1)

# Transformer les colonnes catégorielles en one-hot encoding (si nécessaire)
data = pd.get_dummies(data, columns=['type'])

# Séparer les features (X) et la cible (y)
X = data.drop(['isFraud'], axis=1)
y = data['isFraud']

# Normaliser les features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir les données en tenseurs PyTorch
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test.values)

# Définir le modèle du réseau de neurones
class FraudDetectionModel(nn.Module):
    def __init__(self, input_size):
        super(FraudDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Instancier le modèle
input_size = X_train.shape[1]
model = FraudDetectionModel(input_size)

# Définir la fonction de perte et l'optimiseur
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Époque {epoch + 1}/{epochs}, Perte : {loss.item()}')

# Évaluer le modèle sur l'ensemble de test
with torch.no_grad():
    model.eval()
    test_output = model(X_test_tensor)
    predictions = (test_output.squeeze() > 0.5).float()

    # Calculer l'erreur quadratique moyenne
    mse = mean_squared_error(y_test_tensor, predictions)
    print(f'Erreur Quadratique Moyenne : {mse}')

    # Calculer la précision
    accuracy = accuracy_score(y_test_tensor, predictions)
    print(f'Précision : {accuracy}')

    # Calculer la matrice de confusion
    cm = confusion_matrix(y_test_tensor, predictions)
    print(f'Matrice de Confusion :\n{cm}')

    # Calculer le rapport de classification (Précision, Rappel, Score F1)
    class_report = classification_report(y_test_tensor, predictions)
    print(f'Rapport de Classification :\n{class_report}')

    # Calculer le score F1
    f1 = f1_score(y_test_tensor, predictions)
    print(f'Score F1 : {f1}')

    # Calculer l'AUC ROC
    auc_roc = roc_auc_score(y_test_tensor, predictions)
    print(f'AUC ROC : {auc_roc}')