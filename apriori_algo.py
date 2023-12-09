import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Charger le CSV
df = pd.read_csv('Financial_Fraud.csv')

# Créer des paniers d'articles basés sur la colonne 'type'
transactions = df.groupby('step')['type'].apply(list).values.tolist()

# Utiliser TransactionEncoder pour convertir les paniers en une matrice binaire
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Appliquer l'algorithme Apriori pour obtenir les ensembles fréquents
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)

# Afficher les ensembles fréquents
print(frequent_itemsets)

# Générer des règles d'association
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Afficher les règles d'association
print(rules)