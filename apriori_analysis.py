# =======================================
# apriori_analysis.py
# Uses uploaded dataset (transactions_10000.csv)
# =======================================

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load the uploaded dataset
df = pd.read_csv("transactions_10000.csv")

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# Step 2: Apply Apriori algorithm
frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)
print(f"\n✅ Frequent itemsets found: {len(frequent_itemsets)}")
print(frequent_itemsets.head(10))

# Step 3: Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print(f"\n✅ Association rules generated: {len(rules)}")

# Step 4: Display first 1000 rules
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("\n📊 Showing first 1000 association rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(1000))
