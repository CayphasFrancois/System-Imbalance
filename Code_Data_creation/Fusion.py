import pandas as pd

# Charger l'ancien fichier (qu'il soit en .xlsx ou .csv)
df_ancien = pd.read_excel('Data_Now_2024_min_With_Ladder_2.xlsx') # ou pd.read_csv(...)
print("Premier lu")
# Charger ton nouveau fichier transformé
df_nouveau = pd.read_excel('Data_Now_2024_min_With_Ladder_1.xlsx') # ou pd.read_csv(...)
print("Deuxième lu")

print(f"Lignes ancien fichier : {len(df_ancien)}")
print(f"Lignes nouveau fichier : {len(df_nouveau)}")

print("Fusion des données...")
# Concaténation verticale (l'un à la suite de l'autre)
df_final = pd.concat([df_ancien, df_nouveau], ignore_index=True)

print(f"Total lignes après fusion : {len(df_final)}")

# Sauvegarde au format CSV (recommandé pour dépasser la limite des 1 million de lignes d'Excel)
# Si ton fichier dépasse 1 million de lignes, Excel plantera à l'ouverture d'un .xlsx, 
# le format .csv est donc beaucoup plus sûr pour stocker des données à la minute sur plusieurs années.
output_file = 'Data_Now_2024_min_SI_Price_With_Ladder.xlsx'
df_final.to_excel(output_file, index=False)

print("Fusion et sauvegarde terminées avec succès :", output_file)