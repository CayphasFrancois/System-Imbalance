import json
import pandas as pd
import matplotlib.pyplot as plt

def process_merit_order(file_a_path: str, file_b_path: str, output_csv_path: str):
    """Lit les fichiers A.csv (incremental) et B.csv (decremental) et génère un CSV

    contenant pour chaque quart d'heure les deux dictionnaires {MW_cumulés:
    prix}.
    """

    print("Chargement des fichiers CSV...")
    # 1. Lecture des fichiers CSV
    # On spécifie le séparateur ';' conformément à votre description
    df_inc = pd.read_csv(file_a_path, sep=";")
    df_dec = pd.read_csv(file_b_path, sep=";")

    for df in [df_inc, df_dec]:
        df["Datetime"] = (
            pd.to_datetime(df["Datetime"], utc=True)
            .dt.tz_convert("Europe/Brussels")
            .dt.tz_localize(None))

    # S'assurer que les colonnes numériques sont au bon format
    for df in [df_inc, df_dec]:
        df["Rank"] = pd.to_numeric(df["Rank"])
        df["Bid Volume"] = pd.to_numeric(df["Bid Volume"])
        df["Bid Price"] = pd.to_numeric(df["Bid Price"])

    def build_merit_order_dict(group: pd.DataFrame) -> str:
        """Trie le groupe par 'Rank', calcule le volume cumulé et renvoie le

        dictionnaire sous forme de chaîne JSON.
        """
        # Tri par rang (Merit Order)
        group_sorted = group.sort_values(by="Rank")

        # Calcul du volume cumulé
        cum_volume = group_sorted["Bid Volume"].cumsum()
        prices = group_sorted["Bid Price"]
        activation_order= group_sorted["Activation Order"]

        # Construction du dictionnaire {Volume_Cumulé: Prix}
        # Les clés JSON doivent être des chaînes de caractères, mais vous pouvez caster en float/int
        mo_dict = {
            round(vol, 2): [round(price, 2),int(a_o)]
            for vol, price,a_o in zip(cum_volume, prices, activation_order)
        }

        # Conversion en JSON pour un stockage propre dans le CSV
        return json.dumps(mo_dict)

    print("Traitement des données incrémentales (Fichier A)...")
    # Groupe par Datetime et applique la construction du dictionnaire
    inc_series = df_inc.groupby("Datetime").apply( #type: ignore
        build_merit_order_dict, include_groups=False #type: ignore
    )
    inc_df = inc_series.reset_index(name="incremental_merit_order")

    print("Traitement des données décrémentales (Fichier B)...")
    dec_series = df_dec.groupby("Datetime").apply( #type: ignore
        build_merit_order_dict, include_groups=False #type: ignore
    )
    dec_df = dec_series.reset_index(name="decremental_merit_order")

    print("Fusion des données par quart d'heure...")
    # 2. Fusion (Outer Join) pour s'assurer de ne perdre aucun quart d'heure
    merged_df = pd.merge(inc_df, dec_df, on="Datetime", how="outer")

    # Remplacer les valeurs manquantes éventuelles par un dictionnaire vide '{}'
    merged_df.fillna("{}", inplace=True)

    # 3. Sauvegarde dans le fichier CSV final
    merged_df.to_csv(output_csv_path, sep=";", index=False)
    print(
        f"Traitement terminé ! Fichier de sortie généré : {output_csv_path}"
    )


def plot_first_quarter_merit_orders(csv_path: str):
  """Lit la première ligne du fichier CSV de dictionnaires et trace

  les courbes collées à l'axe central (0) pour l'incrémental et le décrémental.
  """
  # 1. Chargement du fichier CSV
  df = pd.read_csv(csv_path, sep=";")

  if df.empty:
    print("Le fichier CSV est vide.")
    return

  # 2. Récupération de la première ligne
  row = df.iloc[0]
  datetime_str = row["Datetime"]

  # 3. Conversion des chaînes JSON en dictionnaires
  inc_dict = json.loads(row["incremental_merit_order"])
  dec_dict = json.loads(row["decremental_merit_order"])

  # --- Données Incrémentales (positives, à droite de 0) ---
  if inc_dict:
    raw_inc_vols = [float(k) for k in inc_dict.keys()]
    raw_inc_prices = [val[0] for val in inc_dict.values()]

    # On ajoute 0 au début pour coller à l'axe central
    inc_volumes = [0.0] + raw_inc_vols
    # Le prix de la première tranche s'applique dès 0
    inc_prices = [raw_inc_prices[0]] + raw_inc_prices
  else:
    inc_volumes, inc_prices = [], []

  if dec_dict:
      # Tri par volume absolu croissant (les plus proches de 0 en premier)
      sorted_dec = sorted([(float(k), v) for k, v in dec_dict.items()])
      raw_dec_vols = [-item[0] for item in sorted_dec]  # Valeurs négatives
      
      # MODIFICATION ICI : on extrait le prix qui est à l'index [0] de la liste item[1]
      raw_dec_prices = [item[1][0] for item in sorted_dec]

      # Pour aller jusqu'à 0 du côté gauche, on ajoute 0 à la fin des x négatifs
      dec_volumes = [0.0] + raw_dec_vols 
      # Le prix du dernier palier (le plus proche de 0) s'étend jusqu'à l'axe 0
      dec_prices = [raw_dec_prices[-1]] + raw_dec_prices 
  else:
      dec_volumes, dec_prices,_ = [], [], []

  # 4. Tracé du graphique
  plt.figure(figsize=(14, 6))

  # Courbe incrémentale (côté positif, depart à x=0 avec where='post')
  if inc_volumes:
    plt.step(
        inc_volumes,
        inc_prices,
        where="post",
        label="Incrémental (Hausse / SI < 0)",
        color="crimson",
        linewidth=2,
    )

  # Courbe décrémentale (côté négatif, arrivée à x=0 avec where='pre')
  if dec_volumes:
    plt.step(
        dec_volumes,
        dec_prices,
        where="pre",
        label="Décrémental (Baisse / SI > 0)",
        color="dodgerblue",
        linewidth=2,
    )

  # Mise en forme du graphique
  plt.title(
      f"Merit Orders (Jointes au centre) - Quart d'heure : {datetime_str}",
      fontsize=14,
      fontweight="bold",
  )
  plt.xlabel(
      "Volume cumulé (MW)  Décrémental | Incrémental ",
      fontsize=12,
  )
  plt.ylabel("Prix des Bids (€/MWh)", fontsize=12)

  # Lignes de référence
  plt.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
  plt.axvline(
      0, color="black", linestyle="-", linewidth=1.2, alpha=0.8, label="Centre (SI = 0)"
  )

  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend(fontsize=11)
  plt.tight_layout()

  # Affichage
  plt.show()

# --- EXÉCUTION ---
if __name__ == "__main__":
    # Remplacez ces chemins par les noms de vos fichiers réels
    FILE_A = "C:\\Document_UMONS\\Thèse-Code\\TD3_Minute\\Data_Now\\Raw_Data_Elia\\Individual incremental balancing energy bids_05_03_25-25_10_2026.csv"
    FILE_B = "C:\\Document_UMONS\\Thèse-Code\\TD3_Minute\\Data_Now\\Raw_Data_Elia\\Individual decremental balancing energy bids_05_03_25-25_10_2026.csv"
    OUTPUT_FILE = "C:\\Document_UMONS\\Thèse-Code\\TD3_Minute\\Data_Now\\merit_order_dictionaries.csv"

    process_merit_order(FILE_A, FILE_B, OUTPUT_FILE)
    plot_first_quarter_merit_orders("C:\\Document_UMONS\\Thèse-Code\\TD3_Minute\\Data_Now\\merit_order_dictionaries.csv")