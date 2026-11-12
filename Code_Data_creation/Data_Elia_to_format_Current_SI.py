import ast
import gc
import time
import numpy as np
import pandas as pd


def process_system_imbalance(input_csv_path: str, output_csv_path: str):
    start_time = time.time()
    print("Chargement complet du fichier System Imbalance...")

    # Chargement direct du fichier entier
    df = pd.read_csv(input_csv_path, sep=";")
    df.columns = [c.strip().lower() for c in df.columns]

    # CONSERVATION DU FORMAT DATETIME D'ORIGINE
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    df = df.sort_values("datetime").reset_index(drop=True)

    # 1. Calcul de l'IGCC : IGCC_+ - IGCC_-
    col_igcc_plus = [c for c in df.columns if "igcc" in c and ("+" in c or "up" in c)][0]
    col_igcc_minus = [c for c in df.columns if "igcc" in c and ("-" in c or "down" in c)][0]
    
    df["igcc"] = df[col_igcc_plus] - df[col_igcc_minus]
    df["igcc"] = df["igcc"].round(4)

    # 2. Gestion de la NRV (nrv_old et nrv_new)
    df["nrv_old"] = df["ace"] - df["system imbalance"]
    df["nrv_old"] = df["nrv_old"].round(4)

    df["nrv_new"] = df["nrv_old"] - df["igcc"]
    df["nrv_new"] = df["nrv_new"].round(4)

    # Calcul des variables temporelles (sin/cos)
    dt = df["datetime"]
    minutes = dt.dt.minute
    hours = dt.dt.hour
    day_of_year = dt.dt.dayofyear
    is_leap = dt.dt.is_leap_year
    total_days_in_year = np.where(is_leap, 366, 365)

    df["min_sin"] = np.sin(2 * np.pi * minutes / 60).round(6)
    df["min_cos"] = np.cos(2 * np.pi * minutes / 60).round(6)
    df["h_sin"] = np.sin(2 * np.pi * hours / 24).round(6)
    df["h_cos"] = np.cos(2 * np.pi * hours / 24).round(6)
    df["day_sin"] = np.sin(2 * np.pi * (day_of_year - 1) / total_days_in_year).round(6)
    df["day_cos"] = np.cos(2 * np.pi * (day_of_year - 1) / total_days_in_year).round(6)

    # Sélection des colonnes finales (sans les moyennes sur 15 minutes ni alpha)
    columns_to_keep = [
        "datetime",
        "system imbalance",
        "ace",
        "igcc",
        "nrv_old",
        "nrv_new",
        "min_sin",
        "min_cos",
        "h_sin",
        "h_cos",
        "day_sin",
        "day_cos",
    ]
    
    columns_to_keep = [c for c in columns_to_keep if c in df.columns]
    df_final = df[columns_to_keep]

    # Sauvegarde directe du fichier complet
    df_final.to_csv(output_csv_path, sep=";", index=False)
    
    elapsed = time.time() - start_time
    print(f"Traitement complet terminé avec succès en {elapsed:.2f} secondes !")

if __name__ == "__main__":
  INPUT_FILE = (
      "C:\\Document_UMONS\\Thèse-Code\\TD3_Minute\\Data_Now\\Raw_Data_Elia\\"
      "Current system imbalance_05_03_25-25_10_2026.csv"
  )
 
  OUTPUT_FILE = (
      "C:\\Document_UMONS\\Thèse-Code\\TD3_Minute\\Data_Now\\"
      "Current system imbalance_processed.csv"
  )

  process_system_imbalance(INPUT_FILE, OUTPUT_FILE)