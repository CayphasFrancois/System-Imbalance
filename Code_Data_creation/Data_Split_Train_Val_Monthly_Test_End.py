import pandas as pd


def split_dataset(input_csv_path: str):
  print("Chargement du fichier source...")
  df = pd.read_csv(input_csv_path, sep=";")

  # Conversion de la colonne datetime
  df["_temp_datetime"] = pd.to_datetime(df["datetime"])

  # 1. Isolation des deux derniers mois du dataset pour le test_df
  df["year_month"] = df["_temp_datetime"].dt.to_period("M")
  unique_months = sorted(df["year_month"].unique())

  if len(unique_months) < 2:
    raise ValueError(
        "Le dataset contient moins de 2 mois de données, impossible d'isoler"
        " les deux derniers mois pour le test."
    )

  last_two_months = unique_months[-2:]
  print(f"Mois sélectionnés pour le Test (2 derniers mois) : {last_two_months}")

  # Test set : contient uniquement les données des deux derniers mois
  test_df = df[df["year_month"].isin(last_two_months)].drop(
      columns=["_temp_datetime", "year_month"]
  )

  # Reste des données (tout sauf les deux derniers mois) pour Train et Validation
  remaining_df = df[~df["year_month"].isin(last_two_months)]
  rem_days = remaining_df["_temp_datetime"].dt.day

  # Application des règles de découpage sur le reste des données :
  # - Train set : jours 1 à 20 (ou 1 à 5 selon votre préférence initiale, mis à 20 ici en cohérence avec le commentaire)
  # - Validation set : du 25 à la fin du mois
  train_df = remaining_df[rem_days <= 25].drop(
      columns=["_temp_datetime", "year_month"]
  )
  val_df = remaining_df[rem_days >= 25].drop(
      columns=["_temp_datetime", "year_month"]
  )

  # Sauvegarde des trois fichiers CSV (avec le même séparateur ';')
  print("Sauvegarde des jeux de données...")
  train_filename = "C:\\Document_UMONS\\Thèse-Code\\SI_side_repository\\Data\\Split_Train_Val_Month_Test_End\\Dataset_Train_2025_2026_Current_SI.csv"
  test_filename = "C:\\Document_UMONS\\Thèse-Code\\SI_side_repository\\Data\\Split_Train_Val_Month_Test_End\\Dataset_Test_2025_2026_Current_SI.csv"
  val_filename = "C:\\Document_UMONS\\Thèse-Code\\SI_side_repository\\Data\\Split_Train_Val_Month_Test_End\\Dataset_Val_2025_2026_Current_SI.csv"

  train_df.to_csv(train_filename, sep=";", index=False)
  test_df.to_csv(test_filename, sep=";", index=False)
  val_df.to_csv(val_filename, sep=";", index=False)

  print("Découpage terminé avec succès !")
  print(f" -> Train saved: {train_filename} ({len(train_df)} lignes)")
  print(f" -> Test saved: {test_filename} ({len(test_df)} lignes)")
  print(f" -> Validation saved: {val_filename} ({len(val_df)} lignes)")


if __name__ == "__main__":
  # Remplace par le nom de ton fichier CSV source global
  INPUT_FILE = "C:\\Document_UMONS\\Thèse-Code\\SI_side_repository\\Data\\Current system imbalance_processed.csv"
  split_dataset(INPUT_FILE)