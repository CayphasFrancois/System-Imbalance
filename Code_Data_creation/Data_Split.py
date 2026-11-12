import pandas as pd


def split_dataset(input_csv_path: str):
  print("Chargement du fichier source...")
  df = pd.read_csv(input_csv_path, sep=";")

  # Conversion de la colonne datetime pour extraire le jour du mois
  # On gère le format datetime proprement
  df["_temp_datetime"] = pd.to_datetime(df["datetime"])
  days = df["_temp_datetime"].dt.day

  # Application des règles de découpage :
  # 1. Training set : les 20 premiers jours du mois (jours 1 à 20)
  train_df = df[days <= 20].drop(columns=["_temp_datetime"])

  # 2. Test set : du 21 au 24 du mois inclus
  test_df = df[(days >= 21) & (days <= 27)].drop(columns=["_temp_datetime"])

  # 3. Validation set : du 25 à la fin du mois
  val_df = df[days >= 28].drop(columns=["_temp_datetime"])

  # Sauvegarde des trois fichiers CSV (avec le même séparateur ';')
  print("Sauvegarde des jeux de données...")
  train_filename = "C:\\Document_UMONS\\Thèse-Code\\TD3_Minute\\Data_Now\\Dataset_Train_2025_2026_Current_SI.csv"
  test_filename = "C:\\Document_UMONS\\Thèse-Code\\TD3_Minute\\Data_Now\\Dataset_Test_2025_2026_Current_SI.csv"
  val_filename = "C:\\Document_UMONS\\Thèse-Code\\TD3_Minute\\Data_Now\\Dataset_Val_2025_2026_Current_SI.csv"

  train_df.to_csv(train_filename, sep=";", index=False)
  test_df.to_csv(test_filename, sep=";", index=False)
  val_df.to_csv(val_filename, sep=";", index=False)

  print("Découpage terminé avec succès !")
  print(f" -> Train saved: {train_filename} ({len(train_df)} lignes)")
  print(f" -> Test saved: {test_filename} ({len(test_df)} lignes)")
  print(f" -> Validation saved: {val_filename} ({len(val_df)} lignes)")


if __name__ == "__main__":
  # Remplace par le nom de ton fichier CSV source global
  INPUT_FILE = "C:\\Document_UMONS\\Thèse-Code\\TD3_Minute\\Data_Now\\Current system imbalance_processed.csv"
  split_dataset(INPUT_FILE)