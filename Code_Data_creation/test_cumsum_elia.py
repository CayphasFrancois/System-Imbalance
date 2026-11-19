"""Teste un cumsum qui ignore les bids que le Cumulative Volume d'Elia ne compte pas.

1) Identifie les bids ignorés : ceux dont le Cumulative Volume n'augmente pas
   (par rapport au bid précédent du même quart d'heure).
2) Recalcule un cumsum de Bid Volume en ne sommant que les bids comptés et le
   compare au Cumulative Volume du fichier.
3) Profile les bids ignorés pour tenter de trouver la règle d'exclusion.
"""

import pandas as pd

FILES = {
    "INC": "C:\\Document_UMONS\\Thèse-Code\\SI_side_repository\\Data\\Raw_Data_Elia\\Individual incremental balancing energy bids_05_03_25-25_10_2026.csv",
    "DEC" : "C:\\Document_UMONS\\Thèse-Code\\SI_side_repository\\Data\\Raw_Data_Elia\\Individual decremental balancing energy bids_05_03_25-25_10_2026.csv",
}
TOL = 0.01


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    for c in ["Rank", "Activation Order", "Bid Volume", "Bid Price", "Cumulative Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Rank", "Bid Volume", "Cumulative Volume"])
    df = df[(df["Availability"].str.strip() == "Available") & (df["Bid Volume"] > 0)]
    return df.sort_values(["Datetime", "Rank"]).reset_index(drop=True)


def run(label: str, path: str):
    print(f"\n{'=' * 20} {label} {'=' * 20}")
    df = load(path)
    cum_file = df["Cumulative Volume"]

    # --- 1) Identification des bids comptés / ignorés ---
    added = cum_file.groupby(df["Datetime"]).diff().fillna(cum_file)
    df["added"] = added
    df["counted"] = added > 0.001
    df["negative"] = added < -0.001  # cumul qui diminue (anormal)
    df["partial"] = df["counted"] & ((added - df["Bid Volume"]).abs() > TOL)

    print(f"Bids Available analysés : {len(df)}")
    print(f"  comptés par Elia      : {df['counted'].sum()} ({df['counted'].mean():.2%})")
    print(f"  ignorés par Elia      : {(~df['counted']).sum()} ({(~df['counted']).mean():.2%})")
    print(f"  comptés partiellement : {df['partial'].sum()}")
    print(f"  cumul qui diminue     : {df['negative'].sum()}")

    # --- 2) cumsum en ignorant les bids ignorés ---
    vol_counted = df["Bid Volume"].where(df["counted"], 0)
    df["my_cum"] = vol_counted.groupby(df["Datetime"]).cumsum()
    diff = (df["my_cum"] - cum_file).abs()
    print(
        f"\nConcordance cumsum (bids ignorés exclus) vs Cumulative Volume : "
        f"{(diff < TOL).mean():.2%}, écart max {diff.max():.2f} MW"
    )

    bad = df[diff >= TOL]
    if not bad.empty:
        dt = bad["Datetime"].iloc[0]
        cols = ["Rank", "Activation Order", "Product", "Bid Volume",
                "Cumulative Volume", "added", "my_cum", "Bid Price"]
        print("\nExemple de quart d'heure encore divergent :")
        print(df[df["Datetime"] == dt][cols].head(40).to_string())

    # --- 3) Profil des bids ignorés ---
    ign = df[~df["counted"]]
    if ign.empty:
        return

    print("\n--- Profil des bids ignorés ---")
    print("\nPart ignorée par produit :")
    print((~df["counted"]).groupby(df["Product"]).mean().round(4))

    print("\nVolumes les plus fréquents (ignorés) :")
    print(ign["Bid Volume"].value_counts().head(10))

    print("\nVolume des bids ignorés vs comptés :")
    print(pd.DataFrame({
        "ignorés": ign["Bid Volume"].describe(),
        "comptés": df.loc[df["counted"], "Bid Volume"].describe(),
    }).round(2))

    print("\nCouples (Bid Volume, Bid Price) ignorés les plus répétés :")
    print(ign.groupby(["Bid Volume", "Bid Price"]).size().sort_values().tail(10))

    month = df["Datetime"].dt.strftime("%Y-%m")
    print("\nPart de bids ignorés par mois :")
    print((~df["counted"]).groupby(month).mean().round(4).to_string())

    hour = df["Datetime"].dt.tz_convert("Europe/Brussels").dt.hour
    print("\nPart de bids ignorés par heure locale :")
    print((~df["counted"]).groupby(hour).mean().round(4).to_string())

    print("\nRang moyen : ignorés =", round(ign["Rank"].mean(), 1),
          "| comptés =", round(df.loc[df["counted"], "Rank"].mean(), 1))


if __name__ == "__main__":
    for lbl, p in FILES.items():
        run(lbl, p)
