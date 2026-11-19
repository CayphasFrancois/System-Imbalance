import json
import pandas as pd
import matplotlib.pyplot as plt


def _load_bids(path: str, label: str) -> pd.DataFrame:
    """Charge un CSV de bids et nettoie les types.

    Le Datetime est conservé en UTC (non ambigu). La conversion en heure
    locale ne se fait qu'à la fin, pour éviter de fusionner les deux
    occurrences de 02:00-03:00 lors du passage à l'heure d'hiver.
    """
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)

    for col in [
        "Rank",
        "Activation Order",
        "Bid Volume",
        "Bid Price",
        "Cumulative Volume",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Diagnostic rapide
    print(f"[{label}] lignes brutes : {len(df)}")
    for col in ["Product", "Resolution code", "Availability"]:
        if col in df.columns:
            print(f"[{label}] {col} : {df[col].unique()[:10]}")

    # Lignes inutilisables
    n_before = len(df)
    df = df.dropna(subset=["Rank", "Bid Volume", "Bid Price", "Activation Order"])
    if len(df) != n_before:
        print(f"[{label}] {n_before - len(df)} lignes avec NaN supprimées")

    # Les bids de volume nul n'apportent rien et créent des clés dupliquées
    # (même volume cumulé) dans le dictionnaire.
    n_before = len(df)
    df = df[df["Bid Volume"] > 0]
    if len(df) != n_before:
        print(f"[{label}] {n_before - len(df)} bids de volume 0 supprimés")

    # Contrôle du cumul d'Elia (aFRR d'abord, puis mFRR), avec/sans Availability
    _check_cumulative(df, label)

    if AVAILABLE_ONLY and "Availability" in df.columns:
        n_before = len(df)
        df = df[df["Availability"].str.strip() == "Available"]
        print(f"[{label}] {n_before - len(df)} bids 'Unavailable' supprimés")

    # Contrôle : un Rank doit être unique par quart d'heure
    dup = df.duplicated(subset=["Datetime", "Rank"]).sum()
    if dup:
        print(f"[{label}] ATTENTION : {dup} doublons (Datetime, Rank)")

    return df


# Le Rank contient déjà l'ordre (aFRR d'abord, puis mFRR)
SORT_COLS = ["Rank"]
# Ne garder que les bids 'Available' (le cumul d'Elia les exclut)
AVAILABLE_ONLY = True
# True : utilise directement la colonne 'Cumulative Volume' d'Elia comme clé.
# False : recalcule le cumsum de 'Bid Volume' (devrait donner le même résultat).
USE_FILE_CUMULATIVE = True
# Compteurs affichés en fin de traitement
DUP_COUNT = [0]  # clés dupliquées (bids écrasés)
NOT_CUMULATED = [0]  # bids 'Available' que le cumul d'Elia n'a pas comptés (retirés)
PARTIAL = [0]  # bids dont l'incrément du cumul != Bid Volume (comptés partiellement)


def _check_cumulative(df: pd.DataFrame, label: str):
    """Compare mon cumsum (tri Activation Order puis Rank) avec la colonne
    'Cumulative Volume' du fichier, avec et sans filtre Availability."""
    if "Cumulative Volume" not in df.columns:
        return
    ref = pd.to_numeric(df["Cumulative Volume"], errors="coerce")
    for avail_only in (False, True):
        mask = (
            df["Availability"].str.strip() == "Available"
            if avail_only
            else pd.Series(True, index=df.index)
        )
        d = df[mask].sort_values(["Datetime"] + SORT_COLS)
        cum = d.groupby("Datetime")["Bid Volume"].cumsum()
        diff = (cum - ref.loc[d.index]).abs()
        print(
            f"[{label}] Available seulement={avail_only} : "
            f"concordance {(diff < 0.01).mean():.2%}, écart max {diff.max():.2f} MW"
        )


def process_merit_order(file_a_path: str, file_b_path: str, output_csv_path: str):
    """Lit les fichiers incremental (A) et decremental (B) et génère un CSV
    contenant, pour chaque quart d'heure, les deux dictionnaires
    {MW_cumulés: [prix, activation_order]}.
    """
    print("Chargement des fichiers CSV...")
    df_inc = _load_bids(file_a_path, "INC")
    df_dec = _load_bids(file_b_path, "DEC")

    def build_merit_order_dict(group: pd.DataFrame) -> str:
        """Trie par 'Rank', calcule le volume cumulé, renvoie un JSON
        {vol_cumulé: [prix, activation_order]}.
        """
        g = group.sort_values(by=SORT_COLS)
        cum_volume = g["Bid Volume"].cumsum()
        if USE_FILE_CUMULATIVE:
            cum_volume = g["Cumulative Volume"].fillna(cum_volume)
            # Volume réellement ajouté par chaque bid, par rapport au maximum déjà
            # atteint : garantit des volumes cumulés strictement croissants
            # (l'environnement utilise np.searchsorted, qui l'exige).
            prev_max = cum_volume.cummax().shift(1).fillna(0.0)
            added = cum_volume - prev_max
            keep = added > 0.001
            NOT_CUMULATED[0] += int((~keep).sum())
            PARTIAL[0] += int(
                ((added - g["Bid Volume"]).abs().gt(0.01) & keep).sum()
            )
            g = g[keep]
            cum_volume = cum_volume[keep]

        mo_dict = {}
        for vol, price, a_o in zip(
            cum_volume, g["Bid Price"], g["Activation Order"]
        ):
            key = round(float(vol), 2)
            if key in mo_dict:
                DUP_COUNT[0] += 1
            mo_dict[key] = [round(float(price), 2), int(a_o)]

        return json.dumps(mo_dict)

    print("Traitement des données incrémentales...")
    inc_df = (
        df_inc.groupby("Datetime")
        .apply(build_merit_order_dict, include_groups=False)  # type: ignore
        .reset_index(name="incremental_merit_order")
    )

    print("Traitement des données décrémentales...")
    dec_df = (
        df_dec.groupby("Datetime")
        .apply(build_merit_order_dict, include_groups=False)  # type: ignore
        .reset_index(name="decremental_merit_order")
    )

    print(f"Bids écrasés par des clés de volume cumulé dupliquées : {DUP_COUNT[0]}")
    print(f"Bids 'Available' non comptés dans le cumul d'Elia (retirés) : {NOT_CUMULATED[0]}")
    print(f"Bids comptés partiellement (incrément != Bid Volume) : {PARTIAL[0]}")

    print("Fusion des données par quart d'heure...")
    merged_df = pd.merge(inc_df, dec_df, on="Datetime", how="outer")
    merged_df = merged_df.sort_values("Datetime").reset_index(drop=True)

    for col in ["incremental_merit_order", "decremental_merit_order"]:
        merged_df[col] = merged_df[col].fillna("{}")

    # Heure locale (naïve) pour lecture facile, ajoutée APRÈS le groupby.
    # 'Datetime' reste en UTC : c'est la clé non ambiguë.
    merged_df["Datetime_local"] = (
        merged_df["Datetime"].dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    )
    merged_df["Datetime"] = merged_df["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S%z")

    cols = [
        "Datetime",
        "Datetime_local",
        "incremental_merit_order",
        "decremental_merit_order",
    ]
    merged_df[cols].to_csv(output_csv_path, sep=";", index=False)
    print(f"Traitement terminé ! Fichier de sortie généré : {output_csv_path}")


def plot_merit_orders(csv_path: str, row_index: int = 0):
    """Trace les merit orders d'un quart d'heure (par défaut le premier),
    collés à l'axe central (0) : incrémental à droite, décrémental à gauche.
    """
    df = pd.read_csv(csv_path, sep=";")
    if df.empty:
        print("Le fichier CSV est vide.")
        return

    row = df.iloc[row_index]
    datetime_str = row["Datetime_local"]

    inc_dict = json.loads(row["incremental_merit_order"])
    dec_dict = json.loads(row["decremental_merit_order"])

    # --- Incrémental (côté positif) ---
    if inc_dict:
        sorted_inc = sorted((float(k), v) for k, v in inc_dict.items())
        inc_volumes = [0.0] + [k for k, _ in sorted_inc]
        raw_prices = [v[0] for _, v in sorted_inc]
        # where="pre" : y[i] s'applique sur ]x[i-1], x[i]]
        inc_prices = [raw_prices[0]] + raw_prices
    else:
        inc_volumes, inc_prices = [], []

    # --- Décrémental (côté négatif) ---
    if dec_dict:
        sorted_dec = sorted((float(k), v) for k, v in dec_dict.items())
        dec_volumes = [0.0] + [-k for k, _ in sorted_dec]
        raw_prices = [v[0] for _, v in sorted_dec]
        dec_prices = [raw_prices[0]] + raw_prices
    else:
        dec_volumes, dec_prices = [], []

    plt.figure(figsize=(14, 6))

    if inc_volumes:
        plt.step(
            inc_volumes,
            inc_prices,
            where="pre",
            label="Incrémental (Hausse / SI < 0)",
            color="crimson",
            linewidth=2,
        )

    if dec_volumes:
        plt.step(
            dec_volumes,
            dec_prices,
            where="pre",
            label="Décrémental (Baisse / SI > 0)",
            color="dodgerblue",
            linewidth=2,
        )

    plt.title(
        f"Merit Orders (jointes au centre) - Quart d'heure : {datetime_str}",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Volume cumulé (MW)   Décrémental | Incrémental", fontsize=12)
    plt.ylabel("Prix des bids (€/MWh)", fontsize=12)

    plt.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.axvline(
        0, color="black", linestyle="-", linewidth=1.2, alpha=0.8, label="Centre (SI = 0)"
    )

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


# --- EXÉCUTION ---
if __name__ == "__main__":
    FILE_A = "C:\\Document_UMONS\\Thèse-Code\\SI_side_repository\\Data\\Raw_Data_Elia\\Individual incremental balancing energy bids_05_03_25-25_10_2026.csv"
    FILE_B = "C:\\Document_UMONS\\Thèse-Code\\SI_side_repository\\Data\\Raw_Data_Elia\\Individual decremental balancing energy bids_05_03_25-25_10_2026.csv"
    OUTPUT_FILE ="C:\\Document_UMONS\\Thèse-Code\\SI_side_repository\\Data\\merit_order_dictionaries.csv"

    process_merit_order(FILE_A, FILE_B, OUTPUT_FILE)
    plot_merit_orders(OUTPUT_FILE, row_index=0)


# # --- EXÉCUTION ---
# if __name__ == "__main__":
#     FILE_A = "C:\\Document_UMONS\\Thèse-Code\\SI_side_repository\\Data\\Raw_Data_Elia\\Individual incremental balancing energy bids_05_03_25-25_10_2026.csv"
#     FILE_B = "C:\\Document_UMONS\\Thèse-Code\\SI_side_repository\\Data\\Raw_Data_Elia\\Individual decremental balancing energy bids_05_03_25-25_10_2026.csv"
#     OUTPUT_FILE ="C:\\Document_UMONS\\Thèse-Code\\SI_side_repository\\Data\\merit_order_dictionaries.csv"

#     process_merit_order(FILE_A, FILE_B, OUTPUT_FILE)
#     plot_merit_orders(OUTPUT_FILE, row_index=0)