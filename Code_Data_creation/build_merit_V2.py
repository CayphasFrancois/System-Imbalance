import json
import pickle

import numpy as np
import pandas as pd

MERIT_CSV = r"C:\Document_UMONS\Thèse-Code\SI_side_repository\Data\merit_order_dictionaries.csv"
OUT_PKL = r"C:\Document_UMONS\Thèse-Code\SI_side_repository\Data\merit_parsed.pkl"

# "local" : heure belge naïve (colonne Datetime_local), comme df_system si sa
#           colonne 'datetime' est en heure belge.
# "utc"   : UTC naïf (colonne Datetime), si df_system est en UTC.
KEY_MODE = "local"

# Optionnel : chemin d'un CSV contenant la colonne 'datetime' de df_system
# (train, val ou test), pour vérifier que les clés s'alignent avec l'environnement.
SYSTEM_CSV = None  # ex. r"C:\...\df_train.csv"


def make_keys(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "local":
        dt = pd.to_datetime(df["Datetime_local"])
    elif mode == "utc":
        dt = pd.to_datetime(df["Datetime"], utc=True).dt.tz_localize(None)
    else:
        raise ValueError(f"KEY_MODE inconnu : {mode}")
    return dt.astype("datetime64[ns]").astype("int64")


def parse_side(s: str, row: int, problems: list) -> dict:
    d = json.loads(s)  # JSON valide : une erreur de parsing doit lever directement
    if not d:
        return {}
    out = {float(k): (float(v[0]), int(v[1])) for k, v in d.items()}
    vols = list(out.keys())
    if any(b <= a for a, b in zip(vols, vols[1:])):
        problems.append((row, "volumes non strictement croissants"))
    flags = [v[1] for v in out.values()]
    if not set(flags) <= {1, 2}:
        problems.append((row, f"ordres inattendus : {set(flags) - {1, 2}}"))
    if any(b < a for a, b in zip(flags, flags[1:])):
        problems.append((row, "mFRR avant aFRR"))
    return out


def main():
    df = pd.read_csv(MERIT_CSV, sep=";")

    # Tri chronologique sur l'UTC (non ambigu), avant toute conversion locale
    df["_utc"] = pd.to_datetime(df["Datetime"], utc=True)
    df = df.sort_values("_utc").reset_index(drop=True)

    # Vérification d'alignement avec df_system (si fourni)
    if SYSTEM_CSV:
        sysdf = pd.read_csv(SYSTEM_CSV, sep=";")
        sys_keys = set(
            pd.to_datetime(sysdf["datetime"])
            .dt.floor("15min")
            .astype("datetime64[ns]")
            .astype("int64")
        )
        for mode in ("local", "utc"):
            cov = len(sys_keys & set(make_keys(df, mode))) / len(sys_keys)
            print(f"Couverture de df_system avec clés '{mode}' : {cov:.2%}")

    df["_key"] = make_keys(df, KEY_MODE)

    # Clés dupliquées : n'arrive qu'en mode 'local' (heure 02:00-03:00 du
    # passage à l'heure d'hiver). On garde la 1re occurrence (heure d'été).
    dup = df["_key"].duplicated(keep="first")
    print(f"Clés dupliquées supprimées : {int(dup.sum())}")
    df = df[~dup]

    cache, problems = {}, []
    for n, (k, a, b) in enumerate(
        zip(df["_key"], df["incremental_merit_order"], df["decremental_merit_order"])
    ):
        cache[int(k)] = (parse_side(a, n, problems), parse_side(b, n, problems))

    n_inc_empty = sum(1 for i, _ in cache.values() if not i)
    n_dec_empty = sum(1 for _, d in cache.values() if not d)
    print(f"Quarts d'heure : {len(cache)}")
    print(f"  incrémental vide : {n_inc_empty} | décrémental vide : {n_dec_empty}")
    print(f"  problèmes de format : {len(problems)}")
    for p in problems[:10]:
        print("   ", p)
    if problems:
        raise RuntimeError("Merit orders invalides : voir les problèmes ci-dessus")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(cache, f)
    print("done :", OUT_PKL)


if __name__ == "__main__":
    main()