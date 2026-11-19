import os
os.environ["PANDAPOWER_NUMBA"] = "False"
import sys
import gc
import numpy as np
import pandas as pd  # type: ignore

Result_dir = "C:/Document_UMONS/Thèse-Code/SI_side_repository"
Environment_dir = "C:/Document_UMONS/Thèse-Code/System_Imbalance/System-Imbalance/Minute/Environment_min"
sys.path.append(Result_dir)
sys.path.append(Environment_dir)

from Environment_min_initial import Environment, importdata  # type:ignore

LOCAL_TZ = "Europe/Brussels"
DATA_DIR = "C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Split_initial"

# %% ------- Prix de l'agent (action nulle = prix sans batterie) -------
path_training = os.path.normpath(f"{DATA_DIR}/Dataset_Train_2025_2026_Current_SI.csv")
path_validation = os.path.normpath(f"{DATA_DIR}/Dataset_Val_2025_2026_Current_SI.csv")
path_test = os.path.normpath(f"{DATA_DIR}/Dataset_Test_2025_2026_Current_SI.csv")
path_merit_order = os.path.normpath("C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/merit_parsed.pkl")

print("Importing data + preprocessing...")
_, _, df_test, merit_order_dict, mean, std = importdata(
    path_training, path_validation, path_test, path_merit_order)
print("Preprocessed data imported")

nb_quarters_per_episode = 96
battery_max_power = 20
EP_ratio = 1
roundtrip_efficiency = 0.9
battery_replacement_cost = 3e+5 * battery_max_power * EP_ratio

env = Environment(df_test, mean, std, merit_order_dict, battery_max_power,
                  1, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency,
                  battery_replacement_cost, penalty=0)

env._episode = 0
dt_arr = env.df_system['datetime'].to_numpy()      # précalculé : évite .loc dans la boucle

rows = []
prev_miss, prev_empty = env.merit_miss, env.merit_empty
for _ in range(env.n_days):
    state = env.reset()
    done = False
    while not done:
        idx = env._current_idx
        is_end_qh = (env._min_in_qh_arr[idx] == 14)
        if is_end_qh:
            ts_end = dt_arr[idx]
            # SI moyen du QH (sans batterie, car action = 0) : sert au découpage par régime
            si_mean = float(env._si_arr[idx - 14: idx + 1].mean())

        state, r, action, done = env.step(0.0)

        if is_end_qh:
            d_miss = env.merit_miss - prev_miss
            d_empty = env.merit_empty - prev_empty
            prev_miss, prev_empty = env.merit_miss, env.merit_empty
            price = env.qh_prices[-1]
            rows.append({
                'datetime_end': ts_end,
                'price_agent': price,
                'si_mean': si_mean,
                # 1 si le merit order manquait / a été remplacé par le dernier valide
                'merit_fallback': (d_miss + d_empty) > 0,
                # le repli final de _calculate_quarter_price renvoie exactement 0.0
                'price_zero': price == 0.0,
            })
    gc.collect()

assert len(rows) == len(env.qh_prices), "timestamps et prix désalignés"

df_agent = pd.DataFrame(rows)
df_agent['datetime_end'] = pd.to_datetime(df_agent['datetime_end'])
df_agent['qh'] = df_agent['datetime_end'].dt.floor('15min')
df_agent['suspect'] = df_agent['merit_fallback'] | df_agent['price_zero']
print(f"{len(df_agent)} QH agent | {df_agent['suspect'].sum()} QH 'suspects' "
      f"(merit de repli ou prix 0.0)")


# %% ------- Données Elia -------
def load_elia(path):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df['Datetime'] = (pd.to_datetime(df['Datetime'], utc=True)
                      .dt.tz_convert(LOCAL_TZ)
                      .dt.tz_localize(None))
    return df


def drop_dups(df, name):
    n0 = len(df)
    df = df.drop_duplicates('qh', keep=False)
    print(f"{name}: {n0} lignes -> {len(df)} "
          f"({n0 - len(df)} retirées : doublons, ex. changement d'heure)")
    return df


path_Elia_min = os.path.normpath( "C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Raw_Data_Elia/Imbalance price per minute_05_03_25-25_10_2026.csv")
df_elia_min = load_elia(path_Elia_min)
df_elia_min = df_elia_min[df_elia_min['Datetime'].dt.minute % 15 == 14].copy()
df_elia_min['qh'] = df_elia_min['Datetime'].dt.floor('15min')
df_elia_min = (df_elia_min[['qh', 'Imbalance Price', 'Quality status']]
               .rename(columns={'Imbalance Price': 'price_elia_min',
                                'Quality status': 'quality_min'}))
df_elia_min = drop_dups(df_elia_min, "Elia minute")

path_Elia_quarter = os.path.normpath("C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Raw_Data_Elia/Imbalance prices per quarter-hour_05_03_25-25_10_2026.csv")
df_elia_q = load_elia(path_Elia_quarter)
df_elia_q['qh'] = df_elia_q['Datetime']
df_elia_q = (df_elia_q[['qh', 'Imbalance Price', 'Quality status']]
             .rename(columns={'Imbalance Price': 'price_elia_quarter',
                              'Quality status': 'quality_quarter'}))
df_elia_q = drop_dups(df_elia_q, "Elia QH")


# %% ------- Fusion avec diagnostic des pertes -------
print("\nAlignement et fusion des données...")
for name, df_ref in [("Elia QH", df_elia_q), ("Elia minute", df_elia_min)]:
    missing = ~df_agent['qh'].isin(df_ref['qh'])
    days = sorted(df_agent.loc[missing, 'qh'].dt.date.unique())
    print(f"QH agent absents de {name} : {missing.sum()}"
          + (f" | jours concernés : {days[:10]}{' ...' if len(days) > 10 else ''}"
             if missing.any() else ""))

df_comparison = df_agent.merge(df_elia_q, on='qh', how='inner')
print(f"Après fusion Elia QH     : {len(df_comparison)} lignes")
df_comparison = df_comparison.merge(df_elia_min, on='qh', how='inner')
print(f"Après fusion Elia minute : {len(df_comparison)} lignes")
df_comparison = df_comparison.sort_values('qh').reset_index(drop=True)

if len(df_comparison) == 0:
    raise ValueError("Aucune ligne alignée : vérifier le fuseau horaire / les dates.")

# Nettoyage des NaN avant métriques (sinon les % sont biaisés)
cols_needed = ['price_agent', 'price_elia_quarter', 'price_elia_min']
n0 = len(df_comparison)
df_comparison = df_comparison.dropna(subset=cols_needed).reset_index(drop=True)
print(f"NaN retirés : {n0 - len(df_comparison)} | lignes finales : {len(df_comparison)}")

df_comparison['err_vs_quarter'] = df_comparison['price_agent'] - df_comparison['price_elia_quarter']
df_comparison['err_vs_min'] = df_comparison['price_agent'] - df_comparison['price_elia_min']

# Régime de prix (mêmes seuils que _calculate_quarter_price)
si = df_comparison['si_mean']
df_comparison['regime'] = np.select(
    [si < -150, si < -25, si > 150, si > 25],
    ['MIP fort (SI<-150)', 'MIP (-150<SI<-25)', 'MDP fort (SI>150)', 'MDP (25<SI<150)'],
    default='zone morte (|SI|<25)')


# %% ------- Métriques -------
def metrics(err, ref, pred):
    if len(err) == 0:
        return {k: np.nan for k in ['N', 'MAE', 'Médiane |err|', 'RMSE', 'Biais',
                                    'Pearson', 'Spearman', '% |err|<1€',
                                    '% |err|<10€', '% |err|<50€', '% |err|<100€']}
    ae = err.abs()
    return {
        'N': len(err),
        'MAE': ae.mean(),
        'Médiane |err|': ae.median(),
        'RMSE': float(np.sqrt((err ** 2).mean())),
        'Biais': err.mean(),
        'Pearson': pred.corr(ref),
        'Spearman': pred.rank().corr(ref.rank()),
        '% |err|<1€': (ae < 1).mean() * 100,
        '% |err|<10€': (ae < 10).mean() * 100,
        '% |err|<50€': (ae < 50).mean() * 100,
        '% |err|<100€': (ae < 100).mean() * 100,
    }


def summary_table(df):
    return pd.DataFrame({
        'vs Elia QH': metrics(df['err_vs_quarter'], df['price_elia_quarter'], df['price_agent']),
        'vs Elia min (dern. minute)': metrics(df['err_vs_min'], df['price_elia_min'], df['price_agent']),
    })


pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)

df_clean = df_comparison[~df_comparison['suspect']]

print("\n===== 1) Tous les QH =====")
print(summary_table(df_comparison).round(3))

print("\n===== 2) Sans QH suspects (merit de repli / prix 0.0) =====")
print(summary_table(df_clean).round(3))

# Contrôle interne Elia
diff_elia = (df_comparison['price_elia_min'] - df_comparison['price_elia_quarter']).abs()
print(f"\nElia min vs Elia QH : {(diff_elia < 1e-6).mean() * 100:.2f} % identiques")
print(f"Elia min vs Elia QH : {(diff_elia < 5).mean() * 100:.2f} % identiques à 5€ près")

# Par statut de qualité Elia (prix QH validé vs temps réel)
print("\n===== 3) Par statut de qualité Elia (QH), hors QH suspects =====")
for status, g in df_clean.groupby('quality_quarter', dropna=False):
    m = metrics(g['err_vs_quarter'], g['price_elia_quarter'], g['price_agent'])
    print(f"\n[{status}]")
    print(pd.Series(m).round(3).to_string())

# Par régime de prix
print("\n===== 4) Par régime (vs Elia QH), hors QH suspects =====")
by_regime = pd.DataFrame({
    reg: metrics(g['err_vs_quarter'], g['price_elia_quarter'], g['price_agent'])
    for reg, g in df_clean.groupby('regime')
}).T
print(by_regime.round(3))


# %% ------- Test de décalage temporel (alignement du dataset) -------
print("\n===== 5) Test de décalage (agent vs Elia QH décalé de k quarts d'heure) =====")
elia_q_s = df_elia_q.set_index('qh')['price_elia_quarter']
ag = df_agent[~df_agent['suspect']].copy()
res_shift = []
for k in range(-2, 3):
    ref = pd.Series(elia_q_s.reindex(ag['qh'] + pd.Timedelta(minutes=15 * k)).to_numpy(),
                    index=ag.index)
    ok = ref.notna()
    err = (ag.loc[ok, 'price_agent'] - ref[ok])
    res_shift.append({
        'k (QH)': k,
        'N': int(ok.sum()),
        'MAE': err.abs().mean(),
        'Pearson': ag.loc[ok, 'price_agent'].corr(ref[ok]),
    })
df_shift = pd.DataFrame(res_shift).set_index('k (QH)')
print(df_shift.round(3))
best_k = df_shift['MAE'].idxmin()
print(f"Meilleur décalage (MAE minimale) : k = {best_k}"
      + ("  -> OK, alignement correct" if best_k == 0
         else "  -> ATTENTION : décalage probable entre le dataset et Elia"))


# %% ------- Pires écarts -------
print("\n===== 6) 15 pires écarts vs Elia QH =====")
worst = df_comparison.reindex(
    df_comparison['err_vs_quarter'].abs().sort_values(ascending=False).index)
print(worst[['qh', 'price_agent', 'price_elia_quarter', 'price_elia_min', 'si_mean',
             'regime', 'suspect', 'quality_quarter']].head(15))

df_comparison.to_csv(f"{DATA_DIR}/comparaison_prix.csv", sep=";", index=False)
print("\nFichier sauvegardé : comparaison_prix.csv")
