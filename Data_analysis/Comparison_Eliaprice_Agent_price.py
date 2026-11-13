import os
os.environ["PANDAPOWER_NUMBA"] = "False"
import numpy as np
import pandas as pd  # type: ignore
import sys
import gc

Result_dir = r"C:\\Document_UMONS\\Thèse-Code\\SI_side_repository"
Environment_dir = r"C:\\Document_UMONS\\Thèse-Code\\System_Imbalance\\System-Imbalance\\Environment_min"
sys.path.append(Result_dir)
sys.path.append(Environment_dir)

from Environment_min_initial import Environment,importdata #type:ignore

LOCAL_TZ = "Europe/Brussels"

#%% Price Agent
path_training = os.path.normpath('C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Dataset_Train_2025_2026_Current_SI.csv')
path_validation = os.path.normpath('C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Dataset_Val_2025_2026_Current_SI.csv')
path_test = os.path.normpath('C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Dataset_Test_2025_2026_Current_SI.csv')
path_merit_order = os.path.normpath('C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/merit_parsed.pkl')

print('Importing data + preprocessing (PyTorch Pipeline)...')
_,  _, df_test, merit_order_dict, mean,std = importdata(path_training, path_validation, path_test, path_merit_order)   
print('Preprocessed data imported')

nb_quarters_per_episode = 96
battery_max_power = 20
EP_ratio = 1
roundtrip_efficiency = 0.9
battery_replacement_cost = 3e+5 * battery_max_power * EP_ratio

env = Environment(df_test,mean,std, merit_order_dict, battery_max_power,
                                1, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency,
                                battery_replacement_cost,penalty=0) 

env._episode = 0                      
num_episodes = env.n_days 
qh_timestamps = []
for _ in range(num_episodes):
    state = env.reset()
    done= False
    while not done:
        if env._min_in_qh_arr[env._current_idx] == 14:
            qh_timestamps.append(env.df_system.loc[env._current_idx, 'datetime'])
        a = 0
        state, r, action, done = env.step(a)  
    gc.collect()

assert len(qh_timestamps) == len(env.qh_prices), "timestamps et prix désalignés"

# Création d'un DataFrame de référence pour l'agent
df_agent = pd.DataFrame({
    'datetime_end': pd.to_datetime(qh_timestamps),
    'price_agent': env.qh_prices,
})
df_agent['qh'] = df_agent['datetime_end'].dt.floor('15min')
# %% ------- Données Elia -------
def load_elia(path):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    # Elia fournit des offsets (+01:00/+02:00) : on convertit en heure locale belge,
    # comme le dataset de l'environnement, puis on retire le fuseau.
    df['Datetime'] = (pd.to_datetime(df['Datetime'], utc=True)
                        .dt.tz_convert(LOCAL_TZ)
                        .dt.tz_localize(None))
    return df
 
 
# Prix par minute : le prix "final" du QH est celui de la dernière minute (minute % 15 == 14)
path_Elia_min = os.path.normpath("C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Raw_Data_Elia/Imbalance price per minute_05_03_25-25_10_2026.csv")
df_elia_min = load_elia(path_Elia_min)
df_elia_min = df_elia_min[df_elia_min['Datetime'].dt.minute % 15 == 14].copy()
df_elia_min['qh'] = df_elia_min['Datetime'].dt.floor('15min')
df_elia_min = (df_elia_min[['qh', 'Imbalance Price', 'Quality status']]
               .rename(columns={'Imbalance Price': 'price_elia_min',
                                'Quality status': 'quality_min'})
               .drop_duplicates('qh', keep=False))
 
# Prix par quart d'heure : Datetime = début du QH
path_Elia_quarter = os.path.normpath("C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Raw_Data_Elia/Imbalance prices per quarter-hour_05_03_25-25_10_2026.csv")
df_elia_q = load_elia(path_Elia_quarter)
df_elia_q['qh'] = df_elia_q['Datetime']
df_elia_q = (df_elia_q[['qh', 'Imbalance Price', 'Quality status']]
             .rename(columns={'Imbalance Price': 'price_elia_quarter',
                              'Quality status': 'quality_quarter'})
             .drop_duplicates('qh', keep=False))
 
# %% ------- Fusion -------
print("Alignement et fusion des données...")
df_comparison = (df_agent
                 .merge(df_elia_q, on='qh', how='inner')
                 .merge(df_elia_min, on='qh', how='inner')
                 .sort_values('qh').reset_index(drop=True))
 
print(f"{len(df_agent)} QH agent | {len(df_comparison)} lignes alignées")
if len(df_comparison) == 0:
    raise ValueError("Aucune ligne alignée : vérifier le fuseau horaire / les dates.")
 
# %% ------- Métriques -------
df_comparison['err_vs_quarter'] = df_comparison['price_agent'] - df_comparison['price_elia_quarter']
df_comparison['err_vs_min'] = df_comparison['price_agent'] - df_comparison['price_elia_min']
 
 
def metrics(err, ref, pred):
    return {
        'MAE': err.abs().mean(),
        'RMSE': float(np.sqrt((err ** 2).mean())),
        'Biais': err.mean(),
        'Corr': pred.corr(ref),
        '% |err|<1€': (err.abs() < 1).mean() * 100,
    }
 
 
summary = pd.DataFrame({
    'vs Elia QH': metrics(df_comparison['err_vs_quarter'], df_comparison['price_elia_quarter'], df_comparison['price_agent']),
    'vs Elia min (dernière minute)': metrics(df_comparison['err_vs_min'], df_comparison['price_elia_min'], df_comparison['price_agent']),
})
print(summary.round(3))
 
# Vérification interne côté Elia : le prix de la dernière minute doit égaler le prix du QH
diff_elia = (df_comparison['price_elia_min'] - df_comparison['price_elia_quarter']).abs()
print(f"Elia min vs Elia QH : {(diff_elia < 1e-6).mean() * 100:.2f} % identiques")
 
# Pires écarts, utiles pour déboguer l'environnement
print(df_comparison.reindex(df_comparison['err_vs_quarter'].abs().sort_values(ascending=False).index)
      [['qh', 'price_agent', 'price_elia_quarter', 'price_elia_min', 'quality_quarter']].head(15))
 
print(df_comparison.head())