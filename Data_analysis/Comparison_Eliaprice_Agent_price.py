import os
os.environ["PANDAPOWER_NUMBA"] = "False"
import numpy as np
import pandas as pd  # type: ignore
import sys
import gc

Result_dir = r"C:\\Document_UMONS\\Thèse-Code\\SI_side_repository"
Environment_dir = r"C:\\Document_UMONS\\Thèse-Code\\System_Imbalance\\System-Imbalance\\Environment_quarter"
sys.path.append(Result_dir)
sys.path.append(Environment_dir)

from Environment_quarter_initial import Environment,importdata #type:ignore

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
Price_agent = env.qh_prices

# Création d'un DataFrame de référence pour l'agent
df_agent_results = pd.DataFrame({
    'datetime': pd.to_datetime(qh_timestamps),
    'price_agent': Price_agent})

#%% Price Elia Imbalance Price Min

path_Elia_data_min = os.path.normpath("C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Raw_Data_Elia/Imbalance price per minute_05_03_25-25_10_2026.csv")
df_Elia_min= pd.read_csv(path_Elia_data_min, sep=";") if path_Elia_data_min.endswith('.csv') else pd.read_excel(path_Elia_data_min)

df_Elia_min['datetime'] = pd.to_datetime(df_Elia_min['Datetime'])

# Filtrage pour ne garder que la 15ème minute (fin de chaque quart d'heure, ex: minutes 00, 15, 30, 45 selon le format exact d'Elia)
# Vous pouvez ajuster les valeurs de minutes si le format d'Elia utilise 15, 30, 45, 60
df_Elia_min_filtered = df_Elia_min[df_Elia_min['datetime'].dt.minute.isin([0, 15, 30, 45])].copy()

#%% Price Elia Imbalance Price Quarter

path_Elia_data_quarter = os.path.normpath("C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Raw_Data_Elia/Imbalance price per quarter-hour_05_03_25-25_10_2026.csv")
df_Elia_quarter= pd.read_csv(path_Elia_data_quarter, sep=";") if path_Elia_data_quarter.endswith('.csv') else pd.read_excel(path_Elia_data_quarter)

df_Elia_quarter['datetime'] = pd.to_datetime(df_Elia_quarter['Datetime'])

#%% Alignement et fusion des données 

print("Alignement et fusion des données...")

# Fusion des résultats de l'agent avec les données Elia quart-heure
df_comparison = pd.merge(
    df_agent_results, 
    df_Elia_quarter[['datetime', 'Imbalance Price']], # Remplacer 'Imbalance Price' par le nom exact de la colonne de prix dans le CSV Elia si nécessaire
    on='datetime', 
    how='inner',
    suffixes=('', '_elia_quarter')
)

# Fusion avec les données Elia minute filtrées
df_comparison = pd.merge(
    df_comparison, 
    df_Elia_min_filtered[['datetime', 'Imbalance Price']], 
    on='datetime', 
    how='inner',
    suffixes=('_agent', '_elia_min')
)

print(f"Comparaison prête ! {len(df_comparison)} lignes alignées avec succès.")
print(df_comparison.head())
