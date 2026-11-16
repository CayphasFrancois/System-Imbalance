import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # type: ignore
import torch as T
import random
import time
import os
import sys
import shutil 
import gc 
from tqdm import tqdm # type: ignore
from datetime import datetime
import faulthandler, psutil

proc = psutil.Process(os.getpid())
Result_dir = r"C:\\Document_UMONS\\Thèse-Code\\SI_side_repository"
Environment_dir = r"C:\\Document_UMONS\\Thèse-Code\\System_Imbalance\\System-Imbalance\\Environment_quarter"
sys.path.append(Environment_dir)
faulthandler.enable(open(os.path.join(Result_dir, "crash.log"), "ab"))

# --- IMPORTATION DU PIPELINE PYTORCH (Environnements & Prétraitement) ---
from Environment_quarter_initial import Environment, importdata #type:ignore
from TD3_training import evaluate_agent
from TD3_torch import Agent

# %%=============================================================================
# 1. TARGET DIRECTORIES
# ==============================================================================
DOSSIER_RUN = r"C:\\Document_UMONS\\Thèse-Code\\SI_side_repository\\Runs\\Quarter\\Initial\\Run_TD3_PyTorch_quarter_20261113_112007_EARLY_STOP_S1_TEST"  #Fichier où est stocké le modèle sauvegardé 
TXT_PATH = os.path.join(DOSSIER_RUN, "hyperparameters.txt")

# %% =============================================================================
# 2. AUTOMATIC HYPERPARAMETER READING (.txt)
# =============================================================================
print("Reading hyperparameters from text file...\n")
hyperparams = {}
if os.path.exists(TXT_PATH):
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                parts = line.split(":", 1)
                key = parts[0].replace("-", "").strip()
                value = parts[1].strip()
                hyperparams[key] = value

nb_quarters_per_episode = 96

# %%=============================================================================
# 3. PHYSICAL PARAMETERS
# ==============================================================================
battery_max_power = 20
EP_ratio = 1
roundtrip_efficiency = 0.9
battery_replacement_cost = 3e+5 * battery_max_power * EP_ratio

path_training = os.path.normpath('C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Dataset_Train_2025_2026_Current_SI.csv')
path_validation = os.path.normpath('C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Dataset_Val_2025_2026_Current_SI.csv')
path_test = os.path.normpath('C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Dataset_Test_2025_2026_Current_SI.csv')
path_merit_order = os.path.normpath('C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/merit_parsed.pkl')
# %% =============================================================================
# 4. DATA IMPORT AND MIDNIGHT ANCHOR CONFIGURATIONS
# =============================================================================
print("\nImporting data (Pipeline)...")
df_train,  _, df_test, merit_order_dict, mean,std= importdata(path_training, path_validation, path_test, path_merit_order)

# %% =============================================================================
# 5. ENVIRONMENT CREATION AND PYTORCH AGENT INSTANCIATION
# =============================================================================
print("\nCreating Evaluation environment...")
env_eval = Environment(
                        df_test,mean,std, merit_order_dict, battery_max_power,
                        1, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency,
                        battery_replacement_cost,penalty=0
                        )

input_dims = (19,)
n_actions = 1

# Instanciation et chargement des poids du modèle sauvegardé
agent = Agent(
    alpha=1e-4, beta=1e-4, input_dims=input_dims, env=None, tau=5e-3, gamma=0.99,
    n_actions=n_actions, max_size=int(1), layer1_size=500, layer2_size=500,
    batch_size=256, filename=DOSSIER_RUN
)
agent.load_models()
agent.actor.eval()

# %% =============================================================================
# 6. DETAILED SIMULATION OF SPECIFIC DAYS (Search by DATE)
# =============================================================================
# jours_a_simuler = ['2019-06-21', '2019-06-22', '2019-06-23', '2018-11-21', '2018-11-22']

# for date_str in jours_a_simuler:
#     print(f" -> Plotting day {date_str}...")
#     lignes_date = df_dates[df_dates['FROM_DATE'].dt.strftime('%Y-%m-%d') == date_str]
#     if lignes_date.empty: 
#         continue

#     target_index = lignes_date.index[0] - window_size
#     if target_index < 0: 
#         continue

#     # Calibration de l'environnement PyTorch
#     env_eval._observation_index = target_index
#     env_eval._episode = (target_index // nb_quarters_per_episode) + 1
#     env_eval._observation_samples[target_index, env_eval._SOC_index] = 0.5
#     env_eval._episode_ended = False
#     state = env_eval.reset()

#     actions_list, soc_list, profit_cumul = [], [], []
#     price_taker_list, price_maker_list = [], []
#     nrv_initial_list, nrv_modified_list = [], []
#     current_profit = 0.0

#     for _ in range(nb_quarters_per_episode):
#         idx = env_eval._observation_index
#         soc = state[env_eval._SOC_index]

#         base_nrv = env_eval._non_observable_samples[idx, 0]
#         base_price = env_eval._non_observable_samples[idx, 1]

#         # Action déterministe via l'acteur PyTorch
#         state_tensor = T.tensor(state, dtype=T.float).unsqueeze(0).to(agent.actor.device)
#         with T.no_grad():
#             raw_action = agent.actor(state_tensor).cpu().numpy()[0][0]

#         new_soc = np.clip(soc + (raw_action / (4 * EP_ratio)), 0.0, 1.0)
#         fraction_power = (new_soc - soc) * (4 * EP_ratio)
#         bat_MW = fraction_power * battery_max_power
#         net_MW = bat_MW / np.sqrt(roundtrip_efficiency) if bat_MW > 0 else bat_MW * np.sqrt(roundtrip_efficiency)

#         modified_nrv = base_nrv + net_MW
#         idx_raw = (modified_nrv / 100) + 6
#         idx_mod = int(np.floor(idx_raw)) if idx_raw < 6 else int(np.ceil(idx_raw))
#         idx_mod = int(np.clip(idx_mod, 0, 12))

#         realized_price = env_eval._non_observable_samples[idx, 2 + idx_mod]

#         actions_list.append(raw_action)
#         soc_list.append(soc)
#         nrv_initial_list.append(base_nrv)
#         nrv_modified_list.append(modified_nrv)
#         price_taker_list.append(base_price)
#         price_maker_list.append(realized_price)

#         state, reward, done = env_eval.step(raw_action)
#         current_profit += reward
#         profit_cumul.append(current_profit)

    # --- PLOTTING (CRIMSON RED THEME) ---
    # fig, (ax_price, ax_action_si, ax_soc, ax_profit) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    # x_axis = np.arange(nb_quarters_per_episode) / 4

    # ax_price.plot(x_axis, price_taker_list, color='gray', linestyle='--', label='Price Taker (Base)', alpha=0.7)
    # ax_price.plot(x_axis, price_maker_list, color='crimson', linewidth=2, label='Price Maker (Impacted)')
    # ax_price.set_ylabel("Price (€/MWh)")
    # ax_price.set_title(f"Market Price Impact - {date_str} (TD3 PyTorch)")
    # ax_price.legend(loc="upper left"); ax_price.grid(True, alpha=0.5)

    # ax_action_si.plot(x_axis, nrv_initial_list, color='black', linestyle='--', label='Initial SI', alpha=0.4)
    # ax_action_si.plot(x_axis, nrv_modified_list, color='crimson', label='Modified SI', linewidth=1.5, alpha=0.8)
    # ax_action_si.axhline(0, color='black', linewidth=0.8)
    # ax_action_si.set_ylabel("SI (MW)", color='crimson'); ax_action_si.tick_params(axis='y', labelcolor='crimson')

    # ax_action_twin = ax_action_si.twinx()
    # ax_action_twin.bar(x_axis, actions_list, width=0.15, color=['red' if a < 0 else 'green' for a in actions_list], alpha=0.4, label='Agent Action')
    # ax_action_twin.set_ylabel("Action (-1 to 1)", color='green'); ax_action_twin.set_ylim(-1.1, 1.1); ax_action_twin.tick_params(axis='y', labelcolor='green')

    # ax_soc.plot(x_axis, soc_list, color='crimson', linewidth=2)
    # ax_soc.set_ylabel("SoC (0 to 1)"); ax_soc.set_ylim(-0.1, 1.1)
    # ax_soc.axhline(0, color='black', linestyle='--', alpha=0.5); ax_soc.axhline(1, color='black', linestyle='--', alpha=0.5)
    # ax_soc.set_title("Battery Level (SoC)"); ax_soc.grid(True, alpha=0.5)

    # ax_profit.plot(x_axis, profit_cumul, color='goldenrod', linewidth=2)
    # ax_profit.set_ylabel("Cumulative Profit (€)"); ax_profit.set_xlabel("Time (Hours)")
    # ax_profit.set_title("Daily Profitability"); ax_profit.grid(True, alpha=0.5)

    # plt.tight_layout()
    # fig.savefig(os.path.join(DOSSIER_RUN, f"Strategy_Day_{date_str}_TD3_PyTorch.png"), dpi=300)
    # plt.close(fig)

# %% =============================================================================
# 7. FULL DATASET GLOBAL EVALUATION (SYNCHRONIZED MIDNIGHT ANCHOR)
# =============================================================================

num_episodes = env_eval.n_days

episode_profit_list=[]

average_return,episode_profit_list,state_reward_action = evaluate_agent(env_eval,agent)
total_profit = average_return * num_episodes


#%% Stockage des résultats dans un fichier texte

chemin_resultats = os.path.join(DOSSIER_RUN, "Test_Evaluation_Results.txt")
with open(chemin_resultats, "w", encoding="utf-8") as f:
    f.write("="*50 + "\n")
    f.write(f"Number of tested days: {num_episodes} days\n")
    f.write(f"Total Cumulative Profit: {total_profit:,.2f} €\n")
    f.write(f"Daily Average: {average_return:,.2f} € / day\n")
    f.write("="*50 + "\n")

def day_slice_SI(env, k):
    s = int(env._day_starts[k])
    x = env.df_system['system imbalance'].iloc[s : s + 1440].to_numpy(dtype=np.float64)
    y = x.reshape(96, 15).mean(axis=1)
    return y
def day_slice_NRV(env, k):
    s = int(env._day_starts[k])
    x = env.df_system['nrv_new'].iloc[s : s + 1440].to_numpy(dtype=np.float64)
    y = x.reshape(96, 15).mean(axis=1)
    return y

#%% Graphique Profit/jour
fig_f, ax_f = plt.subplots(figsize=(12, 6))
ax_f.plot(range(num_episodes), episode_profit_list, color='darkorange', linewidth=2.5)
ax_f.fill_between(range(num_episodes), episode_profit_list, color='navajowhite', alpha=0.3)
ax_f.set_title("Average Financial Performance per day on Dataset (TD3)", fontsize=14, fontweight='bold')
ax_f.set_xlabel("Days"); ax_f.set_ylabel("Profit/day (€)"); ax_f.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
fig_f.savefig(os.path.join(DOSSIER_RUN, "Profit_per_day.png"), dpi=300)
plt.close(fig_f)
print(f"\nFINAL RESULT: {total_profit:,.2f} € | Avg: {average_return:,.2f} €/day")

#%% Graphique Profit cumulé
cumulated_profit = []
sum_profit= 0 
for i in range (len(episode_profit_list)+1): 
    cumulated_profit.append(sum_profit)
    if i <= len(episode_profit_list)-1:
        sum_profit+=episode_profit_list[i]

fig_f, ax_f = plt.subplots(figsize=(12, 6))
ax_f.plot(range(len(cumulated_profit)), cumulated_profit, color='darkorange', linewidth=2.5)
ax_f.fill_between(range(len(cumulated_profit)), cumulated_profit, color='navajowhite', alpha=0.3)
ax_f.set_title("Cumulative Financial Performance on Dataset (TD3)", fontsize=14, fontweight='bold')
ax_f.set_xlabel("Days"); ax_f.set_ylabel("Total Profit (€)"); ax_f.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
fig_f.savefig(os.path.join(DOSSIER_RUN, "Global_Results.png"), dpi=300)
plt.close(fig_f) 
# %% Graphique SI/Action

# x_axis = np.arange(3*96)
# si_data = env.df_system['system imbalance'].to_numpy(dtype=np.float64)
# si_plot = np.concatenate ([si_data[28*96:29*96], si_data[41*96:42*96], si_data[75*96:76*96]])  
# action_data = [item[2] for item in state_reward_action]
# action_plot = np.concatenate ([action_data[28*96:29*96], action_data[41*96:42*96], action_data[75*96:76*96]]).flatten()

x_axis = np.arange(96)
si_plot = day_slice_SI(env_eval, 28)
action_data = [item[2] for item in state_reward_action]
action_plot = np.concatenate ([action_data[28*96:29*96]]).flatten()

fig, ax_si = plt.subplots(figsize=(12, 4.5))
ax_si.plot(x_axis, si_plot, color='blue', linewidth=1.5, label='SI (MW)')
ax_si.set_ylabel("SI [MW]", color='blue', fontweight='bold')
ax_si.tick_params(axis='y', labelcolor='blue')
max_si = max(abs(si_plot.min()), abs(si_plot.max()))
marge = max_si * 0.1
ax_si.set_ylim(-(max_si + marge), max_si + marge)
ax_si.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.7)  # Ligne zéro

ax_action = ax_si.twinx()
bar_colors = ['green' if a >= 0 else 'gray' for a in action_plot]

ax_action.bar(x_axis, action_plot, width=0.15, color=bar_colors, alpha=0.4, label='Agent Action')
ax_action.set_ylabel("Action [-1, 1]", color='green', fontweight='bold')
ax_action.tick_params(axis='y', labelcolor='green')
ax_action.set_ylim(-1.1, 1.1)  # Fixer les limites de l'action

plt.title("SI - Action on Dataset (TD3) ", fontsize=14, fontweight='bold', pad=12)
ax_si.set_xlabel("Steps")
ax_si.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

plt.savefig(os.path.join(DOSSIER_RUN, "SI - Action on Dataset (TD3) .png"), dpi=300)
plt.show()

#%% Graphique SoC 

SoC_data = [item[0][0] for item in state_reward_action]
SoC_plot = np.concatenate ([SoC_data[28*96:29*96], SoC_data[41*96:42*96], SoC_data[75*96:76*96]]).flatten()

fig_f, ax_f = plt.subplots(figsize=(12, 6))
ax_f.plot(range(len(SoC_plot)), SoC_plot, color='crimson', linewidth=2.5)
ax_f.fill_between(range(len(SoC_plot)), SoC_plot, color='lightcoral', alpha=0.3)
ax_f.set_title("State of Charge (TD3) (3 days)", fontsize=14, fontweight='bold')
ax_f.set_xlabel("Steps"); ax_f.set_ylabel("SoC"); ax_f.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
fig_f.savefig(os.path.join(DOSSIER_RUN, "State of Charge (TD3) (3 days).png"), dpi=300)
plt.close(fig_f) 

# %% Graphiqe SoC/Action

# x_axis = np.arange(3*96)
# si_data = env_eval.df_system['system imbalance'].to_numpy(dtype=np.float64)
# si_plot = np.concatenate ([si_data[28*96:29*96], si_data[41*96:42*96], si_data[75*96:76*96]])  
# action_data = [item[2] for item in state_reward_action]
# action_plot = np.concatenate ([action_data[28*96:29*96], action_data[41*96:42*96], action_data[75*96:76*96]]).flatten()

x_axis = np.arange(96)
SoC_data = [item[0][0] for item in state_reward_action]
SoC_plot = SoC_data[28*96:29*96]
action_data = [item[2] for item in state_reward_action]
action_plot = np.concatenate ([action_data[28*96:29*96]]).flatten()

fig, ax_si = plt.subplots(figsize=(12, 4.5))
ax_si.plot(x_axis, SoC_plot, color='red', linewidth=1.5, label='SoC')
ax_si.set_ylabel("SoC", color='red', fontweight='bold')
ax_si.tick_params(axis='y', labelcolor='red')
ax_si.set_ylim(-1.1, 1.1)
ax_si.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.7)  # Ligne zéro

ax_action = ax_si.twinx()
bar_colors = ['green' if a >= 0 else 'gray' for a in action_plot]

ax_action.bar(x_axis, action_plot, width=0.15, color=bar_colors, alpha=0.4, label='Agent Action')
ax_action.set_ylabel("Action [-1, 1]", color='green', fontweight='bold')
ax_action.tick_params(axis='y', labelcolor='green')
ax_action.set_ylim(-1.1, 1.1)  # Fixer les limites de l'action

plt.title("SoC - Action on Dataset (TD3) ", fontsize=14, fontweight='bold', pad=12)
ax_si.set_xlabel("Steps")
ax_si.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

plt.savefig(os.path.join(DOSSIER_RUN, "SoC - Action on Dataset (TD3) .png"), dpi=300)
plt.show()

# %% Graphique SI/Action

# x_axis = np.arange(3*96)
# si_data = env_eval.df_system['system imbalance'].to_numpy(dtype=np.float64)
# si_plot = np.concatenate ([si_data[28*96:29*96], si_data[41*96:42*96], si_data[75*96:76*96]])  
# action_data = [item[2] for item in state_reward_action]
# action_plot = np.concatenate ([action_data[28*96:29*96], action_data[41*96:42*96], action_data[75*96:76*96]]).flatten()

x_axis = np.arange(96)
si_plot = day_slice_SI(env_eval, 28)
SoC_data = [item[0][0] for item in state_reward_action]
SoC_plot = SoC_data[28*96:29*96]

fig, ax_si = plt.subplots(figsize=(12, 4.5))
ax_si.plot(x_axis, si_plot, color='blue', linewidth=1.5, label='SI (MW)')
ax_si.set_ylabel("SI [MW]", color='blue', fontweight='bold')
ax_si.tick_params(axis='y', labelcolor='blue')
max_si = max(abs(si_plot.min()), abs(si_plot.max()))
marge = max_si * 0.1
ax_si.set_ylim(-(max_si + marge), max_si + marge)
ax_si.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.7)  # Ligne zéro

ax_action = ax_si.twinx()
bar_colors = ['crimson' if a >= 0 else 'gray' for a in SoC_plot]

ax_action.bar(x_axis, SoC_plot, width=0.15, color=bar_colors, alpha=0.4, label='SoC')
ax_action.set_ylabel("SoC [0, 1]", color='crimson', fontweight='bold')
ax_action.tick_params(axis='y', labelcolor='crimson')
ax_action.set_ylim(-1.1, 1.1)  # Fixer les limites de l'action

plt.title("SI - SoC on Dataset (TD3) ", fontsize=14, fontweight='bold', pad=12)
ax_si.set_xlabel("Steps")
ax_si.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

plt.savefig(os.path.join(DOSSIER_RUN, "SI - SoC on Dataset (TD3) .png"), dpi=300)
plt.show()

#%% Graphique Price 

Price_data = env_eval.qh_prices
# Price_plot = np.concatenate ([Price_data[28*96:29*96], Price_data[41*96:42*96], Price_data[75*96:76*96]]).flatten()
Price_plot=Price_data

fig_f, ax_f = plt.subplots(figsize=(12, 6))
ax_f.plot(range(len(Price_plot)), Price_plot, color='gold', linewidth=2.5)
ax_f.fill_between(range(len(Price_plot)), Price_plot, color='lightyellow', alpha=0.3)
ax_f.set_title("Price (TD3) ", fontsize=14, fontweight='bold')
ax_f.set_xlabel("1/4 hour"); ax_f.set_ylabel("Price"); ax_f.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
fig_f.savefig(os.path.join(DOSSIER_RUN, "Price quarter hour (TD3) .png"), dpi=300)
plt.close(fig_f) 


#%% Graphique SI/Price


# x_axis = np.arange(3*96)
# si_data = env_eval.df_system['system imbalance'].to_numpy(dtype=np.float64)
# si_plot = np.concatenate ([si_data[28*96:29*96], si_data[41*96:42*96], si_data[75*96:76*96]])  
# action_data = [item[2] for item in state_reward_action]
# action_plot = np.concatenate ([action_data[28*96:29*96], action_data[41*96:42*96], action_data[75*96:76*96]]).flatten()

x_axis = np.arange(96)
# si_data = env_eval.df_system['system imbalance'].to_numpy(dtype=np.float64)
# si_plot = np.concatenate ([si_data[75*96:76*96]])
si_plot = day_slice_SI(env_eval, 75)
Price_data = env_eval.qh_prices
Price_plot_1= Price_data[75*96:76*96]


fig, ax_si = plt.subplots(figsize=(12, 4.5))
ax_si.plot(x_axis, si_plot, color='blue', linewidth=1.5, label='SI (MW)')
ax_si.set_ylabel("SI [MW]", color='blue', fontweight='bold')
ax_si.tick_params(axis='y', labelcolor='blue')
max_si = max(abs(si_plot.min()), abs(si_plot.max()))
marge = max_si * 0.1
ax_si.set_ylim(-(max_si + marge), max_si + marge)
ax_si.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.7)  # Ligne zéro

ax_action = ax_si.twinx()
max_price = max(abs(min(Price_plot_1)), abs(max(Price_plot_1)))
marge_price = max_price * 0.1 if max_price > 0 else 1.0

ax_action.plot(x_axis, Price_plot_1, color='gold', alpha=0.4, label='Price')
ax_action.set_ylabel("Price (€/MWh)", color='gold', fontweight='bold')
ax_action.tick_params(axis='y', labelcolor='gold')
ax_action.set_ylim(-(max_price + marge_price), max_price + marge_price)

plt.title("SI - Price on Dataset (TD3) ", fontsize=14, fontweight='bold', pad=12)
ax_si.set_xlabel("Steps")
ax_si.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

plt.savefig(os.path.join(DOSSIER_RUN, "SI - Price on Dataset (TD3) .png"), dpi=300)
plt.show()

#%% Graphique NRV_New

NRV_new_plot = day_slice_NRV(env_eval,28)

fig_f, ax_f = plt.subplots(figsize=(12, 6))
ax_f.plot(range(len(NRV_new_plot)), NRV_new_plot, color='purple', linewidth=2.5)
ax_f.fill_between(range(len(NRV_new_plot)), NRV_new_plot, color='plum', alpha=0.3)
ax_f.set_title("NRV new (TD3) ", fontsize=14, fontweight='bold')
ax_f.set_xlabel("Steps hour"); ax_f.set_ylabel("NRV"); ax_f.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
fig_f.savefig(os.path.join(DOSSIER_RUN, "NRV new (TD3) .png"), dpi=300)
plt.close(fig_f) 

#%% Graphique NRV_New/Price


# x_axis = np.arange(3*96)
# si_data = env_eval.df_system['system imbalance'].to_numpy(dtype=np.float64)
# si_plot = np.concatenate ([si_data[28*96:29*96], si_data[41*96:42*96], si_data[75*96:76*96]])  
# action_data = [item[2] for item in state_reward_action]
# action_plot = np.concatenate ([action_data[28*96:29*96], action_data[41*96:42*96], action_data[75*96:76*96]]).flatten()

x_axis = np.arange(96)
# NRV_new_data = env_eval.df_system['nrv_new'].to_numpy(dtype=np.float64)
NRV_new_plot = day_slice_NRV(env_eval,75)
# NRV_new_plot = np.concatenate ([NRV_new_data[75*96:76*96]])
Price_data = env_eval.qh_prices
Price_plot_1= Price_data[75*96:76*96]
Price_plot = []



fig, ax_si = plt.subplots(figsize=(12, 4.5))
ax_si.plot(x_axis, NRV_new_plot, color='purple', linewidth=1.5, label='NRV')
ax_si.set_ylabel("NRV", color='purple', fontweight='bold')
ax_si.tick_params(axis='y', labelcolor='purple')
max_si = max(abs(NRV_new_plot.min()), abs(NRV_new_plot.max()))
marge = max_si * 0.1
ax_si.set_ylim(-(max_si + marge), max_si + marge)
ax_si.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.7)  # Ligne zéro

ax_action = ax_si.twinx()
max_price = max(abs(min(Price_plot_1)), abs(max(Price_plot_1)))
marge_price = max_price * 0.1 if max_price > 0 else 1.0

ax_action.plot(x_axis, Price_plot_1, color='gold', alpha=0.4, label='Price')
ax_action.set_ylabel("Price (€/MWh)", color='gold', fontweight='bold')
ax_action.tick_params(axis='y', labelcolor='gold')
ax_action.set_ylim(-(max_price + marge_price), max_price + marge_price)

plt.title("NRV New - Price on Dataset (TD3) ", fontsize=14, fontweight='bold', pad=12)
ax_si.set_xlabel("Steps")
ax_si.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

plt.savefig(os.path.join(DOSSIER_RUN, "NRV New - Price on Dataset (TD3) .png"), dpi=300)
plt.show()