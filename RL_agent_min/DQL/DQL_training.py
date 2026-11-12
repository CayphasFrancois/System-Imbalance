
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import torch as T
import random
import time
import os
import gc 
from tqdm import tqdm  # type: ignore
from datetime import datetime

# --- IMPORTATION DU PIPELINE PYTORCH (Environnements & DQL Agent) ---
from Environment_min import Environment, importdata
from DQL_torch import DQNAgent, set_seed

# %% -------Title of run-------
title_of_run = ('DQL Agent Training')
print(title_of_run)

# %% ------- Seed (Total Performance Reproducibility Setup) -------
seed = 1
os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(seed)

# %% -------Parameters + Data-------
battery_max_power = 20  # MW
EP_ratio = 1
delta_t = 1/60  # time control interval (h)
roundtrip_efficiency = 0.9
battery_replacement_cost = 3e+5 * battery_max_power * EP_ratio

penal = 200

# Input training data matrices
path_training = os.path.normpath('/Dataset_Train_2025_2026_Current_SI.csv')
path_validation = os.path.normpath('./Dataset_Val_2025_2026_Current_SI.csv')
path_test = os.path.normpath('/Dataset_Test_2025_2026_Current_SI.csv') 
path_merit_order = os.path.normpath('./Data_Now/merit_order_dictionaries.csv')

print('Importing data + preprocessing (PyTorch Pipeline)...')
train_obs, val_obs, test_obs, merit_order_dict = importdata(path_training, path_validation, path_test, path_merit_order)   
print('Preprocessed data imported')

# %% -------Hyperparameters-------
nb_quarters_per_episode = 96
nb_min_per_episode = 60 * 24
replay_buffer_max_length = int(1e6)
batch_size = 256 
num_epochs = 20

gamma = 0.99
lr = 1e-4

# Discrétisation des actions continues pour le DQL (ex: 5 actions de -1 à 1)
action_space_discretized = np.linspace(-1.0, 1.0, 5) 
n_actions = len(action_space_discretized)

# Paramètres Epsilon-Greedy
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

# TIMELINE TRACKING INTERVALS
log_interval = 60 * 24 * 10       
eval_interval = 60 * 24 * 20    

# %% ------- Training setup variables -------
num_training_episodes = int(num_epochs * int(len(train_obs)) / nb_min_per_episode)
num_val_episodes = int(len(val_obs) / nb_min_per_episode)

# %% ------- Print & Export Hyperparameters Summary -------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(os.getcwd(), f'Run_DQL_PyTorch_{timestamp}_EARLY_STOP_S{seed}_TEST')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

config_path = os.path.join(save_dir, "hyperparameters.txt")
with open(config_path, "w", encoding="utf-8") as f:
    f.write(f"{' Category ':*^40}\n")
    f.write(f"  - Algorithm                  : DQL (PyTorch Version)\n")
    f.write(f"  - Random Seed                : {seed}\n")
    f.write(f"  - Global Epochs              : {num_epochs}\n")
    f.write(f"  - Batch size                 : {batch_size}\n")
    f.write(f"  - Action space (Discretized) : {action_space_discretized}\n")
    f.write(f"  - Gamma (Discount factor)    : {gamma}\n")
    f.write(f"  - Learning Rate              : {lr}\n")
    f.write(f"  - Battery Max Power (MW)     : {battery_max_power}\n")
    f.write(f"  - Penalty (Constraint)       : {penal}\n")
    f.write(f"\n{'*'*40}\n")

print(f"Dossier créé et hyperparamètres sauvés : {save_dir}")

# %% -------Environments-------
starting_time = time.time()

train_env = Environment(train_obs, merit_order_dict, battery_max_power,
                        1, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency,
                        battery_replacement_cost, penalty=penal) 

validation_env = Environment(val_obs, merit_order_dict, battery_max_power,
                             1, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency,
                             battery_replacement_cost, penalty=0) 

# %% -------Instanciation de l'Agent DQL PyTorch-------
input_dims = train_obs.shape[1]

agent = DQNAgent(
    state_dim=input_dims,
    action_dim=n_actions,
    lr=lr,
    gamma=gamma,
    buffer_capacity=replay_buffer_max_length
)

# %% ------- Synchronized Manual Evaluation Loop Interface (DQL) -------
def evaluate_agent(env, num_episodes, nb_min):
    state = env.reset()
    total_profit = 0.0

    for ep in range(num_episodes):
        episode_profit = 0.0
        for _ in range(nb_min):
            # Évaluation déterministe (epsilon = 0)
            action_idx = agent.select_action(state, epsilon=0.0)
            action = action_space_discretized[action_idx] #type: ignore
            
            state, reward, done = env.step(action)
            episode_profit += reward

            if done:
                state = env.reset()
                
        total_profit += episode_profit
        if (ep + 1) % 50 == 0:
            gc.collect() 
            
    return total_profit / num_episodes

# %% ------- Évaluation Initiale -------
print("\nPOLICY EVALUATION BEFORE TRAINING \n")
avg_return_initial = evaluate_agent(validation_env, num_val_episodes, nb_min_per_episode)
print(f"Average return (before training): {avg_return_initial:.2f} € / day")

returns = [avg_return_initial]
train_rewards = []
average_train_rewards = []
best_avg_return = avg_return_initial

patience = 5
wait_counter = 0           
min_delta = 0.5            
early_stop_triggered = False 

# %% ------- TRAINING LOOP (DQL PyTorch)-------
print("\n START TRAINING DQL (PyTorch) ... \n")
global_step = 0

for ep in tqdm(range(num_training_episodes), desc="Training Episodes (DQL)", unit="ep"):
    state = train_env.reset()

    for min_idx in range(nb_min_per_episode):
        # Sélection d'action Epsilon-Greedy
        action_idx = agent.select_action(state, epsilon=epsilon)
        action = action_space_discretized[action_idx] #type: ignore

        next_state, reward, done = train_env.step(action)
        
        # Stockage dans le Replay Buffer persistant
        agent.store_transition(state, action_idx, reward, next_state, float(done))

        if min_idx % 15 == 0: 
            train_rewards.append(reward)

        # Apprentissage par mini-batch
        agent.update(batch_size=batch_size)

        state = next_state
        global_step += 1

        # Logging d'entraînement
        if global_step % log_interval == 0:
            total_steps_collected = log_interval
            profit_per_step = np.sum(train_rewards) / total_steps_collected
            avg_train_per_day = profit_per_step * nb_min_per_episode
            
            average_train_rewards.append(avg_train_per_day)
            print(f"Step {global_step:05d} | Avg Reward (Train) : {avg_train_per_day:>8.2f} € / day | Epsilon : {epsilon:.2f}")
            train_rewards = []
            gc.collect()

        # Validation et Early Stopping
        if global_step % eval_interval == 0:
            avg_return = evaluate_agent(validation_env, num_val_episodes, nb_min_per_episode)
            returns.append(avg_return)
            print("\n"+"-"*60)
            print(f"EVALUATION (Step {global_step:05d}) | Validation Return (full): {avg_return:>8.2f} € / day")

            if avg_return > (best_avg_return + min_delta):
                print(f" -> AMÉLIORATION : {avg_return:.2f} > {best_avg_return:.2f} + {min_delta}")
                best_avg_return = avg_return
                wait_counter = 0  
                agent.save(os.path.join(save_dir, "best_dql_agent.pth"))
                print(" -> New best model saved.")
            else:
                wait_counter += 1
                print(f" -> Pas d'amélioration suffisante. Patience : {wait_counter} / {patience}")
                
            print("-" * 60)

            if wait_counter >= patience:
                print(f"\n[!!!] EARLY STOPPING TRIGGERED [!!!]")
                early_stop_triggered = True
                break 
                
    # Décroissance d'epsilon et mise à jour du réseau cible par épisode
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    agent.update_target_network()
                
    if early_stop_triggered:
        break 

print(f"\nEntraînement terminé. Sauvegarde des logs...")

# Export Excel et Graphiques
df_val = pd.DataFrame({'Environment_Steps': np.arange(len(returns)) * eval_interval, 'Validation_Reward_Eur_Jour': returns})
df_train = pd.DataFrame({'Environment_Steps': (np.arange(len(average_train_rewards)) + 1) * log_interval, 'Train_Reward_Eur_Jour': average_train_rewards})

excel_path = os.path.join(save_dir, "Historique_Entrainement_DQL.xlsx")
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    df_val.to_excel(writer, sheet_name='Validation_Reward', index=False)
    df_train.to_excel(writer, sheet_name='Train_Reward', index=False)

print(f"-> Données sauvegardées dans : {excel_path}")

print("\nGenerating and saving plots...")

# 1. Training Returns Layout Plot
fig1, ax1 = plt.subplots(figsize=(10, 6))
steps_train = (np.arange(len(average_train_rewards))+1) * log_interval
ax1.plot(steps_train, average_train_rewards, color='blue', linewidth=1.5, label='Train (Raw)')
ax1.set_title("Training Profit Evolution (DQL - PyTorch)", fontsize=14)
ax1.set_xlabel("Environment Steps")
ax1.set_ylabel("Profit (€ / day)")
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()
plt.tight_layout()
fig1.savefig(os.path.join(save_dir, "1_Train_Reward.png"), dpi=300)

# 2. Full Validation Horizon Performance Return Layout curve
fig2, ax2 = plt.subplots(figsize=(10, 6))
steps_val = np.arange(len(returns)) * eval_interval
ax2.plot(steps_val, returns, marker='o', color='red', linewidth=2, label='Validation (Full Dataset)')
ax2.set_title("Evaluation Performance on Validation Set (DQL - PyTorch)", fontsize=14)
ax2.set_xlabel("Environment Steps")
ax2.set_ylabel("Average Profit (€ / day)")
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()
plt.tight_layout()
fig2.savefig(os.path.join(save_dir, "2_Validation_Reward.png"), dpi=300)

plt.close('all') 
print(f"\nFin d'exécution. Les graphiques ont été sauvegardés dans : {save_dir}")