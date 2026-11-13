import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import torch as T
import random
import time
import os
import sys
import shutil 
import gc 
from tqdm import tqdm 
from datetime import datetime

import faulthandler, psutil 

# import ctypes
# ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
# ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
proc = psutil.Process(os.getpid())
Result_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "C:\\Document_UMONS\\Thèse-Code\\SI_side_repository"))
Environment_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "C:\\Document_UMONS\\Thèse-Code\\System_Imbalance\\System_Imbalance\\Environment_min"))
sys.path.append(Result_dir)
faulthandler.enable(open(os.path.join(Result_dir, "crash.log"), "ab"))

# --- IMPORTATION DU PIPELINE PYTORCH (Environnements & Prétraitement) ---
import importlib
import Environment_min
importlib.reload(Environment_min)
from Environment_min import Environment, importdata, collect_data
from TD3_torch import Agent, set_seed


# %% ------- Synchronized Manual Evaluation Loop Interface (PyTorch) -------
def evaluate_agent(env,agent,num_episodes=None):
    env._episode = 0                      
    num_episodes = env.n_days if num_episodes is None else num_episodes
    agent.actor.eval()
    total = 0.0
    ep_profit_list = []
    state_reward_action_list = []
    for _ in range(num_episodes):
        state = env.reset()
        done, ep_profit = False, 0.0
        while not done:
            with T.no_grad():
                a = agent.actor(T.tensor(state, dtype=T.float).unsqueeze(0)
                                .to(agent.actor.device)).cpu().numpy()[0]
                state_pre=state.copy()
            state, r, action, done = env.step(a)
            state_reward_action_list.append((state_pre,r,a,action))
            ep_profit += r

        total += ep_profit
        ep_profit_list.append(ep_profit)
        

    gc.collect()
    return total / num_episodes,ep_profit_list,state_reward_action_list


def main():
    # %% -------Title of run-------
    title_of_run = ('TD3 Agent Training')
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

    # window_size = 4
    penal = 30

    # Input training data matrices including warm-up sliding indices blocks
    path_training = os.path.normpath('C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Dataset_Train_2025_2026_Current_SI.csv')
    path_validation = os.path.normpath('C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Dataset_Val_2025_2026_Current_SI.csv')
    path_test = os.path.normpath('C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/Dataset_Test_2025_2026_Current_SI.csv')
    path_merit_order = os.path.normpath('C:/Document_UMONS/Thèse-Code/SI_side_repository/Data/merit_parsed.pkl')

    print('Importing data + preprocessing (PyTorch Pipeline)...')
    df_train,  df_val,  _, merit_order_dict, mean,std= importdata(path_training, path_validation, path_test, path_merit_order)   
    # cols = ["system imbalance", "ace", "igcc", "nrv_new",
    #         "min_sin","min_cos","h_sin","h_cos","day_sin","day_cos"]
    # # for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
    # for name, d in [("train", df_train), ("val", df_val)]:
    #     print(name, d[cols].isna().sum().to_dict())
    #     print(name, np.isinf(d[cols].select_dtypes('number')).sum().sum(), "inf")
    print('Preprocessed data imported')

    # %% -------Hyperparameters-------
    tuned_initial_exp = 0   
    nb_quarters_per_episode = 96
    nb_min_per_episode = 60*24
    replay_buffer_max_length = int(1e5)
    batch_size = 128 
    num_epochs = 20

    gamma = 0.99
    tau = 5*1e-3
    actor_lr = 1e-5
    critic_lr = 1e-5

    actor_update_T = 4
    critic_update_T = 1

    layer_size=(500,500)

    # TIMELINE TRACKING INTERVALS
    log_interval = 60*24*50 
    eval_interval = 60*24*100    

    # %% ------- Training setup variables -------
    noise_stddev = 0.2
    initial_collect_steps = batch_size * 20
    # %% ------- Print & Export Hyperparameters Summary -------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("C:\\Document_UMONS\\Thèse-Code\\SI_side_repository\\Runs\\Minute\\Runs", f'Run_TD3_PyTorch_min_{timestamp}_EARLY_STOP_S{seed}_TEST')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- ENREGISTREMENT DU FICHIER DES HYPERPARAMÈTRES ---
    config_path = os.path.join(save_dir, "hyperparameters.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"{' Category ':*^40}\n")
        f.write(f"  - Algorithm                  : TD3 (PyTorch Version)\n")
        f.write(f"  - Random Seed                : {seed}\n")
        # f.write(f"  - Window size (Past History) : {window_size}\n")
        f.write(f"  - Global Epochs              : {num_epochs}\n")
        f.write(f"  - Batch size                 : {batch_size}\n")
        f.write(f"  - Replay buffer max length   : {replay_buffer_max_length}\n")
        f.write(f"  - Gamma (Discount factor)    : {gamma}\n")
        f.write(f"  - Tau (Soft update rate)     : {tau}\n")
        f.write(f"  - Actor LR                   : {actor_lr}\n")
        f.write(f"  - Critic LR                  : {critic_lr}\n")
        f.write(f"  - Hidden layers (Actor)      : {layer_size}\n")
        f.write(f"  - Hidden layers (Critic)     : {layer_size}\n")
        f.write(f"  - Exploration Noise (Std)    : {noise_stddev}\n")
        f.write(f"  - Policy Noise               : 0.2\n")
        f.write(f"  - Noise Clip                 : 0.5\n")
        f.write(f"  - Update Actor Iterations    : {actor_update_T}\n")
        f.write(f"  - Update Critic Iterations   : {critic_update_T}\n")
        f.write(f"  - Battery Max Power (MW)     : {battery_max_power}\n")
        f.write(f"  - Roundtrip Efficiency       : {roundtrip_efficiency}\n")
        f.write(f"  - Battery Replacement Cost   : {battery_replacement_cost}\n")
        f.write(f"  - Penalty (Constraint)       : {penal}\n")
        f.write(f"  - Tuned_initial_exp          : {tuned_initial_exp}\n")
        f.write(f"  - State                      : SoC + 6 calendar variables + window (size = 4) * (SI_prevbatteryaction, Price_final_QH, NRV_withoutIGCC_postbatteryaction) + 14 (number of minutes) * (SI_premodif, ACE, IGCC, NRV_new_postbaterryaction,Price_predictionbasedonQHvalues)\n")
        f.write(f"  - State dims                 : 89 \n")
        f.write(f"  - Ce qui change              : lr 10-5, batch_size 128")
        f.write(f"\n{'*'*40}\n")

    print(f"Dossier créé et hyperparamètres sauvés : {save_dir}")

    # %% -------Environments-------
    starting_time = time.time()

    train_env = Environment(df_train,mean,std, merit_order_dict, battery_max_power,
                            1, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency,
                            battery_replacement_cost, penalty=penal) 

    validation_env = Environment(df_val,mean,std, merit_order_dict, battery_max_power,
                                1, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency,
                                battery_replacement_cost,penalty=0) 
    num_training_episodes = num_epochs * train_env.n_days
    num_val_episodes = validation_env.n_days

    # %% -------Instanciation de l'Agent PyTorch-------
    input_dims = (89,)
    n_actions = 1

    agent = Agent(
            alpha=actor_lr,
            beta=critic_lr,
            input_dims=input_dims,
            env=None,
            tau=tau,
            gamma=gamma,
            n_actions=n_actions,
            max_size=replay_buffer_max_length,
            layer1_size=layer_size[0],
            layer2_size=layer_size[1],
            batch_size=batch_size,
            exploration_noise=noise_stddev,
            policy_noise=0.2,
            noise_clip=0.5,
            update_actor_iter=actor_update_T,
            seed=seed,
            filename=save_dir)


    # %% -------Initial fill (Warm-up / Replay Buffer)-------
    start = datetime.now()
    print("\nData collection (Warm-up) ... \n")
    collect_data(train_env, agent, initial_collect_steps, tuned_initial_exp)  # Tuned_initial_exp = 1 ==> Politique de l'Agent et = 0 ==> Politique forcée pour avoir des samples variés dans le buffer
    train_env._episode = 0  

    print("\nPOLICY EVALUATION BEFORE TRAINING \n")
    avg_return_initial,_,_ = evaluate_agent(validation_env,agent)
    print(f"Average return (before training): {avg_return_initial:.2f} € / day")
    agent.save_models()
    # print(f"Merit hits/misfits : {validation_env.merit_hits/validation_env.merit_miss}")


    returns = [avg_return_initial]
    train_rewards = []
    average_train_rewards = []


    best_avg_return = avg_return_initial

    # --- CONVERGENCE PATIENCE PARAMETERS SETUP FOR EARLY STOPPING PASS ---
    patience = 5             # Allowed sequential evaluation evaluations checks without updates
    wait_counter = 0           
    min_delta = 0.5            
    early_stop_triggered = False 

    # %% ------- TRAINING LOOP (PyTorch)-------
    print("\n START TRAINING (PyTorch) ... \n")
    global_step = 0

    for ep in tqdm(range(num_training_episodes), desc="Training Episodes", unit="ep"):

        state = train_env.reset()
        done = False 

        while not done : 
            
            action = agent.choose_action(state)
            next_state, reward, _ , done = train_env.step(action)
            
            agent.remember(state, action, reward, next_state, done)

            train_rewards.append(reward)
            
            # 2. Entraînement de l'agent (Mise à jour des réseaux Critic et Actor)
            if global_step % critic_update_T == 0:
                agent.learn()

            state = next_state
            global_step += 1

            # CONSOLE CONVERGENCE LOGS OUTPUT PIPELINE    ==> Juste pour être mis au courant pendant l'entrainement de comment ça évolue 
            if global_step % log_interval == 0:
                total_steps_collected = log_interval
                profit_per_step = np.sum(train_rewards) / (total_steps_collected)
                avg_train_per_day = profit_per_step * nb_min_per_episode
                
                average_train_rewards.append(avg_train_per_day)
                print(f"Step {global_step:05d} | Avg Reward (Train) : {avg_train_per_day:>8.2f} € / day")
                print(f"Step {global_step} | RAM {proc.memory_info().rss/1e9:.2f} GB | buffer {agent.memory.mem_cntr}")
                train_rewards = []
                gc.collect()

            # PERIODIC SYNCHRONIZED VALIDATION HORIZON CYCLE RUNS
            if global_step % eval_interval == 0:
                avg_return,_,_ = evaluate_agent(validation_env,agent)
                returns.append(avg_return)
                print("\n"+"-"*60)
                print(f"EVALUATION (Step {global_step:05d}) | Validation Return (full): {avg_return:>8.2f} € / day")

                # --- PROCESS CHECKPOINT WEIGHTS FOR DEFENSIVE EARLY STOPPING CRITERIA ---
                if avg_return > (best_avg_return + min_delta):
                    print(f" -> AMÉLIORATION : {avg_return:.2f} > {best_avg_return:.2f} + {min_delta}")
                    best_avg_return = avg_return
                    wait_counter = 0  
                    
                    agent.save_models()
                    print(" -> New best models saved.")
                else:
                    wait_counter += 1
                    print(f" -> Pas d'amélioration suffisante. Patience : {wait_counter} / {patience}")
                    
                print("-" * 60)

                if wait_counter >= patience:
                    print(f"\n[!!!] EARLY STOPPING TRIGGERED [!!!]")
                    print(f"L'agent TD3 ne progresse plus. Arrêt à l'itération {global_step} pour éviter l'overfitting.")
                    early_stop_triggered = True
                    break 
                    
        if early_stop_triggered:
            break 

    # %% -------End of training time-------
    end = datetime.now()
    td = (end - start).total_seconds()
    nb_hours = td / 3600

    if early_stop_triggered:
        print(f"\n[INFO] Entraînement écourté par l'Early Stopping.")
        
    print(f"\nThe time of execution is : {td:.03f} s ({nb_hours:.03f} h)")

    # %%=============================================================================
    # HISTORICAL TRAINING PROFILE GENERATION LOGS (MULTI-SHEET EXCEL EXPORT)
    # =============================================================================
    print("\nExportation des données d'entraînement vers Excel...")

    df_val_stock = pd.DataFrame({
        'Environment_Steps': np.arange(len(returns)) * eval_interval,
        'Validation_Reward_Eur_Jour': returns,
    })

    df_train_stock = pd.DataFrame({
        'Environment_Steps': (np.arange(len(average_train_rewards)) + 1) * log_interval,
        'Train_Reward_Eur_Jour': average_train_rewards
    })

    excel_path = os.path.join(save_dir, "Historique_Entrainement_TD3.xlsx")
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df_val_stock.to_excel(writer, sheet_name='Validation_Reward', index=False)
        df_train_stock.to_excel(writer, sheet_name='Train_Reward', index=False)

    print(f"-> Données brutes sauvegardées avec succès dans : {excel_path}")

    # %%=============================================================================
    # PLOTTING SEPARATED CHARTS INTERFACE
    # =============================================================================
    print("\nGenerating and saving plots...")

    # 1. Training Returns Layout Plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    steps_train = (np.arange(len(average_train_rewards))+1) * log_interval
    ax1.plot(steps_train, average_train_rewards, color='blue', linewidth=1.5, label='Train (Raw)')
    ax1.set_title("Training Profit Evolution (TD3 - PyTorch)", fontsize=14)
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
    ax2.set_title("Evaluation Performance on Validation Set (TD3 - PyTorch)", fontsize=14)
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Average Profit (€ / day)")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, "2_Validation_Reward.png"), dpi=300)

    plt.close('all') 
    print(f"\nFin d'exécution. Les graphiques ont été sauvegardés dans : {save_dir}")

if __name__ == "__main__":
    main()