import os
os.environ["PANDAPOWER_NUMBA"] = "False"
import random
import numpy as np
import pandas as pd  # type: ignore
from sklearn.preprocessing import StandardScaler 
import pickle

class Environment:
    def __init__(self,df_system, mean,std, df_merit, max_power: float, discount_rate: float, 
                 nb_quarters_per_episode: int, EP: float, eta: float, bat_replacement_cost, penalty):
        """
        Environnement RL configuré avec un départ aléatoire strict à 00:00 par jour,
        un reward différé sur 15 minutes et l'historique réinitialisé par quart d'heure.
        """
        self.df_system = df_system.reset_index(drop=True)
        ds = self.df_system
        self._si_arr   = ds['system imbalance'].to_numpy(dtype=np.float64)
        self._ace_arr  = ds['ace'].to_numpy(dtype=np.float64)
        self._igcc_arr = ds['igcc'].to_numpy(dtype=np.float64)
        self._nrv_arr  = ds['nrv_new'].to_numpy(dtype=np.float64)
        self._cal_arr  = ds[['min_sin','min_cos','h_sin','h_cos','day_sin','day_cos']].to_numpy(dtype=np.float64)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std  = np.asarray(std,  dtype=np.float64)



        # if 'Datetime' in df_merit.columns:
        #     df_merit = df_merit.set_index('Datetime')
        # df_merit.index = pd.to_datetime(df_merit.index)
        self._merit_cache = df_merit          # c'est maintenant un dict déjà converti
        self._inc_keys = np.array(sorted(k for k, (i, d) in df_merit.items() if i is not None), dtype=np.int64)
        self._dec_keys = np.array(sorted(k for k, (i, d) in df_merit.items() if d is not None), dtype=np.int64)
        self._merit_key = None
        self._merit_inc = self._merit_dec = None
        self._last_valid_inc = self._last_valid_dec = None
        self.merit_hits, self.merit_miss, self.merit_empty = 0, 0, 0

        self._nb_quarters_per_episode = int(nb_quarters_per_episode)
        self._max_power = float(max_power)
        self._discount = float(discount_rate)
        self._EP_ratio = float(EP)
        self._eta = float(eta)
        self._R = float(bat_replacement_cost)
        self._penalty = float(penalty)

        self._SOC_index = 0 
        self._episode = 0
        self._episode_ended = False
        
        dt = pd.to_datetime(self.df_system['datetime'])
        steps = self._nb_quarters_per_episode * 15
        self._min_in_qh_arr = (dt.dt.minute.to_numpy() % 15)
        self._qh_key_arr = dt.dt.floor('15min').to_numpy()
        
        self._qh_key_int = self._qh_key_arr.astype('datetime64[ns]').astype('int64')

        midnights = np.flatnonzero(((dt.dt.hour == 0) & (dt.dt.minute == 0)).to_numpy())
        # on ne garde que les jours complets et contigus (pas de trou, pas de changement d'heure)
        self._day_starts = np.array([
            s for s in midnights
            if s + steps <= len(dt)
            and dt.iloc[s + steps - 1] - dt.iloc[s] == pd.Timedelta(minutes=steps - 1)
            ])
        print(len(self._day_starts), "jours complets trouvés dans le dataset")
        if len(self._day_starts) == 0:
            raise ValueError("Aucun jour complet commençant à 00:00 dans ce dataset")
        self.n_days = len(self._day_starts)
        self._current_idx = int(self._day_starts[0])
        self._start_idx = self._current_idx

 
        self.qh_prices = []
        self._init_history_buffers()
        self._observation = self._get_current_observation(soc=0.5)

    def _init_history_buffers(self):
        # Tampons pour les 4 derniers quarts d'heure (moyennes glissantes)
        self.buf_q_si = [0.0] * 4
        self.buf_q_price = [0.0] * 4
        self.buf_q_nrv = [0.0] * 4

        # Accumulateurs pour le quart d'heure en cours
        self._qh_energy_mwh = 0.0
        self._qh_cost_sum = 0.0
        self._qh_penalty = 0.0
        self._qh_si_list = []
        self._qh_si_base_list =[]
        self._qh_nrv_list = []
        self.q_si_mean_previous = 0.0
        
        # Listes temporaires pour stocker les minutes du quart d'heure courant
        self._qh_minutes_price = []
        self._qh_inc_dicts = []
        self._qh_dec_dicts = []
        key0 = int(self._qh_key_int[self._current_idx])
        k_inc = self._nearest_valid_key(self._inc_keys, key0)
        k_dec = self._nearest_valid_key(self._dec_keys, key0)
        self._last_valid_inc = self._merit_cache[k_inc][0] if k_inc is not None else None
        self._last_valid_dec = self._merit_cache[k_dec][1] if k_dec is not None else None
        self._merit_key = None

    def _nearest_valid_key(self, keys, key):
        """Clé valide la plus proche : le dernier quart d'heure <= key,
        sinon (rien avant) le premier quart d'heure après."""
        if len(keys) == 0:
            return None
        j = int(np.searchsorted(keys, key, side="right")) - 1
        return int(keys[max(j, 0)])

    def reset(self):
        self._episode += 1
        self._episode_ended = False
        
        self._current_idx = int(self._day_starts[(self._episode - 1) % self.n_days])
        self._start_idx = self._current_idx

        self._init_history_buffers()
        
        self._observation = self._get_current_observation(soc=0.5)
        return self._observation

    def _get_current_observation(self, soc: float):
        i = min(self._current_idx, len(self._si_arr) - 1)
        m = len(self._qh_nrv_list)

        h = np.zeros((5, 14))                       # lignes : si, ace, igcc, nrv, prix
        if m > 0:
            idx = i - 1 - np.arange(m)              # minute la plus récente d'abord
            h[0, :m] = self._si_arr[idx]
            h[1, :m] = self._ace_arr[idx]
            h[2, :m] = self._igcc_arr[idx]
            h[3, :m] = self._qh_nrv_list[::-1]      # avec effet batterie
            h[4, :m] = self._qh_minutes_price[::-1]
        h = (h - self.mean[:5, None]) / self.std[:5, None]

        sel = [0, 3, 4]                             # si, nrv, prix
        h4 = np.array([self.buf_q_si, self.buf_q_nrv, self.buf_q_price], dtype=np.float64)
        h4 = (h4 - self.mean[sel, None]) / self.std[sel, None]

        return np.concatenate(([soc], self._cal_arr[i], h.ravel(), h4.ravel())).astype(np.float32)

    @staticmethod
    def _unpack(d):
        # d = {volume_cumulé: (prix, ordre)}, déjà trié par volume croissant, ordre 1 = aFRR, 2 = mFRR
        if not d or not isinstance(d, dict):
            return None
        vols   = np.fromiter((float(k)  for k in d),          dtype=float)
        prices = np.fromiter((v[0]      for v in d.values()), dtype=float)
        flags  = np.fromiter((int(v[1]) for v in d.values()), dtype=int)
        return vols, prices, flags

    def _get_merit(self, idx):
        key = int(self._qh_key_int[idx])
        if key != self._merit_key:
            self._merit_key = key
            hit = self._merit_cache.get(key)
            if hit is None:
                self.merit_miss += 1
                inc, dec = None, None
            else:
                self.merit_hits += 1
                inc, dec = hit

            # repli sur le dernier dictionnaire valide, comme dans ta version actuelle
            if inc is not None: self._last_valid_inc = inc
            else:
                inc = self._last_valid_inc; self.merit_empty += 1
            if dec is not None: self._last_valid_dec = dec
            else:
                dec = self._last_valid_dec

            self._merit_inc, self._merit_dec = inc, dec
        return self._merit_inc, self._merit_dec

    # ---------- Briques de calcul ----------
    @staticmethod
    def _marginal_idx(vols, target):
        # Indice de la tranche (V_{i-1}, V_i] qui contient le volume demandé (plafonné à la dernière offre)
        return min(int(np.searchsorted(vols, target, side="left")), len(vols) - 1)

    def _get_max_mfrr_activated(self, u):          # MIP : mFRR la plus chère activée à la hausse
        pos = [n for n in self._qh_nrv_list if n > 0]
        if u is None or not pos:
            return None
        vols, prices, flags = u
        i = self._marginal_idx(vols, max(pos))
        sel = prices[:i + 1][flags[:i + 1] == 2]
        return float(sel.max()) if sel.size else None

    def _get_min_mfrr_activated(self, u):          # MDP : mFRR la moins chère activée à la baisse
        neg = [abs(n) for n in self._qh_nrv_list if n < 0]
        if u is None or not neg:
            return None
        vols, prices, flags = u
        i = self._marginal_idx(vols, max(neg))
        sel = prices[:i + 1][flags[:i + 1] == 2]
        return float(sel.min()) if sel.size else None

    def _get_weighted_afrr_price(self, inc,dec):
            
        num = den = 0.0
        for nrv in self._qh_nrv_list:
            if nrv == 0 : 
                continue 
            if nrv > 0 :
                u = inc 
            else : u = dec
            if u is None:
                continue
            vols, prices, flags = u
            m = flags == 1
            if not m.any() :
                continue
            va, pa = vols[m], prices[m]
            w = min(abs(nrv), float(va.max())) 
            i = self._marginal_idx(va,w)
            num += w * pa[i]
            den += w
        return num / den if den > 0 else None

    # ---------- Prix du quart d'heure ----------
    def _calculate_alpha (self,q_si_mean_now,q_si_mean_previous,price) : 
        a = 0
        b = 200
        c = 450
        d = 65

        x = (abs(q_si_mean_now) + abs(q_si_mean_previous))/2 

        if price >= 400 : 
            cp = 0
        elif 400 > price >=200 :
            cp = (400-price)/200
        elif 200 > price >= 0 : 
            cp = 1
        elif 0 > price >= -200 : 
            cp = (200+price)/200
        else : cp = 0 

        alpha = (a + b/(1 + np.exp((c-x)/d)))*cp

        return alpha

    def _voaa_list(self, inc, dec):
        """VoAA = prix de la 1re offre du merit order, par produit (aFRR=1, mFRR=2) et par direction."""
        up   = float(inc[1].min()) if inc is not None else None   # 1re offre FRR à la hausse (aFRR+mFRR)
        down = float(dec[1].max()) if dec is not None else None   # 1re offre FRR à la baisse (aFRR+mFRR)

        return up,down
    
    def _calculate_quarter_price(self, q_si_mean,q_si_mean_previous):
        inc = self._qh_inc_dicts[0] if self._qh_inc_dicts else None
        dec = self._qh_dec_dicts[0] if self._qh_dec_dicts else None
        up, down = self._voaa_list(inc, dec)
        voaa = [v for v in (up, down) if v is not None]
        if not voaa:
            print("Warning: No merit order data available. Returning price 0.0.")
            return 0.0
        floor, cap = max(voaa), min(voaa)
        
        if q_si_mean < -25.0:                                   # MIP (pénurie)
            c = [x for x in (floor,
                            self._get_max_mfrr_activated(inc),
                            self._get_weighted_afrr_price(inc, dec)) if x is not None]
            if not c:
                return 0.0
            price = float(max(c))

            if abs (q_si_mean) > 150:
                alpha = self._calculate_alpha(q_si_mean,q_si_mean_previous,price)
            else : alpha = 0 

            return (price + alpha) 

        elif q_si_mean > 25.0:                                  # MDP (surplus)
            c = [x for x in (cap,
                            self._get_min_mfrr_activated(dec),
                            self._get_weighted_afrr_price(inc, dec)) if x is not None]

            if not c:
                return 0.0
            price = float(min(c))

            if abs (q_si_mean) > 150:
                alpha = self._calculate_alpha(q_si_mean,q_si_mean_previous,price)
            else : alpha = 0 

            return (price - alpha)

        else:                                                    # zone morte
            return ((floor + cap) / 2.0) if (floor is not None and cap is not None) else 0.0
    

    def step(self, action):
        if self._episode_ended:
            return self.reset(), 0.0,0.0, True

        action_scalar = float(action.item() if hasattr(action, 'item') else action)
        current_soc = self._observation[0]
        
        new_soc = current_soc + (action_scalar * (1/60) * (1/self._EP_ratio))
        eps = 1e-6
        penalized = 0.0 if -eps <= new_soc <= 1 + eps else self._penalty
        self._qh_penalty += penalized    
        new_soc = np.clip(new_soc, 0.0, 1.0)
        
        fraction_of_max_power = (new_soc - current_soc) * (self._EP_ratio / (1/60))

        cost = 0.0
        if fraction_of_max_power < 0:
            cost = self._R * 5.24e-4 * (abs(fraction_of_max_power) / 60) ** 2.03

        battery_charge_MW = fraction_of_max_power * self._max_power
        if battery_charge_MW > 0: 
            network_charge_MW = battery_charge_MW / np.sqrt(self._eta)
        else:        
            network_charge_MW = battery_charge_MW * np.sqrt(self._eta)

        base_si = self._si_arr[self._current_idx]
        base_nrv_new = self._nrv_arr[self._current_idx]

        si_new_val = base_si - network_charge_MW  
        nrv_new_val = base_nrv_new + network_charge_MW

        energy_min_mwh = network_charge_MW * (1/60)
        
        inc_dict, dec_dict = self._get_merit(self._current_idx)
        

        # Stockage des dictionnaires pour le quart d'heure
        self._qh_inc_dicts.append(inc_dict)
        self._qh_dec_dicts.append(dec_dict)

        self._qh_energy_mwh += energy_min_mwh
        self._qh_cost_sum += cost
        self._qh_si_list.append(si_new_val)
        self._qh_nrv_list.append(nrv_new_val)
        self._qh_si_base_list.append(base_si)

        q_si_mean_so_far = np.mean(self._qh_si_list)
        current_minute_price = self._calculate_quarter_price(q_si_mean_so_far,self.q_si_mean_previous)
        self._qh_minutes_price.append(current_minute_price)

        is_end_of_qh = (self._min_in_qh_arr[self._current_idx] == 14)
        reward = 0 

        if is_end_of_qh:
            q_si_mean = np.mean(self._qh_si_base_list)
            final_qh_price = current_minute_price
            q_nrv_mean = np.mean(self._qh_nrv_list)

            self.qh_prices.append(final_qh_price)

            profit_qh = -self._qh_energy_mwh * final_qh_price
            reward = profit_qh - self._qh_cost_sum - self._qh_penalty

            self.buf_q_si.pop(0); self.buf_q_si.append(q_si_mean) #type:ignore
            self.buf_q_price.pop(0); self.buf_q_price.append(final_qh_price)
            self.buf_q_nrv.pop(0); self.buf_q_nrv.append(q_nrv_mean) #type:ignore
            self.q_si_mean_previous = np.mean(self._qh_si_list)

            # Réinitialisation des accumulateurs du quart d'heure
            self._qh_energy_mwh = 0.0
            self._qh_cost_sum = 0.0
            self._qh_penalty = 0.0  
            self._qh_si_list = []
            self._qh_si_base_list =[]
            self._qh_nrv_list = []
            self._qh_minutes_price = []
            self._qh_inc_dicts = []
            self._qh_dec_dicts = []
            

        self._current_idx += 1
        max_steps = self._start_idx + (self._nb_quarters_per_episode * 15)

        if self._current_idx < len(self._si_arr) and self._current_idx < max_steps:
            self._observation = self._get_current_observation(soc=new_soc)
            return self._observation, reward,fraction_of_max_power, False
        else:
            self._episode_ended = True
            self._observation = self._get_current_observation(soc=new_soc)
            return self._observation, reward,fraction_of_max_power, True

def importdata(path_training, path_validation, path_test, path_merit):
    df_train = pd.read_csv(path_training, sep=";") if path_training.endswith('.csv') else pd.read_excel(path_training)
    df_val = pd.read_csv(path_validation, sep=";") if path_validation.endswith('.csv') else pd.read_excel(path_validation)
    df_test= pd.read_csv(path_test, sep=";") if path_test.endswith('.csv') else pd.read_excel(path_test)
    
    with open(path_merit, "rb") as f:
        raw = pickle.load(f)
    merit_cache = {k: (Environment._unpack(i), Environment._unpack(d)) for k, (i, d) in raw.items()}
    del raw

    # Colonnes à scaler
    cols_to_scale = ["system imbalance", "ace", "igcc", "nrv_new"]

    scaler = StandardScaler()

    # Ajustement et transformation du train
    scaler.fit(df_train[cols_to_scale])

    # Gestion des moyennes et écarts-types personnalisés
    mean = np.append(scaler.mean_, 100) #type:ignore
    std = np.append(scaler.scale_, 150) #type:ignore

    
    print("--- Données chargées avec succès ---")
    return df_train, df_val, df_test, merit_cache, mean,std

def collect_data(environment, agent, steps, initial):
    total_reward = 0
    unfeas_action = []
    x = 16 * 20
    arbitrary_actions = np.random.uniform(-1, 1, 20)

    state = environment.reset()

    if initial == 0:
        for _ in range(int(x / 16)):
            for i in range(16):
                sample = np.array([arbitrary_actions[i]], dtype=np.float32)
                next_state, reward,_, done = environment.step(sample)
                agent.remember(state, sample, reward, next_state, done)
                total_reward += reward
                state = next_state
                if done:
                    state = environment.reset()

        for _ in range(steps - x):
            action = agent.choose_action(state)
            next_state, reward, _, done = environment.step(action)
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                state = environment.reset()

    else:
        for _ in range(steps):
            action = agent.choose_action(state)
            next_state, reward, _, done = environment.step(action)
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                state = environment.reset()

    return total_reward, unfeas_action