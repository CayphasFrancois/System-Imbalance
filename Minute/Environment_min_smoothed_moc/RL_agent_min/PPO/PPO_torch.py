import random
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

def set_seed(seed):
    """Fixe la graine aléatoire pour la reproductibilité (Python, NumPy, PyTorch, CUDNN)"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.cuda.manual_seed_all(seed) 
    
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False

# 1. Mémoire pour stocker les trajectoires (Rollout Buffer spécifique à PPO)
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        """Génère des mini-lots aléatoires pour l'optimisation par époques"""
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, prob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

# 2. Réseau Actor-Critic (Combine l'Acteur et le Critique)
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        
        # Partie commune / Partagée ou séparée (ici partagée pour simplifier)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Tête Acteur : prédit la moyenne des actions continues
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        
        # Écart-type (log_std) apprenable pour la distribution normale
        self.actor_log_std = nn.Parameter(T.zeros(1, action_dim))
        
        # Tête Critique : V(s)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))
        
        actor_mean = self.actor_mean(x)
        actor_log_std = self.actor_log_std.expand_as(actor_mean)
        actor_std = T.exp(actor_log_std)

        dist= Normal(actor_mean,actor_std)
        value = self.critic(x)
        return dist, value

# 3. Agent PPO complet
class PPOContinuousAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 policy_clip=0.2, batch_size=64, n_epochs=4):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        
        self.actor_critic = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        self.memory = PPOMemory(batch_size)

    def choose_action(self, state):
        """Sélectionne une action continue via la distribution Normale"""
        state_tensor = T.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with T.no_grad():
            dist, value = self.actor_critic(state_tensor)
            action = dist.sample() # Tire une action continue (ex: [0.45, -1.2])
            prob = dist.log_prob(action).sum(dim=-1) # Somme des log-probabilités si multi-actions
            
        return action.cpu().numpy()[0], prob.item(), value.item()

    def store_transition(self, state, action, prob, val, reward, done):
        self.memory.store_memory(state, action, prob, val, reward, done)

    def learn(self):
        """Mise à jour PPO pour actions continues"""
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, val_arr, reward_arr, done_arr, batches = \
                self.memory.generate_batches()

            # Calcul des avantages (GAE)
            advantages = np.zeros(len(reward_arr), dtype=np.float32)
            gae = 0
            for t in range(len(reward_arr) - 1):
                next_non_terminal = 1 - done_arr[t]
                delta = reward_arr[t] + self.gamma * val_arr[t+1] * next_non_terminal - val_arr[t]
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                advantages[t] = gae

            advantages = T.FloatTensor(advantages).to(self.device)
            values_tensor = T.FloatTensor(val_arr).to(self.device)
            returns = advantages + values_tensor

            for batch in batches:
                states = T.FloatTensor(state_arr[batch]).to(self.device)
                old_probs = T.FloatTensor(old_prob_arr[batch]).to(self.device)
                actions = T.FloatTensor(action_arr[batch]).to(self.device)

                dist, critic_value = self.actor_critic(states)
                critic_value = critic_value.squeeze(1)

                # Nouvelles log-probabilités (sommées sur les dimensions de l'action)
                new_probs = dist.log_prob(actions).sum(dim=-1)

                # Ratio de probabilité r_t(theta)
                prob_ratio = T.exp(new_probs - old_probs)

                # Perte Acteur clippée
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 
                                                 1 + self.policy_clip) * advantages[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # Perte Critique (MSE)
                returns_batch = returns[batch]
                critic_loss = nn.MSELoss()(critic_value, returns_batch)

                total_loss = actor_loss + 0.5 * critic_loss

                # Optimisation
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        self.memory.clear_memory()

    def save(self, filepath="ppo_continuous_checkpoint.pth"):
        T.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Agent PPO continu sauvegardé dans {filepath}")

    def load(self, filepath="ppo_continuous_checkpoint.pth"):
        checkpoint = T.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.actor_critic.eval()
        print(f"Agent PPO continu chargé depuis {filepath}.")