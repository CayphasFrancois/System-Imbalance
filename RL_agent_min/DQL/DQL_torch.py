import random
import os
from collections import deque
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

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

# 1. Architecture du Réseau Q
class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = T.relu(self.fc1(x))
        x = T.relu(self.fc2(x))
        return self.fc3(x)

# 2. Mémoire Tampon (Experience Replay)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            T.FloatTensor(np.array(state)),
            T.LongTensor(action),
            T.FloatTensor(reward),
            T.FloatTensor(np.array(next_state)),
            T.FloatTensor(done)
        )
        
    def __len__(self):
        return len(self.buffer)

# 3. Agent DQL complet avec fonctions d'apprentissage
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, buffer_capacity=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma # Facteur d'actualisation
        
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        
        # Réseau de politique et Réseau cible (Target Network) pour la stabilité
        self.policy_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.memory = ReplayBuffer(buffer_capacity)

    def select_action(self, state, epsilon=0.0):
        """Sélectionne une action via une politique epsilon-greedy (exploration / exploitation)"""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        
        with T.no_grad():
            state_tensor = T.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return T.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Enregistre une transition dans le buffer"""
        self.memory.push(state, action, reward, next_state, done)

    def update(self, batch_size=128):
        """Fonction principale d'apprentissage (Backpropagation)"""
        if len(self.memory) < batch_size:
            return  # Pas assez de données pour apprendre

        # 1. Échantillonner un batch depuis la mémoire
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 2. Calculer les Q-valeurs actuelles Q(s, a)
        q_values = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 3. Calculer les Q-valeurs cibles y = r + gamma * max Q_target(s', a')
        with T.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            expected_state_action_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # 4. Calculer la perte (Loss)
        loss = self.criterion(state_action_values, expected_state_action_values)

        # 5. Optimisation du réseau
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Met à jour le réseau cible avec les poids du réseau de politique"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath="dqn_checkpoint.pth"):
        """Sauvegarde l'agent (poids et optimiseur pour reprendre l'entraînement)"""
        T.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Agent sauvegardé dans {filepath}")

    def load(self, filepath="dqn_checkpoint.pth"):
        """Charge un agent existant pour reprendre ou continuer l'apprentissage"""
        checkpoint = T.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_net.train()
        print(f"Agent chargé avec succès depuis {filepath} (Prêt à continuer l'apprentissage).")