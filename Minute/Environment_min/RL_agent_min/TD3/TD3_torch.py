import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os as os
os.environ["PANDAPOWER_NUMBA"] = "False"
import random


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.cuda.manual_seed_all(seed) 
    
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False



class ReplayBuffer(object):
    def __init__(self,max_size,input_shape,n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size,*input_shape),dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size,*input_shape),dtype=np.float32) #* prend la forme de l'état
        self.action_memory = np.zeros ((self.mem_size, n_actions),dtype=np.float32)
        self.reward_memory = np.zeros (self.mem_size,dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.uint8)

    def store_transition (self , state , action , reward , state_ , done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer (self,batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.randint(0, max_mem, batch_size) #tire batch_size indice entre 0  et max_mem - 1     #Replace=False pour ne pas avoir de doublons, gros impact au début et après plus du tout 

        states  = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory [batch]
        actions = self.action_memory [batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal
    
class CriticNetwork (nn.Module) : # objectif du critic : estimer Q
    def __init__ (self,beta,input_dims,fc1_dims,fc2_dims,n_actions,name,chkpt_dir='C:/Document_UMONS/Thèse-Code'):
        super(CriticNetwork,self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_TD3')

        #Réseau critic 1

        # L'entrée combine la taille de l'état (*self.input_dims) et la taille de l'action (self.n_actions)
        input_dim_total = sum(input_dims) if isinstance(input_dims, (list, tuple)) else input_dims

        self.fc1 = nn.Linear(input_dim_total + self.n_actions,self.fc1_dims)     # type: ignore # première couche où on met que l'état
        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data,-f1, f1) # initialisation des poids
        T.nn.init.uniform_(self.fc1.bias.data,-f1,f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)  # normalise les activations

        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims) # deuxième couche (état transformé)
        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data,-f2,f2) 
        T.nn.init.uniform_(self.fc2.bias.data,-f2,f2)
        self.bn2= nn.LayerNorm (self.fc2_dims)

        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3,f3)
        T.nn.init.uniform_(self.q.bias.data,-f3,f3)

        #Réseau critic 2

        self.fc4 = nn.Linear(input_dim_total + self.n_actions, self.fc1_dims) # type: ignore # première couche où on met que l'état
        f4 = 1/np.sqrt(self.fc4.weight.data.size()[0])
        T.nn.init.uniform_(self.fc4.weight.data,-f4, f4) # initialisation des poids
        T.nn.init.uniform_(self.fc4.bias.data,-f4,f4)
        self.bn3 = nn.LayerNorm(self.fc1_dims)  # normalise les activations

        self.fc5 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f5 = 1/np.sqrt(self.fc5.weight.data.size()[0])
        T.nn.init.uniform_(self.fc5.weight.data,-f5,f5) 
        T.nn.init.uniform_(self.fc5.bias.data,-f5,f5)
        self.bn4= nn.LayerNorm (self.fc2_dims)

        self.q2 = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q2.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q2.bias.data, -f3, f3)
   
        self.optimizer = optim.Adam (self.parameters(),lr=beta)
        self.device = T.device ('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward (self,state,action) : 

        state_action = T.cat([state, action], dim=1)
        # Estimation Q1

        state_value1 = self.fc1(state_action)
        state_value1 = self.bn1(state_value1)
        state_value1 = F.relu(state_value1)
        state_value1 = self.fc2(state_value1)
        state_value1 = self.bn2 (state_value1)
        state_value1 = F.relu(state_value1)
        state_action_value1 = self.q(state_value1)

        # Estimation Q2

        state_value2 = self.fc4(state_action)
        state_value2 = self.bn3(state_value2)
        state_value2 = F.relu(state_value2)
        state_value2 = self.fc5(state_value2)
        state_value2 = self.bn4 (state_value2)
        state_value2 = F.relu(state_value2)
        state_action_value2 = self.q2(state_value2)

        return state_action_value1,state_action_value2
    
    def q1_only(self, state, action):

        # Pour calculer la valeur de la perte pour l'Actor (que Q1)µ
        state_action = T.cat([state, action], dim=1)
        s1 = F.relu(self.bn1(self.fc1(state_action)))
        s1 = F.relu(self.bn2(self.fc2(s1)))
        
        return self.q(s1)     

    def save_checkpoint (self) : 
        print('... saving checkpoint ...')
        T.save(self.state_dict(),self.checkpoint_file)    
    
    def load_checkpoint (self) : 
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module) : 
    def __init__ (self,alpha,input_dims,fc1_dims,fc2_dims,n_actions,name,chkpt_dir='C:/Document_UMONS/Thèse-Code') :
        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims 
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_TD3')
        self.fc1 = nn.Linear (*self.input_dims,self.fc1_dims) # type: ignore
        f1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2,f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2,f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3=0.003
        self.mu = nn.Linear (self.fc2_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data,-f3,f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3,f3)

        self.optimizer = optim.Adam (self.parameters(), lr=alpha)
        self.device = T.device ('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward (self,state):
        x= self.fc1(state)
        x= self.bn1(x)
        x= F.relu(x)
        x= self.fc2(x)
        x= self.bn2(x)
        x= F.relu(x)
        x= T.tanh(self.mu(x))

        return x

    def save_checkpoint (self) : 
        print('... saving checkpoint ...')
        T.save(self.state_dict(),self.checkpoint_file)    
    
    def load_checkpoint (self) : 
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent(object) : 
    def __init__(self,alpha,beta,input_dims,tau,env,gamma=0.99, n_actions=2, max_size= 1000000, layer1_size=400,layer2_size=300,batch_size=64,exploration_noise = 0.1, policy_noise=0.2,noise_clip=0.5,update_actor_iter=2,learn_step_cntr=0,seed=42,filename='_') :
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer (max_size,input_dims,n_actions)
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.update_actor_iter = update_actor_iter
        self.learn_step_cntr = learn_step_cntr
        self.exploration_noise = exploration_noise
        self.seed=seed
        self.filename=filename

        set_seed(seed)  

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions = n_actions, name='Actor',chkpt_dir=filename)

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions = n_actions, name='Target_Actor',chkpt_dir=filename)

        self.critic = CriticNetwork (beta, input_dims, layer1_size, layer2_size, n_actions = n_actions,name= 'Critic',chkpt_dir=filename)

        self.target_critic = CriticNetwork (beta, input_dims, layer1_size, layer2_size, n_actions = n_actions,name= 'Target_Critic',chkpt_dir=filename)

        self.target_actor.eval()
        self.target_critic.eval()       
        for param in self.target_actor.parameters():
            param.requires_grad = False
        for param in self.target_critic.parameters():
            param.requires_grad = False


        self.update_network_parameters(tau=1)


    def choose_action(self,observation) : 
        self.actor.eval() #obligatoire avec batchnorm, NN se met en mode évaluation
        observation = T.tensor(observation, dtype=T.float).unsqueeze(0).to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device) #action recommandée par le NN

        noise = T.randn_like(mu) * self.exploration_noise  
        mu_prime = mu + noise.to(self.actor.device) #ajoute le bruit
        mu_prime = T.clamp(mu_prime, -1, 1) 

        self.actor.train() #NN se met en mode entrainement
        return mu_prime.cpu().detach().numpy()

    def remember (self,state,action,reward, new_state,done) : 
        self.memory.store_transition(state,action,reward,new_state,done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size :
            return
        
        self.critic.train() 
        self.actor.train()
        self.target_actor.eval()
        self.target_critic.eval()

        state,action,reward,new_state,done = self.memory.sample_buffer(self.batch_size)
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device).view(-1, 1)
        done = T.tensor(done).to(self.critic.device).view(-1, 1)
        new_state = T.as_tensor(new_state,dtype=T.float).to(self.critic.device)
        action = T.tensor (action, dtype=T.float).to(self.critic.device)
        state = T.as_tensor (state, dtype=T.float).to(self.critic.device)


        # self.critic.eval()
        
        with T.no_grad():
            target_actions = self.target_actor.forward(new_state)
            xsi = T.randn_like(target_actions) * self.policy_noise # policy_noise = 0.2
            xsi = T.clamp(xsi, -self.noise_clip, self.noise_clip) # noise_clip = 0.5
            target_actions = T.clamp(target_actions + xsi, -1, 1)

        q1_, q2_ = self.target_critic.forward(new_state,target_actions)
        critic_value_= T.min (q1_,q2_)
        q1, q2 = self.critic.forward(state,action)

        target = reward + self.gamma * critic_value_ * done # on a stocké 1-done  # on calcul les targets pour calculer ensuite la fonction de coût 

        # self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target,q1) + F.mse_loss(target,q2)
        critic_loss.backward()
        # T.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)  #tentative
        self.critic.optimizer.step()

        # self.critic.eval()

        self.learn_step_cntr += 1
        if self.learn_step_cntr % self.update_actor_iter == 0 :
            self.actor.optimizer.zero_grad()
            mu = self.actor.forward(state)
            # self.actor.train()
            actor_loss = -self.critic.q1_only(state,mu) #on calcule les Q pour calculer la fonction de coût
            actor_loss = T.mean(actor_loss)
            actor_loss.backward()
            # T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) #tentative
            self.actor.optimizer.step()
            self.update_network_parameters()

        

    def update_network_parameters(self, tau=None) :
        if tau is None :
            tau=self.tau
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict (actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

            # Update Actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        # Update Critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_models (self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models (self) : 
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
