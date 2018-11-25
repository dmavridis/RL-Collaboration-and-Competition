# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import Agent
import torch
import torch.nn.functional as F
from utils import OUNoise, ReplayBuffer, soft_update

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'



class MADDPG:
    def __init__(self, state_size, action_size):
        super(MADDPG, self).__init__()
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, 1)

        self.maddpg_agent = [Agent(state_size, action_size, 0), 
                             Agent(state_size, action_size, 13)]
        
        
    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.actor_target for ddpg_agent in self.maddpg_agent]
        return target_actors

    def reset(self):
        for agent in self.maddpg_agent:
            agent.reset()

    def act(self, states):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(state) for agent, state in zip(self.maddpg_agent, states)]
        return actions
    
    def target_act(self, states):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(state) for ddpg_agent, state in zip(self.maddpg_agent, states)]
        return target_actions
    
    
    
    def step(self, states, actions, rewards, next_states, dones):
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.memory.add(s, a, r, ns, d)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            for a_i in range(2):
                samples = self.memory.sample(BATCH_SIZE)
                self.update(samples, a_i)
            self.update_targets() #soft update the target network towards the actual networks
        
        
    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = samples


        agent = self.maddpg_agent[agent_number]


        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = agent.actor_target(next_states)
        Q_targets_next = agent.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = agent.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = agent.actor_local(states)
        actor_loss = -agent.critic_local(states, actions_pred).mean()
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(), 1)
        agent.actor_optimizer.step()

    def update_targets(self):
        """soft update targets"""
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.actor_target, ddpg_agent.actor_local, TAU)
            soft_update(ddpg_agent.critic_target, ddpg_agent.critic_local, TAU)
            
    def save_to_file(self):
        for idx, agent in enumerate(self.maddpg_agent): 
            torch.save(agent.actor_local.state_dict(), './models/checkpoint_actor' + str(idx) + '.pth')
            torch.save(agent.critic_local.state_dict(), './models/checkpoint_critic' + str(idx) + '.pth')
        
    def load_from_file(self):
        return None