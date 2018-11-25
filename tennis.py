from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import sys
sys.path.append('./code')
from maddpg import MADDPG

sys.path.append('./code')

def ddpg(n_episodes=3000, max_t=300):
    scores_deque = deque(maxlen=100)
    scores = []
    scores_avg = []
    score_target = 0.50

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros((2,))
        agents.reset()

        for t in range(max_t):
            actions = agents.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agents.step(states, actions, rewards, next_states, dones)


            states = next_states
            score += np.array(rewards)
            if any(dones):
                break 

        
        scores_deque.append(np.max(score))
        scores.append(np.max(score))
        scores_avg.append(np.mean(scores_deque))

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), np.mean(score)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
                   
        if np.mean(scores_deque) >= score_target:
            # save the weights
            agents.save_to_file()
            print("\n Agents saved for score {:.2f}".format(score_target))
            score_target += 0.01
        
    return scores, scores_avg
    
if __name__ == "__main__":
    

    env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    
    # number of agents 
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    
    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)
    
    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0]) 

    agents = MADDPG(state_size, action_size)

    scores, scores_avg = ddpg()
    env.close()
    
    
    fig = plt.figure(figsize=(12,7))
    plt.grid()
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.plot(np.arange(1, len(scores)+1), scores_avg,'r',  linewidth=2)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.axhline(y=0.5, color='g')
    plt.show()
