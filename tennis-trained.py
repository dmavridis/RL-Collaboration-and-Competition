from unityagents import UnityEnvironment
import numpy as np
import sys
sys.path.append('./code')
from maddpg import MADDPG

   
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
    agents.load_from_file()
    
    for i in range(1, 20):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        score = np.zeros((2,))
        agents.reset()

        for t in range(100):
            actions = agents.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            states = next_states
            score += np.array(rewards)
            if any(dones):
                break 

    env.close()
    
    