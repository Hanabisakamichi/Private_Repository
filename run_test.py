from network_env import Environment
from simple_DQN_modified import DeepQNetwork
from simple_DQN_Prioritized_replay_Dueling import DQNPrioritizedReplay_Dueling

import numpy as np
import matplotlib.pyplot as plt
import configparser
import time

TEST_EPISODE = 10000

def test_routing():
    step = 0
    reward_sum = 0
    n_congest = 0
    reward_sum_list = []
    n_congest_list = []

    throughput = 0
    packet_loss_rate = 0

    for episode in range(TEST_EPISODE):
        # initial observation
        raw_state,input_state = env.reset()

        while True:
            #normalize
            '''state_normalize = state[:]
            state_normalize[:env.n_flows] /= env.max_flow_demand
            state_normalize[env.n_flows:(env.n_flows+env.n_links)] /= 100'''
            
            # RL choose action based on observation
            #s = time.time()
            action = RL.choose_action(input_state)
            #e = time.time()
            #print('choose_action',e-s)
            # RL take action and get next observation and reward
            #s = time.time()
            raw_state_, reward, done = env.step(raw_state,action)
            #e = time.time()
            #print('step',e-s)
            '''state_normalize_ = state_[:]
            state_normalize_[:env.n_flows] /= env.max_flow_demand 
            state_normalize_[env.n_flows:(env.n_flows+env.n_links)] /= 100'''
            #s = time.time()
            input_state_ = env.get_next_input_state(raw_state_,input_state)
            #e = time.time()
            #print('get_input_state',e-s)
            #s = time.time()
            RL.store_transition(input_state, action, reward, input_state_)
            #e = time.time()
            #print('store_transition',e-s)
            # if (step > 200) and (step % 5 == 0):
                # RL.learn()

            # swap observation
            raw_state = raw_state_
            input_state = input_state_

            # break while loop when end of this episode
            if done:
                reward_sum += reward

                if reward <0:
                    n_congest += 1

                t,l = loss_throughput()
                throughput += t
                packet_loss_rate += l

                break

            step += 1

        if (episode+1)%10000 == 0:
            print('episode%d a_r:%f\tn_c:%d' %(episode+1,reward_sum/10000,n_congest))
            print('RL_routing throuput:%f\tloss:%f' %(throughput/10000,packet_loss_rate/10000))

            throughput = 0
            packet_loss_rate = 0

            reward_sum_list.append(reward_sum/10000)
            n_congest_list.append(n_congest)
            reward_sum = 0
            n_congest = 0

    print('test finished')

    return reward_sum_list,n_congest_list

def loss_throughput():
    sum_load_under_threshold = 0
    for i in range(env.n_links):
        if env.total_link_load[i] >= env.links_bandwidth_threshold[i]:
            sum_load_under_threshold += env.links_bandwidth_threshold[i]
        else:
            sum_load_under_threshold += env.total_link_load[i]
    
    if sum(env.total_link_load) == 0:
        return 0,0
    else:
        throughput_rate = sum_load_under_threshold/sum(env.total_link_load)
        throughput = sum(env.flow_demand_vector)*throughput_rate
        packet_loss_rate = 1-sum_load_under_threshold/sum(env.total_link_load)
    
        return throughput,packet_loss_rate
        
if __name__ == "__main__":

    cf = configparser.ConfigParser()
    cf.read('config.ini', encoding='utf8')
    learning_rate = cf.getfloat('dqn', 'learning_rate')
    reward_decay = cf.getfloat('dqn', 'reward_decay')
    epsilon_max = cf.getfloat('dqn', 'epsilon_max')
    e_greedy_increment = cf.getfloat('dqn', 'e_greedy_increment')
    replay_capacity = cf.getint('dqn', 'replay_capacity')
    target_update_iter = cf.getint('dqn', 'target_update_iter')
    min_replay_history = cf.getfloat('dqn', 'min_replay_history')

    env = Environment()
    
    RL = DQNPrioritizedReplay_Dueling(n_actions=env.n_actions, 
                              n_features=env.n_features, 
                              learning_rate=learning_rate,
                              reward_decay=reward_decay,
                              e_greedy=epsilon_max,
                              replace_target_iter=target_update_iter,
                              memory_size=replay_capacity,
                              e_greedy_increment=e_greedy_increment,  
                              prioritized=True,
                              dueling = True, 
                              n_neurons = 100,
                              # output_graph=True,
                              )

    RL.load_model('./model/checkpoint-1000')

    test_routing()

