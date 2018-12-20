from network_env import Environment
from simple_DQN_modified import DeepQNetwork
# from simple_DQN_Prioritized_replay import DQNPrioritizedReplay
from simple_DQN_Prioritized_replay_Dueling import DQNPrioritizedReplay_Dueling
# from simple_DQN_double import DoubleDQN
# from simple_DQN_dueling import DuelingDQN
import numpy as np
import matplotlib.pyplot as plt
import configparser
import time

# 单次实验比较

TRAIN_EPISODE = 4000000
TEST_EPISODE = 100000

def train_routing():
    step = 0
    reward_sum = 0
    n_congest = 0
    reward_sum_list = []
    n_congest_list = []

    throughput = 0
    packet_loss_rate = 0

    for episode in range(TRAIN_EPISODE):
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
                # print(reward)
                if (episode > min_replay_history):
                    RL.learn()

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

        if (episode+1)%400000 == 0:
            RL.save_model(episode+1)

    print('training finished')

    return reward_sum_list,n_congest_list

def run_test():

    # no need to explore any more
    RL.epsilon=1.0

    final_reward_sum = 0
    final_n_congest = 0

    throughput = 0
    packet_loss_rate = 0


    for episode in range(TEST_EPISODE):
        # initial observation
        state = env.reset()

        while True:
            n = int(state[-1])
            
            if env.flow_demand_vector[n] == 0 and n != env.n_flows-1:
                state_ = state[:]
                state_[-1] += 1
            else:
                #normalize
                '''state_normalize = state[:]
                state_normalize[:env.n_flows] /= env.max_flow_demand 
                state_normalize[env.n_flows:(env.n_flows+env.n_links)] /= 100'''
                # RL choose action based on observation
                action = RL.choose_action(state)

                # RL take action and get next observation and reward
                state_, reward, done = env.step(state,action)
                '''state_normalize_ = state_[:]
                state_normalize_[:env.n_flows] /= env.max_flow_demand 
                state_normalize_[env.n_flows:(env.n_flows+env.n_links)] /= 100'''

                RL.store_transition(state, action, reward, state_)

                # swap observation
                state = state_

                # break while loop when end of this episode
                if done:
                    # print(reward)

                    final_reward_sum += reward

                    if reward <0:
                        final_n_congest += 1

                    t,l = loss_throughput()
                    throughput += t
                    packet_loss_rate += l

                    break

        if episode+1 == TEST_EPISODE:
            print('after 1M episode training: a_r:%f\tn_c:%d' %(final_reward_sum/TEST_EPISODE,final_n_congest/10))
            print('RL_routing throuput:%f\tloss:%f' %(throughput/TEST_EPISODE,packet_loss_rate/TEST_EPISODE))

'''
def full_flow_capacity_test():

    state = env.reset_max_capacity()

    while True:
        # RL choose action based on observation
        action = RL.choose_action(state)

        # RL take action and get next observation and reward
        state_, reward, done = env.step(state,action)

        # swap observation
        state = state_

        # break while loop when end of this episode
        if done:
            print(env.flow_routing)
            print(reward)

            break
'''

def run_random_routing():

    step = 0
    reward_sum = 0
    n_congest = 0
    reward_sum_list = []
    n_congest_list = []

    throughput = 0
    packet_loss_rate = 0

    for episode in range(TEST_EPISODE):
        # initial observation
        state = env.reset()

        while True:

            n = int(state[-1])
            
            if env.flow_demand_vector[n] == 0 and n != env.n_flows-1:
                state_ = state[:]
                state_[-1] += 1
            else:

                action = np.random.randint(0,env.n_actions)

                state_, reward, done = env.step(state,action)

                state = state_

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

        if (episode+1)%(TRAIN_EPISODE/200) == 0:
            # print('episode%d a_r:%f\tn_c:%d' %(episode+1,reward_sum/10000,n_congest))

            reward_sum_list.append(reward_sum/(TRAIN_EPISODE/200))
            n_congest_list.append(n_congest)
            reward_sum = 0
            n_congest = 0


    print('random_routing a_r:%f\tn_c:%d' %(np.array(reward_sum_list).mean(),np.array(n_congest_list).mean()))
    print('random_routing throuput:%f\tloss:%f' %(throughput/TEST_EPISODE,packet_loss_rate/TEST_EPISODE))

    return reward_sum_list,n_congest_list


def run_ecmp_routing():

    step = 0
    reward_sum = 0
    n_congest = 0
    reward_sum_list = []
    n_congest_list = []

    throughput = 0
    packet_loss_rate = 0

    for episode in range(TEST_EPISODE):
        # initial observation
        state = env.reset()

        reward = env.ecmp()

        reward_sum += reward

        if reward <0:
            n_congest += 1

        t,l = loss_throughput()
        throughput += t
        packet_loss_rate += l

        step += 1

        if (episode+1)%(TRAIN_EPISODE/200) == 0:
            # print('episode%d a_r:%f\tn_c:%d' %(episode+1,reward_sum/10000,n_congest))

            reward_sum_list.append(reward_sum/(TRAIN_EPISODE/200))
            n_congest_list.append(n_congest)
            reward_sum = 0
            n_congest = 0

    print('ecmp_routing a_r:%f\tn_c:%d' %(np.array(reward_sum_list).mean(),np.array(n_congest_list).mean()))
    print('ecmp_routing throuput:%f\tloss:%f' %(throughput/TEST_EPISODE,packet_loss_rate/TEST_EPISODE))

    return reward_sum_list,n_congest_list


def run_ospf_routing():

    step = 0
    reward_sum = 0
    n_congest = 0
    reward_sum_list = []
    n_congest_list = []

    throughput = 0
    packet_loss_rate = 0

    for episode in range(TEST_EPISODE):
        # initial observation
        state = env.reset()

        while True:

            n = int(state[-1])
            
            if env.flow_demand_vector[n] == 0 and n != env.n_flows-1:
                state_ = state[:]
                state_[-1] += 1
            else:

                action = 0

                state_, reward, done = env.step(state,action)

                state = state_

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

        if (episode+1)%(TRAIN_EPISODE/200) == 0:
            # print('episode%d a_r:%f\tn_c:%d' %(episode+1,reward_sum/10000,n_congest))

            reward_sum_list.append(reward_sum/(TRAIN_EPISODE/200))
            n_congest_list.append(n_congest)
            reward_sum = 0
            n_congest = 0


    print('ospf_routing a_r:%f\tn_c:%d' %(np.array(reward_sum_list).mean(),np.array(n_congest_list).mean()))
    print('ospf_routing throuput:%f\tloss:%f' %(throughput/TEST_EPISODE,packet_loss_rate/TEST_EPISODE))

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

'''
def show_reward(a,b,c):
    plt.plot(a,label='RL')
    plt.plot(b,label='random')
    plt.plot(c,label='ecmp')
    plt.ylabel('average reward')
    plt.xlabel('training steps(10k)')
    plt.legend()
    plt.show()


def show_congest(a,b,c):
    plt.plot(a,label='RL')
    plt.plot(b,label='random')
    plt.plot(c,label='ecmp')
    plt.ylabel('number of congestion')
    plt.xlabel('training steps(10k)')
    plt.legend()
    plt.show()


def show_congest(a,b,c):
    plt.plot(a,label='RL')
    plt.plot(b,label='random')
    plt.plot(c,label='ecmp')
    plt.ylabel('number of congestion')
    plt.xlabel('training steps(10k)')
    plt.legend()
    plt.show()
'''

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
    '''
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=learning_rate,
                      reward_decay=reward_decay,
                      e_greedy=epsilon_max,
                      replace_target_iter=target_update_iter,
                      memory_size=replay_capacity,
                      e_greedy_increment=e_greedy_increment,
                      output_graph=True
                      )
    
    
    RL = DQNPrioritizedReplay_Dueling(n_actions=env.n_actions, 
                              n_features=env.n_features, 
                              learning_rate=0.0002,
                              reward_decay=0.9,
                              e_greedy=0.9,
                              replace_target_iter=200,
                              memory_size=10000,
                              e_greedy_increment=0.0000005,  
                              prioritized=True,
                              dueling = True, 
                              n_neurons = 100,
                              # output_graph=True,
                              )
    '''
    r1,c1 = train_routing()
    '''
    for n in range(20):
        print('**************************************This is the',n+1,'times*********************************************')
        print('max_flow_demand = ',env.max_flow_demand)
        run_test()
        r2,c2 = run_random_routing()
        r3,c3 = run_ecmp_routing()
        r4,c4 = run_ospf_routing()
        env.max_flow_demand -= 2
    '''

    '''
    with open('data.txt','a') as f:
        f.write('reward:%s' % r1)
        f.write('\n\ncongestion:%s' % c1)
    '''
    
    # full_flow_capacity_test()


    #show_reward(r1,r2,r3)
    #show_congest(c1,c2,c3)