from network_env import Environment
from simple_DQN_Prioritized_replay_Dueling import DQNPrioritizedReplay_Dueling
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 固定threshold变化flow大小做趋势图

TRAIN_EPISODE = 1000000
TEST_EPISODE = 100000
ITER_EPISODE = 10000


def save_data():
    with open('data/data.txt','a') as f:
        f.write()

def train_routing():
    step = 0
    reward_sum = 0
    n_congest = 0
    reward_sum_list = []
    n_congest_list = []
    for episode in range(TRAIN_EPISODE):
        # initial observation
        state = env.reset()

        while True:
            # RL choose action based on observation
            action = RL.choose_action(state)

            # RL take action and get next observation and reward
            state_, reward, done = env.step(state,action)

            RL.store_transition(state, action, reward, state_)

            # if (step > 200) and (step % 5 == 0):
                # RL.learn()

            # swap observation
            state = state_

            # break while loop when end of this episode
            if done:
                # print(reward)
                if (step > 600):
                    RL.learn()

                reward_sum += reward

                if reward <0:
                    n_congest += 1

                break

            step += 1

        if (episode+1)%ITER_EPISODE == 0:
            # print('threshold:%d episode%d a_r:%f\tn_c:%d' %(env.link_capacity_threshold,episode+1,reward_sum/ITER_EPISODE,n_congest))
            print('episode%d a_r:%f\tn_c:%d' %(episode+1,reward_sum/ITER_EPISODE,n_congest))
            # f.write('\nthreshold:%d episode%d a_r:%f\tn_c:%d' %(env.link_capacity_threshold,episode+1,reward_sum/ITER_EPISODE,n_congest))
            f.write('\nepisode%d a_r:%f\tn_c:%d' %(episode+1,reward_sum/ITER_EPISODE,n_congest))

            reward_sum_list.append(reward_sum/ITER_EPISODE)
            n_congest_list.append(n_congest)
            reward_sum = 0
            n_congest = 0

    print('training finished')
    f.write('\ntraining finished')

    return reward_sum_list,n_congest_list


def run_test():

    RL.epsilon=1.0

    final_reward_sum = 0
    final_n_congest = 0

    for episode in range(TEST_EPISODE):
        # initial observation
        state = env.reset()

        while True:
            # RL choose action based on observation
            action = RL.choose_action(state)

            # RL take action and get next observation and reward
            state_, reward, done = env.step(state,action)

            # swap observation
            state = state_

            # break while loop when end of this episode
            if done:
                # print(reward)

                final_reward_sum += reward

                if reward <0:
                    final_n_congest += 1

                break

        if episode+1 == TEST_EPISODE:
            print('after training: a_r:%f\tn_c:%d' %(final_reward_sum/TEST_EPISODE,final_n_congest/10))
            f.write('\nafter training: a_r:%f\tn_c:%d' %(final_reward_sum/TEST_EPISODE,final_n_congest/10))

    return final_reward_sum/TEST_EPISODE,final_n_congest/10

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
            f.write('\nfull_flow_capacity_test:%f' % reward)

            break


def run_random_routing():

    step = 0
    reward_sum = 0
    n_congest = 0
    reward_sum_list = []
    n_congest_list = []

    for episode in range(TEST_EPISODE):
        # initial observation
        state = env.reset()

        while True:

            action = np.random.randint(0,env.n_actions)

            state_, reward, done = env.step(state,action)

            state = state_

            # break while loop when end of this episode
            if done:

                reward_sum += reward

                if reward <0:
                    n_congest += 1

                break

            step += 1

        if (episode+1)%(ITER_EPISODE) == 0:
            # print('episode%d a_r:%f\tn_c:%d' %(episode+1,reward_sum/ITER_EPISODE,n_congest))
            reward_sum_list.append(reward_sum/ITER_EPISODE)
            n_congest_list.append(n_congest)
            reward_sum = 0
            n_congest = 0

    a_r = np.array(reward_sum_list).mean()
    n_c = np.array(n_congest_list).mean()
    #print(reward_sum_list,n_congest_list)

    print('random_routing a_r:%f\tn_c:%d' % (a_r,n_c))
    f.write('\nrandom_routing a_r:%f\tn_c:%d' % (a_r,n_c))

    return reward_sum_list, n_congest_list, a_r, n_c


def run_ecmp_routing():

    step = 0
    reward_sum = 0
    n_congest = 0
    reward_sum_list = []
    n_congest_list = []

    for episode in range(TEST_EPISODE):
        # initial observation
        state = env.reset()

        reward = env.ecmp()

        reward_sum += reward

        if reward <0:
            n_congest += 1


        step += 1

        if (episode+1)%(ITER_EPISODE) == 0:
            # print('episode%d a_r:%f\tn_c:%d' %(episode+1,reward_sum/ITER_EPISODE,n_congest))
            reward_sum_list.append(reward_sum/ITER_EPISODE)
            n_congest_list.append(n_congest)
            reward_sum = 0
            n_congest = 0

    a_r = np.array(reward_sum_list).mean()
    n_c = np.array(n_congest_list).mean()
    #print(reward_sum_list,n_congest_list)

    print('ecmp_routing a_r:%f\tn_c:%d' % (a_r,n_c))
    f.write('\necmp_routing a_r:%f\tn_c:%d' % (a_r,n_c))

    return reward_sum_list, n_congest_list, a_r, n_c


def run_shortest_routing():

    step = 0
    reward_sum = 0
    n_congest = 0
    reward_sum_list = []
    n_congest_list = []

    for episode in range(TEST_EPISODE):
        # initial observation
        state = env.reset()

        while True:

            action = np.random.randint(0,env.n_actions)

            state_, reward, done = env.step(state,action)

            state = state_

            # break while loop when end of this episode
            if done:

                reward_sum += reward

                if reward <0:
                    n_congest += 1

                break

            step += 1

        if (episode+1)%(ITER_EPISODE) == 0:
            # print('episode%d a_r:%f\tn_c:%d' %(episode+1,reward_sum/ITER_EPISODE,n_congest))
            reward_sum_list.append(reward_sum/ITER_EPISODE)
            n_congest_list.append(n_congest)
            reward_sum = 0
            n_congest = 0

    a_r = np.array(reward_sum_list).mean()
    n_c = np.array(n_congest_list).mean()

'''
def save_reward(a,b,c):
    plt.plot(a,label='RL')
    plt.plot(b,label='random')
    plt.plot(c,label='ecmp')
    plt.ylabel('average reward')
    plt.xlabel('training steps(10k)')
    plt.legend()
    plt.savefig("data/%d %s r.jpg" % (env.link_capacity_threshold,datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')))
    plt.close('all')


def save_congest(a,b,c):
    plt.plot(a,label='RL')
    plt.plot(b,label='random')
    plt.plot(c,label='ecmp')
    plt.ylabel('number of congestion')
    plt.xlabel('training steps(10k)')
    plt.legend()
    plt.savefig("data/%d %s c.jpg" % (env.link_capacity_threshold,datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')))
    plt.close('all')

def save_threshold_reward(a,b,c,d):
    plt.plot(d,a,label='RL',linestyle='-')
    plt.plot(d,b,label='random',linestyle=':')
    plt.plot(d,c,label='ecmp',linestyle='--')
    plt.ylabel('average reward')
    plt.xlabel('threshold')
    plt.legend()
    plt.savefig("data/threshold %d %s r.jpg" % (env.link_capacity_threshold,datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')))
    plt.close('all')


def save_threshold_congest(a,b,c,d):
    plt.plot(d,a,label='RL',linestyle='-')
    plt.plot(d,b,label='random',linestyle=':')
    plt.plot(d,c,label='ecmp',linestyle='--')
    plt.ylabel('number of congestion')
    plt.xlabel('threshold')
    plt.legend()
    plt.savefig("data/threshold %d %s c.jpg" % (env.link_capacity_threshold,datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')))
    plt.close('all')
'''
if __name__ == "__main__":
    # maze game
    env = Network_Env()

    trained_reward = []
    trained_congest = []
    random_reward = []
    random_congest = []
    ecmp_reward = []
    ecmp_congest = []
    threshold = []

    for i in range(100):


        #env.random_link_capacity_threshold()
        #print(env.link_capacity_different_threshold)
        
        threshold.append(env.link_capacity_threshold)

        with open('data/data.txt','a') as f:
            f.write('\n\naverage flow demand:%f' % ((env.flow_capacity_increase+10)/2.))

            RL_class = {}

            RL_class[i] = DQNPrioritizedReplay_Dueling(n_actions=env.n_actions, 
                                      n_features=env.n_features, 
                                      learning_rate=0.00005,
                                      reward_decay=0.9,
                                      e_greedy=0.9,
                                      replace_target_iter=200,
                                      memory_size=10000,
                                      e_greedy_increment=0.000002, 
                                      prioritized=True,
                                      dueling = True, 
                                      n_neurons=60
                                      # output_graph=True,
                                      )

            RL = RL_class[i]

            r1,c1 = train_routing()
            # full_flow_capacity_test()
            trained_r1,trained_c1 = run_test()  

            r2,c2,a_r2,n_c2= run_random_routing()
            r3,c3,a_r3,n_c3= run_ecmp_routing()
            
            trained_reward.append(trained_r1)
            trained_congest.append(trained_c1)
            random_reward.append(a_r2)
            random_congest.append(n_c2)
            ecmp_reward.append(a_r3)
            ecmp_congest.append(n_c3)

            
        #save_reward(r1,r2,r3)
        #save_congest(c1,c2,c3)

        env.flow_capacity_increase += 0.5

    #save_threshold_reward(trained_reward,random_reward,ecmp_reward,((env.flow_capacity_increase+10)/2.))
    #save_threshold_congest(trained_congest,random_congest,ecmp_congest,((env.flow_capacity_increase+10)/2.))

    print(trained_reward,'\n',random_reward,'\n',ecmp_reward)
    print(trained_congest,'\n',random_congest,'\n',ecmp_congest)
    '''
    print('r-r:',np.array(random_reward).mean(),'max:',np.array(random_reward).max(),'min:',np.array(random_reward).min())
    print('r-c:',np.array(random_congest).mean(),'max:',np.array(random_congest).max(),'min:',np.array(random_congest).min())
    print('e-r:',np.array(ecmp_reward).mean(),'max:',np.array(ecmp_reward).max(),'min:',np.array(ecmp_reward).min())
    print('e-c:',np.array(ecmp_congest).mean(),'max:',np.array(ecmp_congest).max(),'min:',np.array(ecmp_congest).min())
    '''