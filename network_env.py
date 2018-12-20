import numpy as np
import random
import network

# choose which type of network to run on, two topology 'NSFNET' or 'FATTREE'
NETWORK_TYPE = 'S_FATTREE'
N_CANDIDATE_ROUTING_PATH = 4
FLOW_DEMAND_MAX = 0.25
FLOW_DENSITY = 0.25

class Environment():
    def __init__(self):

        # ----------------------- network parameter configuration -------------------------

        if NETWORK_TYPE == 'NSFNET':
            self.network_config = network.NSFNET()
        elif NETWORK_TYPE == 'FATTREE':
            self.network_config = network.FATTREE()
        elif NETWORK_TYPE == 'TEST':
            self.network_config = network.TEST()
        elif NETWORK_TYPE == 'CELLCULAR':
            self.network_config = network.CELLCULAR()
        elif NETWORK_TYPE == 'S_FATTREE':
            self.network_config = network.S_FATTREE()
        elif NETWORK_TYPE == 'B_FATTREE':
            self.network_config = network.B_FATTREE()
        else:
            print('Wrong network type')

        self.n_nodes = self.network_config.n_nodes                  # number of nodes
        self.n_hosts = self.network_config.n_hosts                  # number of hosts
        self.n_links = self.network_config.n_links                  # number of links
        if NETWORK_TYPE == 'S_FATTREE' or NETWORK_TYPE == 'B_FATTREE':
            self.n_flows = self.n_hosts*(self.n_hosts-2)    # not care flow between two hosts which have same access layer node
        else:
            self.n_flows = self.n_hosts*(self.n_hosts-1)    # number of flows

        self.links_bandwidth_matrix = self.network_config.links_bandwidth   # bandwidth of all links 
        self.links_bandwidth_threshold = self.network_config.links_bandwidth[2]

        self.total_link_load = np.zeros(self.n_links)

        # ----------------------- flows demand configuration -------------------------

        self.max_flow_demand = FLOW_DEMAND_MAX
        self.flow_demand_vector = np.zeros(self.n_flows)
        self.flow_routing = np.zeros([self.n_flows,self.n_links])

        # ----------------------- candidate path configuration -------------------------

        # calculate k shortest path between source and destination with YEN algorithm
        # self.y = yen.Yen(self.n_nodes,self.n_links,self.links_bandwidth_matrix)

        # 3-dimensional matrix, storing the candidate paths of all flows, links in one path are stored as indices
        # self.index_routing_links = self.get_index_routing_links()

        self.index_routing_links = self.network_config.index_routing_links

        print(self.index_routing_links)

        # use flow priority algorithm to reorder the index_routing_links matrix
        # self.flow_sort_by_priority()

        # 3-dimensional matrix, storing the candidate paths of all flows, links in one path are stored as vector 
        self.accessible_routing_links = self.set_routing_links()

        #self.links_count_for_shortest_paths()
        #print(self.accessible_routing_links)

        # ----------------------- other configuration -------------------------

        # n_actions and n_features are used as parameters for DNN
        self.n_actions = N_CANDIDATE_ROUTING_PATH
        if NETWORK_TYPE == 'S_FATTREE':
            self.n_features = self.n_flows + self.n_links + 6 + 1 # 1->current flow  6->current order number
        if NETWORK_TYPE == 'B_FATTREE':
            self.n_features = self.n_flows + self.n_links + 8 + 1 # 1->current flow  8->current order number


    def get_index_routing_links(self):

        index_routing_links = []
        for i in range(self.n_hosts):
            for j in range(self.n_hosts):
                if i == j:
                    continue
                else:
                    index_routing_links.append(self.y.YEN_path(i,j,N_CANDIDATE_ROUTING_PATH))
        return index_routing_links

    # get accessible_routing_links matrix from index_routing_links matrix, accessible_routing_links will be used in other functions
    def set_routing_links(self):

        accessible_routing_links = np.zeros([self.n_flows,N_CANDIDATE_ROUTING_PATH,self.n_links])

        for i in range(len(self.index_routing_links)):
            for j in range(len(self.index_routing_links[i])):
                for link in self.index_routing_links[i][j]:
                    accessible_routing_links[i][j][link] = 1

        return accessible_routing_links

    def links_count_for_shortest_paths(self):

        count = {}

        for i in range(len(self.index_routing_links)):
            for j in self.index_routing_links[i][0]:
                if j not in count:
                    count[j] = 1
                else:
                    count[j] += 1

        print(count)

        return 0

    # flow priority algorithm which reorder the index_routing_links matrix
    def flow_sort_by_priority(self):

        # expect available remaining bandwidth of every link
        earb_l = {}

        for flow in range(self.n_flows):
            for path in range(len(self.index_routing_links[flow])):
                for link in self.index_routing_links[flow][path]:
                    expect_load = (self.max_flow_demand*FLOW_DENSITY/2)/len(self.index_routing_links[flow])

                    if link not in earb_l:
                        earb_l[link] = self.links_bandwidth_threshold[link]-expect_load
                    else:
                        earb_l[link] -= expect_load

        self.index_routing_links.sort(key=lambda x:self.flow_priority(x,earb_l),reverse = True)


    # calculate the priority value of every flow for flow priority algorithm
    def flow_priority(self, dandidate_paths_of_one_flow, earb_l):

        # priority value of all dandidate paths
        pv_adp = []

        for path in dandidate_paths_of_one_flow:
            earb_in_path = []
            for link in path:
                earb_in_path.append(earb_l[link])
            pv_adp.append(min(earb_in_path))

        pv_adp.sort(reverse = True)

        # priority value of this flow
        pv_f = pv_adp[0]-pv_adp[1]

        return pv_f

    # generate stochastic flow demand for every source-destination pair in network
    def generate_stochastic_flow_demand(self):

        for i in range(self.n_flows):
            if random.random() < FLOW_DENSITY:
                self.flow_demand_vector[i]=random.uniform(0.01,self.max_flow_demand)

        return None

    # after every time we choose candidate path for one flow, update the traffic load in all links
    def update_total_link_load(self, n):

        self.total_link_load += np.dot(self.flow_demand_vector[n],self.flow_routing[n])

        return None

    # get raw_state for input_state
    def get_raw_state(self, n):

        raw_state = np.hstack((self.flow_demand_vector,self.total_link_load,n))

        return raw_state

    def convolve5(self,array):

        new_array = []
        l =len(array)
        array = np.hstack((np.zeros(5),array,np.zeros(5)))
        for i in range(l):
            target = array[i]*0.2+array[i+1]*0.4+array[i+2]*0.6+array[i+3]*0.8+array[i+4]*1+array[i+5]*2+array[i+6]*1+array[i+7]*0.8+array[i+8]*0.6+array[i+9]*0.4+array[i+10]*0.2
            new_array.append(target/8)
        return new_array

    def convolve10(self,array):

        new_array = []
        l =len(array)
        array = np.hstack((np.zeros(10),array,np.zeros(10)))
        for i in range(l):
            target = 0
            for j in range(10):
                target += array[i+j]*0.1*(j+1)
            target += array[i+10]*4
            for k in range(10):
                target += array[i+11+k]*0.1*(10-k)
            new_array.append(target/15)
        return new_array

    def convolve20(self,array):

        new_array = []
        l =len(array)
        array = np.hstack((np.zeros(20),array,np.zeros(20)))
        for i in range(l):
            target = 0
            for j in range(20):
                target += array[i+j]*0.05*(j+1)
            target += array[i+20]*4
            for k in range(20):
                target += array[i+21+k]*0.05*(20-k)
            new_array.append(target/25)
        return new_array

    def dec2bin(self,num):

        res = []
        while (num!=0):
            res.append(num%2)
            num = int(num/2)
        if NETWORK_TYPE == 'S_FATTREE':
            return [0 for x in range(6-len(res))] + res[::-1]
        if NETWORK_TYPE == 'B_FATTREE':
            return [0 for x in range(8-len(res))] + res[::-1]
        else:
            return None

    # get raw_state for input to DQN
    def get_start_input_state(self, raw_state):

        if NETWORK_TYPE == 'B_FATTREE':
            s1 = self.convolve20(raw_state[:self.n_flows])
        if NETWORK_TYPE == 'S_FATTREE':
            s1 = self.convolve5(raw_state[:self.n_flows])
        s2 = self.convolve5(raw_state[self.n_flows:-1])
        s3 = self.dec2bin(raw_state[-1])
        s4 = raw_state[int(raw_state[-1])]
        input_state = np.hstack((s1,s2,s3,s4))

        return input_state

    def get_next_input_state(self, raw_state, input_state):

        s1 = input_state[:self.n_flows]
        
        if raw_state[int(raw_state[-1])-1] == 0:
            s2 = input_state[self.n_flows:self.n_flows+self.n_links]
        else:
            s2 = self.convolve5(raw_state[self.n_flows:-1])

        s3 = self.dec2bin(raw_state[-1])
        s4 = raw_state[int(raw_state[-1])]

        input_state_ = np.hstack((s1,s2,s3,s4))

        return input_state_

    # get reward at every step
    def get_reward(self):

        high_than_threshold_max = 0
        high_than_threshold_sum = 0
        low_than_threshold_min = -min(self.links_bandwidth_threshold)

        for i in range(self.n_links):
            d_value = self.total_link_load[i] - self.links_bandwidth_threshold[i]
            if d_value > 0:
                high_than_threshold_sum += d_value
                if d_value > high_than_threshold_max:
                    high_than_threshold_max = d_value
            else:
                if d_value > low_than_threshold_min:
                    low_than_threshold_min = d_value

        if high_than_threshold_max == 0 and high_than_threshold_sum == 0:
            return -low_than_threshold_min
        else:
            return -(0.5*high_than_threshold_sum+high_than_threshold_max)

    # reset the specific parameter for every new episode
    def reset(self):

        self.flow_demand_vector = np.zeros(self.n_flows)
        self.flow_routing = np.zeros([self.n_flows,self.n_links])
        self.total_link_load = np.zeros(self.n_links)

        self.generate_stochastic_flow_demand()

        raw_state = self.get_raw_state(0)

        return raw_state, self.get_start_input_state(raw_state)   #raw state and its convolution

    def step(self, raw_state, action):

        n = int(raw_state[-1])
        '''
        while(self.flow_demand_vector[n] == 0 and n != self.n_flows-1):
            n += 1
            raw_state_ = raw_state[:]
            raw_state_[-1] += 1
        '''
        # if there are not enough candidate routing path for flow, choose the shortest one
        if any(self.accessible_routing_links[n][action]) == True:
            self.flow_routing[n] = self.accessible_routing_links[n][action]
        else:
            self.flow_routing[n] = self.accessible_routing_links[n][0]

        self.update_total_link_load(n)

        raw_state_ = self.get_raw_state(n+1)

        if n < self.n_flows-1:
            reward = 0
            done = False
        else:
            reward = self.get_reward()
            done = True

        return raw_state_, reward, done

    def ecmp(self):
        
        # equal_flow_demand = self.flow_demand_vector.repeat(self.n_actions)/self.n_actions

        equal_flow_demand = np.zeros(self.n_flows*self.n_actions)

        for index,i in enumerate(self.index_routing_links):
            l = len(i)
            for j in range(l):
                equal_flow_demand[self.n_actions*index+j] = self.flow_demand_vector[index]/l

        equal_flow_routing = self.accessible_routing_links.reshape([self.n_flows*self.n_actions ,self.n_links])

        self.total_link_load = np.dot(equal_flow_demand,equal_flow_routing)

        reward = self.get_reward()

        return reward

    def step_mp(self, raw_state ,action):

        n = int(raw_state[-1])
        '''
        if self.flow_demand_vector[n] == 0 and n != self.n_flows-1:
            state_ = state[:]
            state_[-1] += 1
        else: 
        '''
        # if there are not enough candidate routing path for flow, choose the shortest one
        for i in range(self.n_actions):
            if any(self.accessible_routing_links[n][i]) == True:
                self.total_link_load += self.accessible_routing_links[n][i]*self.flow_demand_vector[n]*action[i]
            else:
                self.total_link_load += self.accessible_routing_links[n][0]*self.flow_demand_vector[n]*action[i]

        raw_state_ = self.get_raw_state(n+1)

        if n != self.n_flows-1:
            reward = 0
            done = False
        else:
            reward = self.get_reward()
            done = True

        return raw_state_, reward, done



'''
env = Environment()
state = env.reset()
print(env.accessible_routing_links)
env.step(state,1)
print(env.flow_demand_vector)
print(env.flow_routing)
print(env.total_link_load)
env.ecmp()
print()
'''

