import numpy as np
import random

SAME_BANDWIDTH = True

class NSFNET():
	def __init__(self):

		bandwidth = 100
		
		# number of nodes in the network
		self.n_nodes = 14
		# number of hosts in the network
		self.n_hosts = 14
		# number of links in the network
		self.n_links = 21
		# bandwidth of all links
		self.links_bandwidth = np.array([
			[0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 8,10,10,11,12],		# source nodes
			[1, 2, 7, 2, 3, 5, 4,10, 5, 6, 9,12, 7, 8, 9,11,13,11,13,13,13],		# destination nodes
			[1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1]])		# bandwidth

		# whether all links get same bandwidth
		if SAME_BANDWIDTH:
			self.links_bandwidth[2] = np.ones(self.n_links)*bandwidth
		else:
			for i in range(self.n_links):
				self.links_bandwidth[2][i] = random.uniform(-20,20)+bandwidth

class FATTREE():
	def __init__(self):
		
		bandwidth = 100
		
		# number of nodes in the network
		self.n_nodes = 20
		# number of hosts in the network
		self.n_hosts = 16
		# number of links in the network
		self.n_links = 32
		# bandwidth of all links
		self.links_bandwidth = np.array([
			[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9 ,9,10,10,11,11,12,12,13,13,14,14,15,15],
			[8, 9, 8 ,9,10,11,10,11,12,13,12,13,14,15,14,15,16,17,18,19,16,17,18,19,16,17,18,19,16,17,18,19],
			[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

		# whether all links get same bandwidth
		if SAME_BANDWIDTH:
			self.links_bandwidth[2] = np.ones(self.n_links)*bandwidth
		else:
			for i in range(self.n_links):
				self.links_bandwidth[2][i] = random.uniform(-25,25)+bandwidth

class CELLCULAR():
	def __init__(self):

		bandwidth = 100
		
		# number of nodes in the network
		self.n_nodes = 14
		# number of hosts in the network
		self.n_hosts = 14
		# number of links in the network
		self.n_links = 29
		# bandwidth of all links
		self.links_bandwidth = np.array([
			[0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 9, 9,10,10,10,11,12],		# source nodes
			[1, 2, 3, 3, 4, 3, 5, 6, 4, 6, 7, 7, 8, 6, 9, 7, 9,10, 8,10,11,11,10,12,11,12,13,13,13],		# destination nodes
			[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])		# bandwidth

		# whether all links get same bandwidth
		if SAME_BANDWIDTH:
			self.links_bandwidth[2] = np.ones(self.n_links)*bandwidth
		else:
			for i in range(self.n_links):
				self.links_bandwidth[2][i] = random.uniform(-20,20)+bandwidth

class TEST():
	def __init__(self):
		
		bandwidth = 100
		
		# number of nodes in the network
		self.n_nodes = 6
		# number of hosts in the network
		self.n_hosts = 6
		# number of links in the network
		self.n_links = 9
		# bandwidth of all links
		self.links_bandwidth = np.array([
			[0, 0, 0, 1, 1, 2, 2, 3, 4],
			[1, 2, 4 ,2, 3, 3, 4, 5, 5],
			[1, 1, 1, 1, 1, 1, 1, 1, 1]])

		# whether all links get same bandwidth
		if SAME_BANDWIDTH:
			self.links_bandwidth[2] = np.ones(self.n_links)*bandwidth
		else:
			for i in range(self.n_links):
				self.links_bandwidth[2][i] = random.uniform(-20,20)+bandwidth

class S_FATTREE():
	def __init__(self):
		
		bandwidth = 1
		
		# number of nodes in the network
		self.n_nodes = 12
		# number of hosts in the network
		self.n_hosts = 8
		# number of links in the network
		self.n_links = 16
		# bandwidth of all links
		self.links_bandwidth = np.array([
			[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
			[4, 5, 4 ,5, 6, 7, 6, 7, 8, 9,10,11, 8, 9,10,11],
			[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
		self.index_routing_links = [[[1,3],[1,3],[2,4],[2,4]],[[1,3],[1,3],[2,4],[2,4]],												#A<->C
			 [[1,3],[1,3],[2,4],[2,4]],[[1,3],[1,3],[2,4],[2,4]],												#A<->D
			 [[1,3],[1,3],[2,4],[2,4]],[[1,3],[1,3],[2,4],[2,4]],												#B<->C
			 [[1,3],[1,3],[2,4],[2,4]],[[1,3],[1,3],[2,4],[2,4]],												#B<->D
			 [[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]],[[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]], #A<->E
			 [[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]],[[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]],	#A<->F
			 [[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]],[[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]],	#B<->E
			 [[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]],[[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]],	#B<->F
			 [[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],[[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],	#A<->G
			 [[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],[[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],	#A<->H
			 [[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],[[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],	#B<->G
			 [[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],[[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],	#B<->H
			 [[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],[[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],	#C<->G
			 [[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],[[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],	#C<->H
			 [[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],[[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],	#D<->G
			 [[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],[[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],	#D<->H
			 [[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],[[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],	#C<->E
			 [[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],[[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],	#C<->F
			 [[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],[[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],	#D<->E
			 [[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],[[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],	#D<->F
			 [[5,7],[5,7],[6,8],[6,8]],[[5,7],[5,7],[6,8],[6,8]],												#E<->G
			 [[5,7],[5,7],[6,8],[6,8]],[[5,7],[5,7],[6,8],[6,8]],												#E<->H
			 [[5,7],[5,7],[6,8],[6,8]],[[5,7],[5,7],[6,8],[6,8]],												#F<->G
			 [[5,7],[5,7],[6,8],[6,8]],[[5,7],[5,7],[6,8],[6,8]]]												#F<->H
		for i in range(len(self.index_routing_links)):
			for j in range(len(self.index_routing_links[i])):
				for k in range(len(self.index_routing_links[i][j])):
					self.index_routing_links[i][j][k] -= 1

		# whether all links get same bandwidth
		if SAME_BANDWIDTH:
			self.links_bandwidth[2] = np.ones(self.n_links)*bandwidth
		else:
			for i in range(self.n_links):
				self.links_bandwidth[2][i] = random.uniform(-25,25)+bandwidth

class B_FATTREE():
	def __init__(self):
		
		bandwidth = 1
		
		# number of nodes in the network
		self.n_nodes = 20
		# number of hosts in the network
		self.n_hosts = 16
		# number of links in the network
		self.n_links = 32
		# bandwidth of all links
		self.links_bandwidth = np.array([
			[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9 ,9,10,10,11,11,12,12,13,13,14,14,15,15],
			[8, 9, 8 ,9,10,11,10,11,12,13,12,13,14,15,14,15,16,17,18,19,16,17,18,19,16,17,18,19,16,17,18,19],
			[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
		self.index_routing_links = [[[1,3],[1,3],[2,4],[2,4]],[[1,3],[1,3],[2,4],[2,4]],												#A<->C
			 [[1,3],[1,3],[2,4],[2,4]],[[1,3],[1,3],[2,4],[2,4]],												#A<->D
			 [[1,3],[1,3],[2,4],[2,4]],[[1,3],[1,3],[2,4],[2,4]],												#B<->C
			 [[1,3],[1,3],[2,4],[2,4]],[[1,3],[1,3],[2,4],[2,4]],												#B<->D
			 [[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]],[[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]], #A<->E
			 [[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]],[[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]],	#A<->F
			 [[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]],[[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]],	#B<->E
			 [[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]],[[1,5,9,10],[1,5,11,12],[2,6,13,14],[2,6,15,16]],	#B<->F
			 [[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],[[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],	#A<->G
			 [[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],[[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],	#A<->H
			 [[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],[[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],	#B<->G
			 [[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],[[1,7,9,10],[1,7,11,12],[2,8,13,14],[2,8,15,16]],	#B<->H
			 [[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],[[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],	#C<->G
			 [[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],[[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],	#C<->H
			 [[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],[[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],	#D<->G
			 [[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],[[3,7,9,10],[3,7,11,12],[4,8,13,14],[4,8,15,16]],	#D<->H
			 [[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],[[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],	#C<->E
			 [[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],[[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],	#C<->F
			 [[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],[[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],	#D<->E
			 [[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],[[3,5,9,10],[3,5,11,12],[4,6,13,14],[4,6,15,16]],	#D<->F
			 [[5,7],[5,7],[6,8],[6,8]],[[5,7],[5,7],[6,8],[6,8]],												#E<->G
			 [[5,7],[5,7],[6,8],[6,8]],[[5,7],[5,7],[6,8],[6,8]],												#E<->H
			 [[5,7],[5,7],[6,8],[6,8]],[[5,7],[5,7],[6,8],[6,8]],												#F<->G
			 [[5,7],[5,7],[6,8],[6,8]],[[5,7],[5,7],[6,8],[6,8]]]												#F<->H
		for i in range(len(self.index_routing_links)):
			for j in range(len(self.index_routing_links[i])):
				for k in range(len(self.index_routing_links[i][j])):
					self.index_routing_links[i][j][k] -= 1

		# whether all links get same bandwidth
		if SAME_BANDWIDTH:
			self.links_bandwidth[2] = np.ones(self.n_links)*bandwidth
		else:
			for i in range(self.n_links):
				self.links_bandwidth[2][i] = random.uniform(-25,25)+bandwidth