import random
import numpy as np
from agent import Agent


class Market:
    def __init__(self, N, J=1, alpha=0.1, beta=1.45, a=0.1, b=0.1, p_star=100, mu_init=100):
        self.N = N  # Size of one side of the square grid
        self.J = J  # Interaction strength
        self.alpha = alpha  # Coupling coefficient
        self.beta = beta
        self.a = a  # Response strength of fundamentalists
        self.b = b  # Response strength of noise traders
        self.p_star = p_star  # Assumed fair price
        self.grid = [[None for _ in range(N)] for _ in range(N)]  # Initialize the grid with None
        self.previous_price = p_star  # Initialize with a base price
        self.mu_init = mu_init


    def init_agents(self):
        for i in range(self.N):
            for j in range(self.N):
                initial_time = int(np.ceil(np.random.exponential(scale=2) / 0.125))
                
                # Determine if the agent is a buyer or seller
                is_buyer = np.random.choice([True, False])
                is_noise = np.random.choice([True, False])
                
                if is_buyer:
                    mu = self.mu_init * 0.995
                    S_i = 1
                else:
                    mu = self.mu_init * 1.005
                    S_i = -1

                C_i = -1 if is_noise else 1
                sigma = self.mu_init * 0.01
                    
                initial_price = np.random.normal(mu, sigma)
                self.grid[i][j] = Agent(i, j, initial_time, initial_price, S_i, C_i)


    def populate_agents(self):
        for i in range(self.N):
            for j in range(self.N):
                if self.grid[i][j].active:
                    self.grid[i][j].time = int(np.ceil(np.random.exponential(scale=2) / 0.125))
                else:
                    self.grid[i][j].time = 0
                
                # Determine if the agent is a buyer or seller
                is_buyer = self.grid[i][j].price > 0
                is_noise = self.grid[i][j].Ci < 0
                
                if is_buyer:
                    mu = self.mu_init * 0.995
                    self.grid[i][j].S_i = 1
                else:
                    mu = self.mu_init * 1.005
                    self.grid[i][j].S_i = -1

                self.grid[i][j].C_i = -1 if is_noise else 1
                sigma = self.mu_init * 0.01
                    
                price = np.random.normal(mu, sigma)
                self.grid[i][j].price = price if is_buyer else -price


    def update_agents(self, t):
        for i in range(self.N):
            for j in range(self.N):
                if self.grid[i][j] and self.grid[i][j].active:
                        
                    self.calculate_magnetization()
                    self.update_price(i, j)
                    
                    # Generate the waiting time from an exponential distribution
                    waiting_time = np.random.exponential(scale=2)
                    
                    # Convert the waiting time to the corresponding trading interval
                    next_interval = t + int(np.ceil(waiting_time) / 0.125)
                    
                    # Ensure the next interval does not exceed the total intervals
                    self.grid[i][j].time = next_interval

                    self.grid[i][j].active = False


    def update_market(self, t):
        # Sort buyers and sellers based on price priority
        buyers = [(self.grid[i][j].price, i, j) for i in range(self.N) for j in range(self.N) \
                  if self.grid[i][j].price >= 0 and self.grid[i][j].time <= t and not self.grid[i][j].active]
        sellers = [(self.grid[i][j].price, i, j) for i in range(self.N) for j in range(self.N) \
                   if self.grid[i][j].price < 0 and self.grid[i][j].time <= t and not self.grid[i][j].active]
        
        buyers.sort(key=lambda x: x[0], reverse=True)
        sellers.sort(key=lambda x: abs(x[0]))

        transaction_price = None
        
        while buyers and sellers:
            buyer_price, buyer_i, buyer_j = buyers[0]
            seller_price, seller_i, seller_j = sellers[0]
            
            if buyer_price >= abs(seller_price):
                transaction_price = (buyer_price + abs(seller_price)) / 2
                buyers.pop(0)
                sellers.pop(0)
                self.grid[buyer_i][buyer_j].active = True
                self.grid[seller_i][seller_j].active = True
            else:
                break

        if not transaction_price is None:
            self.p_star = transaction_price


    def is_valid_pos(self, i, j):
        return 0 <= i < self.N and 0 <= j < self.N
    
    
    def calculate_magnetization(self):
        # Calculate the market magnetization M(t)
        self.M = sum(agent.Si for row in self.grid for agent in row if agent) / self.N**2

    
    def calculate_hamiltonian(self, i, j):
        h = 0
        # Sum the influences of neighboring agents for the Hamiltonian
        for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:  # Neighbors
            ni, nj = (i + di), (j + dj)
            if self.is_valid_pos(ni, nj):
                h += self.grid[ni][nj].Si * self.J

        h -= self.alpha * self.grid[i][j].Ci * self.M
        return h
    

    def update_Si(self, i, j):
        h_i = self.calculate_hamiltonian(i, j)
        q = 1 / (1 + np.exp(-2 * h_i * self.beta))
        random_number = random.random()
        self.grid[i][j].Si = 1 if random_number <= q else -1


    def update_Ci(self, i, j):
        if self.grid[i][j].Si * self.grid[i][j].Ci * self.M < 0:
            self.grid[i][j].Ci = -self.grid[i][j].Ci


    def update_price(self, i, j):

        Delta_1 = 1 if self.grid[i][j].price >= self.p_star and self.grid[i][j].price >= 0 else 0
        Delta_2 = 1 if self.grid[i][j].price < self.p_star and self.grid[i][j].price >= 0 else 0
        Delta_3 = 1 if self.grid[i][j].price >= self.p_star and self.grid[i][j].price < 0 else 0
        Delta_4 = 1 if self.grid[i][j].price < self.p_star and self.grid[i][j].price < 0 else 0
        Delta = np.array([Delta_1, Delta_2, Delta_3, Delta_4])
        h_n = Delta * 4 * self.J - self.alpha * abs(self.M) * (self.M * (self.N ** 2))
        probabilities = np.exp(-self.beta * h_n)
        probabilities /= probabilities.sum()  # Normalize probabilities
        probabilities = np.cumsum(probabilities)

        # Implementing the probability distribution for price updates
        deltas = np.random.exponential(scale=1, size=4)

        random_number = random.random()
        if random_number <= probabilities[0]:
            self.grid[i][j].price = min(self.p_star + deltas[0], self.previous_price * 1.1)
        elif random_number <= probabilities[1]:
            self.grid[i][j].price = min(self.p_star - deltas[1], self.previous_price * 1.1)
        elif random_number <= probabilities[2]:
            self.grid[i][j].price = -max(self.p_star - deltas[2], self.previous_price * 0.9)
        else:
            self.grid[i][j].price = -max(self.p_star + deltas[3], self.previous_price * 0.9)