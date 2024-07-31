import numpy as np


class Agent:
    def __init__(self, i, j, time, price, Si, Ci):
        self.i = i  # Agent's row index in the grid
        self.j = j  # Agent's column index in the grid
        self.time = time  # Time of the agent's bid
        self.price = price  # Price of the agent's bid
        self.active = True  # Whether the agent will place an order in the next interval
        self.Si = Si
        self.Ci = Ci