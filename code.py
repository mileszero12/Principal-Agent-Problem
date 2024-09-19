import math
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np

N = 500  # num of segments
delta = 0.01  # the length of each segment
l = 0.14  # the cost for rejection
sigma_sq = 2
A = 10
B = 0.1
k = 1  # holding cost
alpha = 0.01
rho = 1
p_ub = 20
mu_ub = 10


class Y:
    def __init__(self, p, mu):
        self.y = {}
        self.last_p = p
        self.last_mu = mu
        self.last_lambda = {}

    def prepare(self):
        for i in range(1, N - 1):
            self.last_lambda[i] = A * math.exp(-B * self.last_p[i])

    def get_y(self):
        return self.y

    def get_last_p(self):
        return self.last_p

    def get_last_mu(self):
        return self.last_mu

    def get_last_lambda(self):
        return self.last_lambda

    def solve_y(self, p, mu):
        self.prepare()
        model = gp.Model("OPD")
        model.Params.OutputFlag = 0
        y_temp = model.addVars(N, vtype=gp.GRB.CONTINUOUS)

        # boundary conditions
        model.addConstr(y_temp[1] == y_temp[0], name='boundary_0')
        model.addConstr(y_temp[N - 1] - y_temp[N - 2] == - delta * l, name='boundary_N')

        # constraints
        for i in range(1, N - 1):
            model.addConstr(0.5 * sigma_sq * (y_temp[i + 1] - 2 * y_temp[i] + y_temp[i - 1]) / (delta * delta) + (
                        self.last_lambda[i] - self.last_mu[i]) * (y_temp[i + 1] - y_temp[i - 1]) / (2 * delta) +
                            self.last_lambda[i] * self.last_p[i] - k * delta * i - self.last_mu[i] * self.last_mu[
                                i] * alpha / 2 == rho * y_temp[i], name='y_' + str(i))

        model.optimize()
        if model.status == gp.GRB.OPTIMAL:
            for i in range(N):
                self.y[i] = y_temp[i].x
        else:
            print("Error")
            # exit(0)

    def get_next_p_mu(self):
        temp_p = {}
        temp_mu = {}
        for i in range(1, N - 1):
            C = (self.y[i + 1] - self.y[i - 1]) / (2 * delta)
            best_p = (1 - B * C) / B
            if best_p > p_ub:
                temp_p[i] = p_ub
            else:
                temp_p[i] = best_p

            best_mu = -C / alpha
            if best_mu < mu_ub:
                temp_mu[i] = mu_ub
            else:
                temp_mu[i] = best_mu

        return temp_p, temp_mu

    def plot_y(self):
        xpoints = []
        ypoints = []

        for i in range(0, N):
            xpoints.append(delta * i)
            ypoints.append(self.y[i])

        plt.plot(xpoints, ypoints)
        plt.show()

    def plot_z(self, y_col):
        xpoints = []
        ypoints = []

        for i in range(1, N - 1):
            xpoints.append(delta * i)
            ypoints.append(y_col[i])

        plt.plot(xpoints, ypoints)
        plt.show()

first_p, first_mu = {}, {}
for i in range(1, N - 1):
    first_p[i] = 5
    first_mu[i] = 3
firstY = Y(first_p, first_mu)
firstY.solve_y(first_p, first_mu)
nextp, nextmu = firstY.get_next_p_mu()
firstY.plot_z(nextp)


