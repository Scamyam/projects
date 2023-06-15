import numpy as np
import matplotlib.pypot as plt
import random 
import math 
class neural_net:
    def __init__(self,hidden_num_layers,act_func):
        self.hidden_num_layers=hidden_num_layers 
        self.act_func=act_func
        self.weight_in=np.array([[random.random() for i in range(784)]for i in range(10)])
        self.weights_hidden=np.array([[random.random() for i in range(10)]for i in range(10)])
        self.weights_out=np.array([[random.random()]for i in range(10)])
        self.biases=np.array([[random.random()for i in range(10)] for i in range(3)])
    def forward(self,X):
        out=np.array([0 for i  in range(10)])
        for x in range(10):
            s=0
            for y in range(784):
                for z in range(10):
                    s+=X[x][y]*self.weights_hidden[z]
            out[x]= sum
    def matrix_multiply(matrix1,matrix2,dim_a,dim_c):
        out=[[0]* dim_a]*dim_c
        for i in range(dim_a):
            for j in range(dim_c):
                out[i][j] += mat_1[i][j]*mat[i]
            return (out[1])
            
        


        