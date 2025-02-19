#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:20:31 2025

@author: epge903150
"""

# Importing libraries
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Used to plot distributions for eMG
from scipy.stats import multivariate_normal
from itertools import combinations

# Importing libraries
# Binomial cumulative distribution function
from scipy.stats import binom
# Inverse chi square
from scipy.stats.distributions import chi2

class base():
    
    def __init__(self):
        
        # Shared parameters
        self.parameters_list = []
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.y_pred_training = None
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = None
        # Computing the output in the testing phase
        self.y_pred_test = None
    
    def n_rules(self):
        return self.rules[-1]
    
    def output_training(self):
        return self.y_pred_training

class ePL_KRLS_DISCO(base):
    
    def __init__(self, alpha = 0.001, beta = 0.05, lambda1 = 0.0000001, sigma = 0.5, tau = 0.05, omega = 1, e_utility = 0.05):
        
        # Call __init__ of the base class
        super().__init__()

        
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be a float between 0 and 1.")
        if not (0 <= beta <= 1):
            raise ValueError("beta must be a float between 0 and 1.")
        if not (0 <= lambda1 <= 1):
            raise ValueError("lambda1 must be a float between 0 and 1.")
        if not (0 <= e_utility <= 1):
            raise ValueError("e_utility must be a float between 0 and 1.")
        if not (0 <= tau <= 1):  # tau can be NaN or in [0, 1]
            raise ValueError("tau must be a float between 0 and 1, or NaN.")
        if not (sigma > 0):
            raise ValueError("sigma must be a positive float.")
        if not (isinstance(omega, int) and omega > 0):
            raise ValueError("omega must be a positive integer.")
            
        # Hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.sigma = sigma
        self.tau = tau
        self.omega = omega
        self.e_utility = e_utility
        
        # Parameters used to calculate the utility measure
        self.epsilon = []
        self.eTil = [0.]
        # Monitoring if some rule was excluded
        self.ExcludedRule = 0
    
    def get_params(self, deep=True):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'lambda1': self.lambda1,
            'sigma': self.sigma,
            'tau': self.tau,
            'omega': self.omega,
            'e_utility': self.e_utility,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def is_numeric_and_finite(self, array):
        return np.isfinite(array).all() and np.issubdtype(np.array(array).dtype, np.number)
    
    def show_rules(self):
        rules = []
        for i in self.parameters.index:
            rule = f"Rule {i}"
            for j in range(self.parameters.loc[i,"Center"].shape[0]):
                rule = f'{rule} - {self.parameters.loc[i,"Center"][j].item():.2f} ({self.sigma:.2f})'
            print(rule)
            rules.append(rule)
        
        return rules
    
    def plot_rules(self, num_points=100):
        # Set plot-wide configurations only once
        plt.rc('font', size=13)
        plt.rc('axes', titlesize=15)
        # plt.figure(figsize=(19.20, 10.80))
    
        # Determine the number of rules (rows) and attributes per rule
        num_rules = len(self.parameters.index)
        num_attributes = self.parameters.loc[self.parameters.index[0], "Center"].shape[0]
    
        # Create a figure with subplots (one per rule)
        fig, axes = plt.subplots(num_rules, 1, figsize=(8, num_rules * 4), squeeze=False, sharey=True)
    
        # Iterate through rules
        for i, rule_idx in enumerate(self.parameters.index):
            ax = axes[i, 0]  # Select the subplot for the rule
            
            # Iterate through all attributes and plot them in the same subplot
            for j in range(num_attributes):
                center = self.parameters.loc[rule_idx, "Center"][j]
                x_vals = np.linspace(center - 3 * self.sigma, center + 3 * self.sigma, num_points)
                y_vals = np.exp(-((x_vals - center) ** 2) / (2 * self.sigma ** 2))
                
                ax.plot(x_vals, y_vals, linewidth=3, label=f'Attr {j}: μ={center.item():.2f}, σ={self.sigma:.2f}')
            
            # Subplot settings
            ax.set_title(f'Rule {rule_idx}')
            ax.legend(loc="lower center", ncol=2)
            ax.grid(False)
    
        # Set a single y-label for the entire figure
        fig.supylabel("Membership")
        fig.supxlabel("Values")
    
        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()
    
    def plot_gaussians(self, num_points=100):
        # Set plot-wide configurations only once
        plt.rc('font', size=30)
        plt.rc('axes', titlesize=30)
        
        # Iterate through rules and attributes
        for i in self.parameters.index:
            for j in range(self.parameters.loc[i,"Center"].shape[0]):
                
                # Generate x values for smooth curve
                x_vals = np.linspace(self.parameters.loc[i,"Center"][j] - 3*self.sigma, self.parameters.loc[i,"Center"][j] + 3*self.sigma, num_points)
                y_vals = np.exp(-((x_vals - self.parameters.loc[i,"Center"][j])**2) / (2 * self.sigma**2))
                
                # Create and configure the plot
                plt.figure(figsize=(19.20, 10.80))
                plt.plot(x_vals, y_vals, color='blue', linewidth=3, label=f'Gaussian (μ={self.parameters.loc[i,"Center"][j].item():.2f}, σ={self.sigma:.2f})')
                plt.title(f'Rule {i} - Attribute {j}')
                plt.xlabel('Values')
                plt.ylabel('Membership')
                plt.legend()
                plt.grid(False)
                
                # Display the plot
                plt.show()
         
    def fit(self, X, y):
        
        # Shape of X and y
        X_shape = X.shape
        y_shape = y.shape
        
        # Correct format X to 2d
        if len(X_shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y_shape) > 1 and y_shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y_shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X_shape[0] != y_shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
            
        # Preallocate space for the outputs for better performance
        self.y_pred_training = np.zeros((y_shape))
        self.ResidualTrainingPhase = np.zeros((y_shape))
        
        # Initialize outputs
        self.y_pred_training[0,] = y[0]
        self.ResidualTrainingPhase[0,] = 0.
        
        # Prepare the first input vector
        x = X[0,].reshape((1,-1)).T
        # Initialize the first rule
        self.NewRule(x, y[0], 1., None, True)
        
        for k in range(1, X.shape[0]):
            
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            
            # Compute the compatibility measure and the arousal index for all rules
            MinArousal, MaxCompatibility, MaxCompatibilityIdx = (np.inf, 0, 0)
            for i in range(len(self.parameters_list)):
                
                # Update the compatibility measure and the respective arousal index (inside compatibility measure function)
                self.CompatibilityMeasure(x, i)
                
                # Find the minimum arousal index
                if self.parameters_list[i][6] < MinArousal:
                    MinArousal = self.parameters_list[i][6]
            
                # Find the maximum compatibility measure
                if self.parameters_list[i][11] > MaxCompatibility:
                    MaxCompatibility = self.parameters_list[i][11]
                    MaxCompatibilityIdx = i
            
            # Verify the needing to creating a new rule
            if MinArousal > self.tau and self.ExcludedRule == 0 and self.epsilon != []:
                # Create a new rule
                self.NewRule(x, y[k], k+1, MaxCompatibilityIdx, False)
                # Save the position of the created rule
                MaxCompatibility = 1.
                MaxCompatibilityIdx = len(self.parameters_list) - 1
                
            else:
                # Update the most compatible rule
                self.UpdateRule(x, y[k], MaxCompatibilityIdx)
                # Update the consequent parameters
                self.KRLS(x, y[k], MaxCompatibilityIdx)
            
            # Update lambda values
            self.Lambda(x)
            
            # Check wheather it is necessary to remove a rule
            if len(self.parameters_list) > 1:
                self.UtilityMeasure(X[k,], k+1)
                
            # Update the number of rules at the current iteration
            self.rules.append(len(self.parameters_list))
            
            # Computing the output
            Output = self.parameters_list[MaxCompatibilityIdx][5].T @ self.GaussianKernel(self.parameters_list[MaxCompatibilityIdx][1], x)
            
            # Store the results
            self.y_pred_training[k,] = Output.item()
            residual = abs(Output - y[k])
            self.ResidualTrainingPhase[k,] = residual ** 2
            
            # Update epsilon and e_til
            quociente = math.exp(-0.8 * self.eTil[-1] - residual)
            if quociente == 0:
                self.epsilon.append(max(self.epsilon))
            else:
                epsilon = ( math.exp(-0.5) * (2/(math.exp(-0.8 * self.eTil[-1] - abs(Output - y[k]))) - 1) )
                if epsilon >= 1. and len(self.epsilon) != 0:
                    self.epsilon.append(max(self.epsilon))
                elif epsilon >= 1.:
                    self.epsilon.append(0.8)
                else:
                    epsilon = ( math.exp(-0.5) * (2/(math.exp(-0.8 * self.eTil[-1] - abs(Output - y[k]))) - 1) )
                    self.epsilon.append(epsilon)
            
            self.eTil.append(0.8 * self.eTil[-1] + abs(Output - y[k]))
            
        # Save the rules to a dataframe
        self.parameters = pd.DataFrame(self.parameters_list, columns=['Center', 'dictionary', 'nu', 'P', 'Q', 'Theta','arousal_index', 'utility', 'sum_lambda', 'time_creation', 'CompatibilityMeasure', 'old_Center', 'tau', 'lambda', 'lambda'])
    
    def evolve(self, X, y):
        
        # Be sure that X is with a correct shape
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center'].shape[0])
        
        # Shape of X and y
        X_shape = X.shape
        y_shape = y.shape
        
        # Check the format of y
        if not isinstance(y, (np.ndarray)):
            y = np.array(y, ndmin=1)
            
        # Correct format X to 2d
        if len(X_shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y_shape) > 1 and y_shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y_shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X_shape[0] != y_shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
        
        for k in range(X.shape[0]):
            
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            
            # Compute the compatibility measure and the arousal index for all rules
            MinArousal, MaxCompatibility, MaxCompatibilityIdx = (np.inf, 0, 0)
            for i in range(len(self.parameters_list)):
                
                # Update the compatibility measure and the respective arousal index (inside compatibility measure function)
                self.CompatibilityMeasure(x, i)
                
                # Find the minimum arousal index
                if self.parameters_list[i][6] < MinArousal:
                    MinArousal = self.parameters_list[i][6]
            
                # Find the maximum compatibility measure
                if self.parameters_list[i][11] > MaxCompatibility:
                    MaxCompatibility = self.parameters_list[i][11]
                    MaxCompatibilityIdx = i
            
            # Verify the needing to creating a new rule
            if MinArousal > self.tau and self.ExcludedRule == 0 and self.epsilon != []:
                # Create a new rule
                self.NewRule(x, y[k], k+1, MaxCompatibilityIdx, False)
                # Save the position of the created rule
                MaxCompatibility = 1.
                MaxCompatibilityIdx = len(self.parameters_list) - 1
                
            else:
                # Update the most compatible rule
                self.UpdateRule(x, y[k], MaxCompatibilityIdx)
                # Update the consequent parameters
                self.KRLS(x, y[k], MaxCompatibilityIdx)
            
            # Update lambda values
            self.Lambda(x)
            
            # Check wheather it is necessary to remove a rule
            if len(self.parameters_list) > 1:
                self.UtilityMeasure(X[k,], k+1)
                
            # Update the number of rules at the current iteration
            self.rules.append(len(self.parameters_list))
            
            # Computing the output
            Output = self.parameters_list[MaxCompatibilityIdx][5].T @ self.GaussianKernel(self.parameters_list[MaxCompatibilityIdx][1], x)
            
            # Store the prediction
            self.y_pred_training = np.append(self.y_pred_training, Output)
            # Compute the error
            residual = abs(Output - y[k])
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, residual**2)
            
            # Update epsilon and e_til
            quociente = math.exp(-0.8 * self.eTil[-1] - residual)
            if quociente == 0:
                self.epsilon.append(max(self.epsilon))
            else:
                epsilon = ( math.exp(-0.5) * (2/(math.exp(-0.8 * self.eTil[-1] - abs(Output - y[k]))) - 1) )
                if epsilon >= 1. and len(self.epsilon) != 0:
                    self.epsilon.append(max(self.epsilon))
                elif epsilon >= 1.:
                    self.epsilon.append(0.8)
                else:
                    epsilon = ( math.exp(-0.5) * (2/(math.exp(-0.8 * self.eTil[-1] - abs(Output - y[k]))) - 1) )
                    self.epsilon.append(epsilon)
            
            self.eTil.append(0.8 * self.eTil[-1] + abs(Output - y[k]))
            
        # Save the rules to a dataframe
        self.parameters = pd.DataFrame(self.parameters_list, columns=['Center', 'dictionary', 'nu', 'P', 'Q', 'Theta','arousal_index', 'utility', 'sum_lambda', 'time_creation', 'CompatibilityMeasure', 'old_Center', 'tau', 'lambda', 'lambda'])
            
    def predict(self, X):
                
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
        
        # Be sure that X is with a correct shape
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center'].shape[0])
        
        # Preallocate space for the outputs for better performance
        self.y_pred_test = np.zeros((X.shape[0]))
            
        for k in range(X.shape[0]):
            
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            
            # Compute the compatibility measure and the arousal index for all rules
            MaxCompatibility, MaxCompatibilityIdx = (0, 0)
            for i in range(len(self.parameters_list)):
                
                # Update the compatibility measure and the respective arousal index (inside compatibility measure function)
                self.CompatibilityMeasure(x, i)
                            
                # Find the maximum compatibility measure
                if self.parameters_list[i][11] > MaxCompatibility:
                    MaxCompatibility = self.parameters_list[i][11]
                    MaxCompatibilityIdx = i
            
            # Computing the output
            Output = self.parameters_list[MaxCompatibilityIdx][5].T @ self.GaussianKernel(self.parameters_list[MaxCompatibilityIdx][1], x)
            
            # Store the results
            self.y_pred_test[k,] = Output.item()
        
        return self.y_pred_test
            
    def NewRule(self, x, y, k=1., i=None, isFirst = False):
        
        if isFirst:
            
            kernel_value = self.GaussianKernel(x, x)
            Q = np.linalg.inv(np.ones((1,1)) * (self.lambda1 + kernel_value))
            Theta = Q*y
            self.parameters_list.append([x, x, float(self.sigma), np.ones((1,1)), Q, Theta, 0., 1., 0., 1., k, 1., np.zeros((x.shape[0],1)), 1., 0.])
        
        else:
            
            kernel_value = self.GaussianKernel(x, x)
            Q = np.linalg.inv(np.ones((1,1)) * (self.lambda1 + kernel_value))
            Theta = Q*y
            # Compute nu
            distance = np.linalg.norm(x - self.parameters_list[i][0])
            log_epsilon = math.sqrt(-2 * np.log(max(self.epsilon)))
            nu = float(distance / log_epsilon)
            self.parameters_list.append([x, x, nu, np.ones((1,1)), Q, Theta, 0., 1., 0., 1., k, 1., np.zeros((x.shape[0],1)), 1., 0.])
    
    def CompatibilityMeasure(self, x, i):
        
        # The compatibility measure can be lower than 0 if the input data is not normalized
        # Verify if it is possible to compute the correlation
        if (not np.all(np.isfinite(x)) or np.std(x, axis=0).min() == 0) or (not np.all(np.isfinite(self.parameters_list[i][0])) or np.std(self.parameters_list[i][0], axis=0).min() == 0):
            # Compute the norm used to calculate the compatibility measure
            norm_cm = (np.linalg.norm(x - self.parameters_list[i][0]) / x.shape[0])
            # Compute the compatibility measure without the correlation
            CompatibilityMeasure = 1 - norm_cm if norm_cm < 1. else 0.
        else:
            # Compute the correlation
            correlation = np.corrcoef(self.parameters_list[i][0].T, x.T)[0, 1]
            # Compute the norm used to calculate the compatibility measure
            norm_cm = (np.linalg.norm(x - self.parameters_list[i][0]) / x.shape[0])
            # Compute the compatibility measure
            CompatibilityMeasure = (1 - norm_cm) * ((correlation + 1) / 2) if norm_cm < 1. else 0.
        
        # Update the compatibility measure
        self.parameters_list[i][11] = CompatibilityMeasure
        
        # Update the respective arousal index
        self.Arousal_Index(i)
    
    def Arousal_Index(self, i):
        
        # Atualização para todas as regras no DataFrame
        self.parameters_list[i][6] += self.beta * (1 - self.parameters_list[i][11] - self.parameters_list[i][6])

    
    def UpdateRule(self, x, y, i):
        
        # Update the parameters
        # Update the number of observations
        self.parameters_list[i][9] += 1
        
        # Update the old and new Centers
        old_Center = self.parameters_list[i][0]
        if np.isnan(self.parameters_list[i][6]):
            print(self.parameters_list[i][6])
        compatibility_adjustment = self.alpha * (self.parameters_list[i][11]) ** (1 - self.parameters_list[i][6])
        new_Center = old_Center + compatibility_adjustment * (x - old_Center)
        
        self.parameters_list[i][12], self.parameters_list[i][0] = (old_Center, new_Center)
            
    def GaussianKernel(self, v1, v2):
        
        n = v1.shape[1]
        if n == 1:
            # Compute the kernel distance
            distance = np.linalg.norm(v1 - v2)**2
            return np.array([math.exp(-distance / (2 * self.sigma**2))]).reshape(-1,1)
        else:
            v3 = np.zeros((n,))
            for j in range(n):
                v3[j,] = math.exp(- (np.linalg.norm(v1[:,j].reshape(-1,1) - v2)**2) / (2 * self.sigma**2))
            return v3.reshape(-1,1)
    
    def GaussianMF(self, v1, v2):
        mf = np.zeros((v1.shape))
        for j in range(v1.shape[0]):
            denominator = (2 * self.sigma ** 2)
            if denominator != 0:
                mf[j,0] = math.exp( - ( v1[j,0] - v2[j,0] ) ** 2 / denominator )
            else:
                mf[j,0] = math.exp( - ( v1[j,0] - v2[j,0] ) ** 2 / 2 )
        return mf.prod()
    
    def Tau(self, x):
        for row in range(len(self.parameters_list)):
            tau = self.GaussianMF(x, self.parameters_list[row][0])
            # Evoid tau with values zero
            if abs(tau) < (10 ** -100):
                tau = (10 ** -100)
            self.parameters_list[row][13] = tau
    
    def Lambda(self, x):
        
        # Update the values of Tau
        self.Tau(x)
        
        # Compute the sum of tau
        tau_sum = 0
        for i in range(len(self.parameters_list)):
            tau_sum += self.parameters_list[i][13]
        
        for i in range(len(self.parameters_list)):
            
            # Update lambda
            self.parameters_list[i][14] = self.parameters_list[i][13] / tau_sum
        
            # Update the sum of lambda
            self.parameters_list[i][8] += self.parameters_list[i][14]
            
    def UtilityMeasure(self, x, k):
        
        remove = []
        for i in range(len(self.parameters_list)):
            
            # Compute how long ago the rule was created
            time_diff = k - self.parameters_list[i][10]
            
            # Compute the utilit measure
            self.parameters_list[i][7] = self.parameters_list[i][8] / time_diff if time_diff != 0 else 1
            
        
            # Find rules with utility lower than a threshold
            if self.parameters_list[i][7] < self.e_utility:
                remove.append(i)
                    
        # Remove old rules
        if not remove:
            
            self.parameters_list = [item for i, item in enumerate(self.parameters_list) if i not in remove]
    
            # Inform that a rules was excluded and create no more rules
            self.ExcludedRule = 1
            
            
    def KRLS(self, x, y, i):
        
        # Verify the number of observations
        num_obs = self.parameters_list[i][9]
        if num_obs <= 0:
            raise ValueError("Número de observações deve ser maior que zero para evitar divisão por zero.")
    
        # Atualizar 'nu' (kernel size)
        Center = self.parameters_list[i][0]
        old_Center = self.parameters_list[i][12]
        nu = self.parameters_list[i][2]
        norm_diff = np.linalg.norm(x - Center)
        Center_shift = np.linalg.norm(Center - old_Center)
    
        self.parameters_list[i][2] = math.sqrt(
            nu**2 + (norm_diff**2 - nu**2) / num_obs + (num_obs - 1) * Center_shift**2 / num_obs
        )
    
        # Calcular vetor g
        dictionary = self.parameters_list[i][1]
        g = np.array([self.GaussianKernel(dictionary, x)]).reshape(-1,1)
    
        # Calcular z, r, erro estimado
        z = np.matmul(self.parameters_list[i][4], g)
        r = self.lambda1 + 1 - np.matmul(z.T, g).item()
        estimated_error = y - np.matmul(g.T, self.parameters_list[i][5])
    
        # Calcular distâncias
        distance = np.linalg.norm(dictionary - x, axis=0)
        min_distance = np.min(distance)
    
        # Critério de novidade
        if min_distance > 0.1 * self.parameters_list[i][2]:
            # Atualizar dicionário
            self.parameters_list[i][1] = np.hstack([dictionary, x])
    
            # Atualizar Q
            Q = self.parameters_list[i][4]
            sizeQ = Q.shape[0]
            new_Q = np.zeros((sizeQ + 1, sizeQ + 1))
            new_Q[:sizeQ, :sizeQ] = Q + (1 / r) * np.matmul(z, z.T)
            new_Q[-1, -1] = (1 / r) * self.omega
            new_Q[:sizeQ, -1] = new_Q[-1, :sizeQ] = -(1 / r) * z.flatten()
            self.parameters_list[i][4] = new_Q
    
            # Atualizar P
            P = self.parameters_list[i][3]
            new_P = np.zeros((P.shape[0] + 1, P.shape[1] + 1))
            new_P[:P.shape[0], :P.shape[1]] = P
            new_P[-1, -1] = self.omega
            self.parameters_list[i][3] = new_P
    
            # Atualizar Theta
            Theta = self.parameters_list[i][5]
            self.parameters_list[i][5] = np.vstack([Theta - (z * (1 / r) * estimated_error), (1 / r) * estimated_error])
        else:
            # Atualizar P e Theta (caso de baixa novidade)
            P = self.parameters_list[i][3]
            q = np.matmul(P, z) / (1 + np.matmul(np.matmul(z.T, P), z))
            self.parameters_list[i][3] = P - np.matmul(q, np.matmul(z.T, P))
            self.parameters_list[i][5] += np.matmul(self.parameters_list[i][4], q) * estimated_error
            
class ePL_plus(base):
    
    def __init__(self, alpha = 0.001, beta = 0.1, lambda1 = 0.35, tau = None, omega = 1000, sigma = 0.25, e_utility = 0.05, pi = 0.5):
        
        # Call __init__ of the base class
        super().__init__()
        
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be a float between 0 and 1.")
        if not (0 <= beta <= 1):
            raise ValueError("beta must be a float between 0 and 1.")
        if not (0 <= lambda1 <= 1):
            raise ValueError("lambda1 must be a float between 0 and 1.")
        if not (tau is None or (isinstance(tau, float) and (0 <= tau <= 1))):  # tau can be NaN or in [0, 1]
            raise ValueError("tau must be a float between 0 and 1, or None.")
        if not (isinstance(omega, int) and omega > 0):
            raise ValueError("omega must be a positive integer.")
        if not (0 <= e_utility <= 1):
            raise ValueError("e_utility must be a float between 0 and 1.")
        if not (0 <= sigma <= 1):
            raise ValueError("sigma must be a float between 0 and 1.")
        if not (0 <= pi <= 1):
            raise ValueError("pi must be a float between 0 and 1.")
            
        # Hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.tau = beta if tau is None else tau
        self.omega = omega
        self.sigma = sigma
        self.e_utility = e_utility
        self.pi = pi
        
        # Monitoring if some rule was excluded
        self.ExcludedRule = 0
    
    def get_params(self, deep=True):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'lambda1': self.lambda1,
            'tau': self.tau,
            'omega': self.omega,
            'sigma': self.sigma,
            'e_utility': self.e_utility,
            'pi': self.pi,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def is_numeric_and_finite(self, array):
        return np.isfinite(array).all() and np.issubdtype(np.array(array).dtype, np.number)
    
    def show_rules(self):
        rules = []
        for i in self.parameters.index:
            rule = f"Rule {i}"
            for j in range(self.parameters.loc[i,"Center"].shape[0]):
                rule = f'{rule} - {self.parameters.loc[i,"Center"][j].item():.2f} ({self.parameters.loc[i, "sigma"][j].item():.2f})'
            print(rule)
            rules.append(rule)
        
        return rules
    
    def plot_rules(self, num_points=100):
        
        # Warning for this function
        warnings.warn("ePL+ does not compute the standard deviation; it calculates the radius, which is different. As a result, the Gaussians may not be meaningful.", UserWarning)
        
        # Set plot-wide configurations only once
        plt.rc('font', size=13)
        plt.rc('axes', titlesize=15)
        # plt.figure(figsize=(19.20, 10.80))
    
        # Determine the number of rules (rows) and attributes per rule
        num_rules = len(self.parameters.index)
        num_attributes = self.parameters.loc[self.parameters.index[0], "Center"].shape[0]
    
        # Create a figure with subplots (one per rule)
        fig, axes = plt.subplots(num_rules, 1, figsize=(8, num_rules * 4), squeeze=False, sharey=True)
    
        # Iterate through rules
        for i, rule_idx in enumerate(self.parameters.index):
            ax = axes[i, 0]  # Select the subplot for the rule
            
            # Iterate through all attributes and plot them in the same subplot
            for j in range(num_attributes):
                center = self.parameters.loc[rule_idx, "Center"][j]
                sigma = self.parameters.loc[rule_idx, "sigma"][j]
                x_vals = np.linspace(center - 3 * sigma, center + 3 * sigma, num_points)
                y_vals = np.exp(-((x_vals - center) ** 2) / (2 * sigma ** 2))
                
                ax.plot(x_vals, y_vals, linewidth=3, label=f'Attr {j}: μ={center.item():.2f}, σ={sigma.item():.2f}')
            
            # Subplot settings
            ax.set_title(f'Rule {rule_idx}')
            ax.legend(loc="lower center", ncol=2)
            ax.grid(False)
    
        # Set a single y-label for the entire figure
        fig.supylabel("Membership")
        fig.supxlabel("Values")
    
        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()
    
    def plot_gaussians(self, num_points=100):
        
        # Warning for this function
        warnings.warn("ePL+ does not compute the standard deviation; it calculates the radius, which is different. As a result, the Gaussians may not be meaningful.", UserWarning)
        
        # Set plot-wide configurations only once
        plt.rc('font', size=30)
        plt.rc('axes', titlesize=30)
        
        # Iterate through rules and attributes
        for i in self.parameters.index:
            for j in range(self.parameters.loc[i,"Center"].shape[0]):
                
                # Generate x values for smooth curve
                x_vals = np.linspace(self.parameters.loc[i,"Center"][j] - 3*self.parameters.loc[i,"sigma"][j], self.parameters.loc[i,"Center"][j] + 3*self.parameters.loc[i,"sigma"][j], num_points)
                y_vals = np.exp(-((x_vals - self.parameters.loc[i,"Center"][j])**2) / (2 * self.parameters.loc[i,"sigma"][j]**2))
                
                # Create and configure the plot
                plt.figure(figsize=(19.20, 10.80))
                plt.plot(x_vals, y_vals, color='blue', linewidth=3, label=f'Gaussian (μ={self.parameters.loc[i,"Center"][j].item():.2f}, σ={self.parameters.loc[i,"sigma"][j].item()**(1/2):.2f})')
                plt.title(f'Rule {i} - Attribute {j}')
                plt.xlabel('Values')
                plt.ylabel('Membership')
                plt.legend()
                plt.grid(False)
                
                # Display the plot
                plt.show()
         
    def fit(self, X, y):
        
        # Shape of X and y
        X_shape = X.shape
        y_shape = y.shape
        
        # Correct format X to 2d
        if len(X_shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y_shape) > 1 and y_shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y_shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X_shape[0] != y_shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
            
        # Prepare the first input vector
        x = X[0,].reshape((1,-1)).T
        # Compute xe
        xe = np.insert(x.T, 0, 1, axis=1).T
        # Compute z
        z = np.concatenate((x.T, y[0].reshape(-1,1)), axis=1).T
        
        # Preallocate space for the outputs for better performance
        self.y_pred_training = np.zeros((y_shape))
        self.ResidualTrainingPhase = np.zeros((y_shape))
        
        # Initialize outputs
        self.y_pred_training[0,] = y[0]
        self.ResidualTrainingPhase[0,] = 0.
        
        # Initialize the first rule
        self.NewRule(x, y[0], z, True)
        
        # Compute the normalized firing level
        self.Lambda(x)
        
        # Update the consequent parameters of the fist rule
        self.RLS(x, y[0], xe)
        
        for k in range(1, X.shape[0]):
            
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Compute z
            z = np.concatenate((x.T, y[k].reshape(-1,1)), axis=1).T
            
            # Compute the compatibility measure and the arousal index for all rules
            MinArousal, MaxCompatibility, MaxCompatibilityIdx = (np.inf, 0, 0)
            for i in range(len(self.parameters_list)):
                
                # Update the compatibility measure and the respective arousal index (inside compatibility measure function)
                self.CompatibilityMeasure(x, i)
                
                # Find the minimum arousal index
                if self.parameters_list[i][3] < MinArousal:
                    MinArousal = self.parameters_list[i][3]
            
                # Find the maximum compatibility measure
                if self.parameters_list[i][4] > MaxCompatibility:
                    MaxCompatibility = self.parameters_list[i][4]
                    MaxCompatibilityIdx = i
            
            # Verify the needing to creating a new rule
            if MinArousal > self.tau:
                self.NewRule(x, y[k], z, k+1, False)
            else:
                self.UpdateRule(x, y[k], z, MaxCompatibilityIdx)
            
            # Check if it is possible to remove any rule
            if len(self.parameters_list) > 1:
                self.SimilarityIndex()
                
            # Compute the number of rules at the current iteration
            self.rules.append(len(self.parameters_list))
            
            # Compute the normalized firing level
            self.Lambda(x)
            
            # Update the consequent parameters of the fist rule
            self.RLS(x, y[k], xe)
            
            # Utility Measure
            if len(self.parameters_list) > 1:
                self.UtilityMeasure(X[k,], k+1)
                
            # Compute the output
            Output = sum([self.parameters_list[row][7] * xe.T @ self.parameters_list[row][2] for row in range(len(self.parameters_list))])
            
            # Store the results
            self.y_pred_training[k,] = Output.item()
            residual = abs(Output - y[k])
            self.ResidualTrainingPhase[k,] = residual ** 2
            
        # Save the rules to a dataframe
        self.parameters = pd.DataFrame(self.parameters_list, columns=['Center', 'P', 'Theta', 'ArousalIndex', 'CompatibilityMeasure', 'TimeCreation', 'NumObservations', 'lambda', 'SumLambda', 'Utility', 'sigma', 'support', 'z', 'diff_z', 'local_scatter', 'tau'])
    
    def evolve(self, X, y):
        
        # Be sure that X is with a correct shape
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center'].shape[0])
        
        # Check the format of y
        if not isinstance(y, (np.ndarray)):
            y = np.array(y, ndmin=1)
            
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
            
        for k in range(1, X.shape[0]):
            
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Compute z
            z = np.concatenate((x.T, y[k].reshape(-1,1)), axis=1).T
            
            # Compute the compatibility measure and the arousal index for all rules
            MinArousal, MaxCompatibility, MaxCompatibilityIdx = (np.inf, 0, 0)
            for i in range(len(self.parameters_list)):
                
                # Update the compatibility measure and the respective arousal index (inside compatibility measure function)
                self.CompatibilityMeasure(x, i)
                
                # Find the minimum arousal index
                if self.parameters_list[i][3] < MinArousal:
                    MinArousal = self.parameters_list[i][3]
            
                # Find the maximum compatibility measure
                if self.parameters_list[i][4] > MaxCompatibility:
                    MaxCompatibility = self.parameters_list[i][4]
                    MaxCompatibilityIdx = i
            
            # Verify the needing to creating a new rule
            if MinArousal > self.tau:
                self.NewRule(x, y[k], z, k+1, False)
            else:
                self.UpdateRule(x, y[k], z, MaxCompatibilityIdx)
            
            # Check if it is possible to remove any rule
            if len(self.parameters_list) > 1:
                self.SimilarityIndex()
                
            # Compute the number of rules at the current iteration
            self.rules.append(len(self.parameters_list))
            
            # Compute the normalized firing level
            self.Lambda(x)
            
            # Update the consequent parameters of the fist rule
            self.RLS(x, y[k], xe)
            
            # Utility Measure
            if len(self.parameters_list) > 1:
                self.UtilityMeasure(X[k,], k+1)
                
            # Compute the output
            Output = sum([self.parameters_list[row][7] * xe.T @ self.parameters_list[row][2] for row in range(len(self.parameters_list))])
            
            # Store the results
            self.y_pred_training = np.append(self.y_pred_training, Output)
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
            
        # Save the rules to a dataframe
        self.parameters = pd.DataFrame(self.parameters_list, columns=['Center', 'P', 'Theta', 'ArousalIndex', 'CompatibilityMeasure', 'TimeCreation', 'NumObservations', 'lambda', 'SumLambda', 'Utility', 'sigma', 'support', 'z', 'diff_z', 'local_scatter', 'tau'])
            
    def predict(self, X):
                
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
        
        # Reshape X
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center'].shape[0])
        
        # Preallocate space for the outputs for better performance
        self.y_pred_test = np.zeros((X.shape[0]))
        
        for k in range(X.shape[0]):
            
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T
            
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
                                    
            # Compute the normalized firing level
            self.Lambda(x)
            
            # Compute the output
            Output = sum([self.parameters_list[row][7] * xe.T @ self.parameters_list[row][2] for row in range(len(self.parameters_list))])
            
            # Store the results
            self.y_pred_test[k,] = Output.item()
            
        return self.y_pred_test
            
    def NewRule(self, x, y, z, k=1., isFirst = False):
        
        if isFirst:
            
            # List of parameters
            self.parameters_list.append([x, self.omega * np.eye(x.shape[0] + 1), np.zeros((x.shape[0] + 1, 1)), 0., 1., 1., 1., 0., 0., 1., self.sigma * np.ones((x.shape[0] + 1, 1)), 1., z, np.zeros((x.shape[0] + 1, 1)), np.zeros((x.shape[0], 1, 0.))])
        
        else:
            
            # List of parameters
            self.parameters_list.append([x, self.omega * np.eye(x.shape[0] + 1), np.zeros((x.shape[0] + 1, 1)), 0., 1., k, 1., 0., 0., 1., self.sigma * np.ones((x.shape[0] + 1, 1)), 1., z, np.zeros((x.shape[0] + 1, 1)), np.zeros((x.shape[0] + 1, 1)), 0.])
        
    def CompatibilityMeasure(self, x, i):
        
        # The compatibility measure can be lower than 0 if the input data is not normalized
        # Compute the norm used to calculate the compatibility measure
        norm_cm = (np.linalg.norm(x - self.parameters_list[i][0]) / x.shape[0])
        # Compute the compatibility measure without the correlation
        CompatibilityMeasure = 1 - norm_cm if norm_cm < 1. else 0.
        
        # Update the compatibility measure
        self.parameters_list[i][11] = CompatibilityMeasure
        
        # Update the respective arousal index
        self.Arousal_Index(i)
    
    def Arousal_Index(self, i):
        
        # Atualização para todas as regras no DataFrame
        self.parameters_list[i][3] += self.beta * (1 - self.parameters_list[i][4] - self.parameters_list[i][3])
            
    def GaussianMF(self, v1, v2, sigma):
        mf = np.zeros((v1.shape))
        for j in range(v1.shape[0]):
            denominator = (2 * sigma[j,0] ** 2)
            if denominator != 0:
                mf[j,0] = math.exp( - ( v1[j,0] - v2[j,0] ) ** 2 / denominator )
            else:
                mf[j,0] = math.exp( - ( v1[j,0] - v2[j,0] ) ** 2 / 2 )
        return mf.prod()
    
    def Tau(self, x):
        for row in range(len(self.parameters_list)):
            tau = self.GaussianMF(x, self.parameters_list[row][0], self.parameters_list[row][10])
            # Evoid tau with values zero
            if abs(tau) < (10 ** -100):
                tau = (10 ** -100)
            self.parameters_list[row][15] = tau
            
    def Lambda(self, x):
        
        # Update the values of Tau
        self.Tau(x)
        
        # Compute the sum of tau
        tau_sum = 0
        for i in range(len(self.parameters_list)):
            tau_sum += self.parameters_list[i][15]
            
        for row in range(len(self.parameters_list)):
            self.parameters_list[row][7] = self.parameters_list[row][15] / tau_sum
            self.parameters_list[row][8] += self.parameters_list[row][7]
           
    def UpdateRule(self, x, y, z, i):
        # Update the number of observations in the rule
        self.parameters_list[i][6] += 1
        # Update the cluster Center
        self.parameters_list[i][0] += (self.alpha*(self.parameters_list[i][4])**(1 - self.alpha))*(x - self.parameters_list[i][0])
        # Update the cluster support
        self.parameters_list[i][11] += 1
        # Update the cluster diff z
        self.parameters_list[i][13] += ( self.parameters_list[i][12] - z ) ** 2
        # Update the cluster local scatter
        self.parameters_list[i][14] = (self.parameters_list[i][13] / ( self.parameters_list[i][11] - 1 )) ** (1/2)
        # Update the cluster radius
        self.parameters_list[i][10] = self.pi * self.parameters_list[i][10] + ( 1 - self.pi) * self.parameters_list[i][14]
    
    def UtilityMeasure(self, x, k):
        
        # List of rows to remove
        remove = []
        for i in range(len(self.parameters_list)):
            
            # Compute how long ago the rule was created
            time_diff = k - self.parameters_list[i][5]
            
            # Compute the utilit measure
            self.parameters_list[i][9] = self.parameters_list[i][8] / time_diff if time_diff != 0 else 1
            
            # Find rules with utility lower than a threshold
            if self.parameters_list[i][9] < self.e_utility:
                remove.append(i)
                    
        # Remove old rules
        if not remove:
            
            self.parameters_list = [item for i, item in enumerate(self.parameters_list) if i not in remove]

            # Inform that a rules was excluded and create no more rules
            self.ExcludedRule = 1
            
    def SimilarityIndex(self):
        
        # List of rows to remove
        remove = []
        
        # Look for indexes to remove
        for i in range(len(self.parameters_list) - 1):
            for j in range(i + 1, len(self.parameters_list)):
                vi, vj = (self.parameters_list[i][0], self.parameters_list[j][0])
                compat_ij = (1 - ((np.linalg.norm(vi - vj))))
                if compat_ij >= self.lambda1:
                    self.parameters_list[j][0] = ( (self.parameters_list[i][0] + self.parameters_list[j][0]) / 2)
                    self.parameters_list[j][1] = ( (self.parameters_list[i][1] + self.parameters_list[j][1]) / 2)
                    self.parameters_list[j][2] = np.array((self.parameters_list[i][2] + self.parameters_list[j][2]) / 2)
                    remove.append(int(i))

        # Remove rules
        if not remove:
            
            self.parameters_list = [item for i, item in enumerate(self.parameters_list) if i not in remove]
    
            # Inform that a rules was excluded
            self.ExcludedRule = 1

    def RLS(self, x, y, xe):
        for row in range(len(self.parameters_list)):
            self.parameters_list[row][1] -= (( self.parameters_list[row][7] * self.parameters_list[row][1] @ xe @ xe.T @ self.parameters_list[row][1])/(1 + self.parameters_list[row][7] * xe.T @ self.parameters_list[row][1] @ xe))
            self.parameters_list[row][2] += (self.parameters_list[row][1] @ xe * self.parameters_list[row][7] * (y - xe.T @ self.parameters_list[row][2]))
            
            
class eMG(base):
    
    def __init__(self, alpha = 0.01, lambda1 = 0.1, w = 10, sigma = 0.05, omega = 10^2, maximum_rules = 200):
        
        # Call __init__ of the base class
        super().__init__()
        
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be a float between 0 and 1.")
        if not (0 <= lambda1 <= 1):
            raise ValueError("lambda1 must be a float between 0 and 1.")
        if not (isinstance(w, int) and w > 0):  # w can be NaN or in [0, 1]
            raise ValueError("w must be an integer greater than 0.")
        if not (sigma > 0):
            raise ValueError("sigma must be a positive float.")
        if not (isinstance(omega, int) and omega > 0):
            raise ValueError("omega must be a positive integer.")
            
        # Hyperparameters
        self.alpha = alpha
        self.lambda1 = lambda1
        self.w = w
        self.sigma = sigma
        self.omega = omega
        self.maximum_rules = maximum_rules
        
        # # Model's parameters
        # self.parameters = pd.DataFrame(columns = ['Center', 'ArousalIndex', 'CompatibilityMeasure', 'NumObservations', 'Sigma', 'o', 'Theta', 'Q', 'LocalOutput'])
        # Defining the initial dispersion matrix
        self.Sigma_init = np.array([])
        # Defining the threshold for the compatibility measure
        self.Tp = None
        # Defining the threshold for the arousal index
        self.Ta = 1 - self.lambda1
        
    def get_params(self, deep=True):
        return {
            'alpha': self.alpha,
            'lambda1': self.lambda1,
            'w': self.w,
            'sigma': self.sigma,
            'omega': self.omega,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def is_numeric_and_finite(self, array):
        return np.isfinite(array).all() and np.issubdtype(np.array(array).dtype, np.number)
    
    def show_rules(self):
        rules = []
        for i in self.parameters.index:
            rule = f"Rule {i}"
            for j in range(self.parameters.loc[i,"Center"].shape[1]):
                rule = f'{rule} - {self.parameters.loc[i, "Center"][0,j]:.2f} ({self.parameters.loc[i, "Sigma"][j,j]**(1/2):.2f})'
            print(rule)
            rules.append(rule)
        
        return rules
    
    def plot_rules(self, num_points=100):
        # Set plot-wide configurations only once
        plt.rc('font', size=13)
        plt.rc('axes', titlesize=15)
        # plt.figure(figsize=(19.20, 10.80))
    
        # Determine the number of rules (rows) and attributes per rule
        num_rules = len(self.parameters.index)
        num_attributes = self.parameters.loc[self.parameters.index[0], "Center"].shape[1]
    
        # Create a figure with subplots (one per rule)
        fig, axes = plt.subplots(num_rules, 1, figsize=(8, num_rules * 4), squeeze=False, sharey=True)
    
        # Iterate through rules
        for i, rule_idx in enumerate(self.parameters.index):
            ax = axes[i, 0]  # Select the subplot for the rule
            
            # Iterate through all attributes and plot them in the same subplot
            for j in range(num_attributes):
                center = self.parameters.loc[rule_idx, "Center"][0,j]
                sigma = self.parameters.loc[rule_idx, "Sigma"][j,j]**(1/2)
                x_vals = np.linspace(center - 3 * sigma, center + 3 * sigma, num_points)
                y_vals = np.exp(-((x_vals - center) ** 2) / (2 * sigma ** 2))
                
                ax.plot(x_vals, y_vals, linewidth=3, label=f'Attr {j}: μ={center:.2f}, σ={sigma:.2f}')
            
            # Subplot settings
            ax.set_title(f'Rule {rule_idx}')
            ax.legend(loc="lower center", ncol=2)
            ax.grid(False)
    
        # Set a single y-label for the entire figure
        fig.supylabel("Membership")
        fig.supxlabel("Values")
    
        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()
    
    def plot_gaussians(self, num_points=100):
        # Set plot-wide configurations only once
        plt.rc('font', size=30)
        plt.rc('axes', titlesize=30)
        
        # Iterate through rules and attributes
        for i in self.parameters.index:
            for j in range(self.parameters.loc[i,"Center"].shape[1]):
                
                # Generate x values for smooth curve
                x_vals = np.linspace(self.parameters.loc[i,"Center"][0,j] - 3*self.parameters.loc[i, "Sigma"][j,j]**(1/2), self.parameters.loc[i,"Center"][0,j] + 3*self.parameters.loc[i, "Sigma"][j,j]**(1/2), num_points)
                y_vals = np.exp(-((x_vals - self.parameters.loc[i,"Center"][0,j])**2) / (2 * self.parameters.loc[i, "Sigma"][j,j]))
                
                # Create and configure the plot
                plt.figure(figsize=(19.20, 10.80))
                plt.plot(x_vals, y_vals, color='blue', linewidth=3, label=f'Gaussian (μ={self.parameters.loc[i,"Center"][0,j]:.2f}, σ={self.parameters.loc[i, "Sigma"][j,j]**(1/2):.2f})')
                plt.title(f'Rule {i} - Attribute {j}')
                plt.xlabel('Values')
                plt.ylabel('Membership')
                plt.legend()
                plt.grid(False)
                
                # Display the plot
                plt.show()
            
    def plot_2d_projections(self, num_points=100):
                
        """
        Plots 2D projections of 4D Gaussian distributions in a grid of subfigures.
        
        - The grid has n rows (number of data rows in self.parameters) 
          and m columns (number of 2D projections from 4D space).
        - Each subfigure represents a different Gaussian projection.
        """
    
        num_rows = len(self.parameters)  # Number of Gaussian distributions
        dim_combinations = list(combinations(range(4), 2))  # All 2D projections
        num_cols = len(dim_combinations)  # Number of projections per row
    
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    
        # Ensure axes is always a 2D array for consistent indexing
        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if num_cols == 1:
            axes = np.expand_dims(axes, axis=1)
    
        for row_idx in range(num_rows):
            # Convert 'Center' to a 1D array
            center = np.array(self.parameters.loc[row_idx, "Center"]).flatten()
            # Convert 'Sigma' (Covariance matrix) to a 2D array
            covariance_matrix = np.array(self.parameters.loc[row_idx, "Sigma"])
    
            for col_idx, dim_pair in enumerate(dim_combinations):
                ax = axes[row_idx, col_idx]  # Get subplot axis
                
                # Extract relevant 2D center and covariance submatrix
                covariance_2d = covariance_matrix[np.ix_(dim_pair, dim_pair)]
                center_2d = center[list(dim_pair)]
    
                # Create multivariate normal distribution for this 2D projection
                rv = multivariate_normal(mean=center_2d, cov=covariance_2d)
    
                # Define axis limits dynamically based on eigenvalues of covariance
                eigenvalues, _ = np.linalg.eig(covariance_2d)
                axis_limit = 3 * np.sqrt(max(eigenvalues))  # 3 standard deviations
                x = np.linspace(center_2d[0] - axis_limit, center_2d[0] + axis_limit, num_points)
                y = np.linspace(center_2d[1] - axis_limit, center_2d[1] + axis_limit, num_points)
                X, Y = np.meshgrid(x, y)
                pos = np.dstack((X, Y))
    
                # Plot the contour for this 2D projection
                ax.contour(X, Y, rv.pdf(pos), levels=10)
                ax.set_title(f"Rule {row_idx} - Attributes {dim_pair[0]} x {dim_pair[1]}")
                ax.set_xlabel(f'Attribute {dim_pair[0]}')
                ax.set_ylabel(f'Attribute {dim_pair[1]}')
                ax.set_xlim(center_2d[0] - axis_limit, center_2d[0] + axis_limit)
                ax.set_ylim(center_2d[1] - axis_limit, center_2d[1] + axis_limit)
                ax.set_aspect('equal')
    
        plt.tight_layout()
        plt.show()
         
    def fit(self, X, y):
        
        # Shape of X and y
        X_shape = X.shape
        y_shape = y.shape
        
        # Correct format X to 2d
        if len(X_shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y_shape) > 1 and y_shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y_shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X_shape[0] != y_shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
            
        # Preallocate space for the outputs for better performance
        self.y_pred_training = np.zeros((y_shape))
        self.ResidualTrainingPhase = np.zeros((y_shape))
        
        # Initialize outputs
        self.y_pred_training[0,] = y[0]
        self.ResidualTrainingPhase[0,] = 0.
            
        # Initializing the initial dispersion matrix
        self.Sigma_init = self.sigma * np.eye(X.shape[1])
        
        # Initializing the threshold for the compatibility measure
        self.Tp = chi2.ppf(1 - self.lambda1, df=X.shape[1])
        
        # Initialize the first rule
        self.NewRule(X[0,].reshape(1,-1), y[0])
        
        # Update consequent parameters
        xk = np.insert(X[0,], 0, 1, axis=0).reshape(1,X.shape[1]+1)
        self.RLS_NewRule(xk, y[0], 0)
        
        for k in range(1,X.shape[0]):
                        
            # Prepare inputs
            xk = np.insert(X[k,], 0, 1, axis=0).reshape(1,-1)
            x = X[k,].reshape(1,-1)
            
            # Compute the compatibility measure and the arousal index for all rules
            sumCompatibility = 0
            for i in range(len(self.parameters_list)):
                
                # Compute the values of the compatibility measure
                self.CompatibilityMeasure(X[k,], i)
                
                # Total value of the compatibility measure
                sumCompatibility = sumCompatibility + self.parameters_list[i][2]
                
            # Compute the normalized compatibility measure
            for i in range(len(self.parameters_list)):
                
                # Compute the values of the normalized compatibility measure
                self.parameters_list[i][8] = self.parameters_list[i][2] / sumCompatibility
                
            # Output
            Output = sum(xk @ self.parameters_list[i][6] * self.parameters_list[i][8] for i in range(len(self.parameters_list)))
                            
            # Store the results
            self.y_pred_training[k,] = Output.item()
            residual = abs(Output - y[k])
            self.ResidualTrainingPhase[k,] = residual ** 2
            
            # Look for a compatible rule
            Center = -1
            compatibility = -1
            for i in range(len(self.parameters_list)):
                chistat = self.M_Distance(X[k,].reshape(1, X.shape[1]), i)
                if self.parameters_list[i][1] < self.Ta and chistat < self.Tp:
                    if self.parameters_list[i][2] > compatibility:
                        compatibility = self.parameters_list[i][2]
                        Center = i
            
            # Check if it is necessary to create or update a rule
            if Center == -1:
                # Initialize the new rule
                self.NewRule(x, y[k])
                # Update the most compatible center
                Center = len(self.parameters_list) - 1
                # Update the RLS for the new rule
                self.RLS_NewRule(xk, y[k], Center)
            else:
                self.UpdateRule(x, y[k], Center)
            
            # Update the consequent parameters
            for i in range(len(self.parameters_list)):
                self.RLS(xk, y[k], i)
            
            # Check if it is necessary to remove a rule
            if len(self.parameters_list) > 1:
                self.Merging_Rules(x, Center)
                
            # Update the number of rules
            self.rules.append(len(self.parameters_list))
            
        # Save the rules to a dataframe
        self.parameters = pd.DataFrame(self.parameters_list, columns=['Center', 'ArousalIndex', 'CompatibilityMeasure', 'NumObservations', 'Sigma', 'o', 'Theta', 'Q', 'NormCompatibilityMeasure'])
    
    def evolve(self, X, y):
        
        # Be sure that X is with a correct shape
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center'].shape[1])
        
        # Check the format of y
        if not isinstance(y, (np.ndarray)):
            y = np.array(y, ndmin=1)
            
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
        
        for k in range(X.shape[0]):
            
            # Prepare inputs
            xk = np.insert(X[k,], 0, 1, axis=0).reshape(1,-1)
            x = X[k,].reshape(1,-1)
            
            # Compute the compatibility measure and the arousal index for all rules
            sumCompatibility = 0
            for i in range(len(self.parameters_list)):
                
                # Compute the values of the compatibility measure
                self.CompatibilityMeasure(X[k,], i)
                
                # Total value of the compatibility measure
                sumCompatibility = sumCompatibility + self.parameters_list[i][2]
                
            # Compute the normalized compatibility measure
            for i in range(len(self.parameters_list)):
                
                # Compute the values of the normalized compatibility measure
                self.parameters_list[i][8] = self.parameters_list[i][2] / sumCompatibility
                
            # Output
            Output = sum(xk @ self.parameters_list[i][6] * self.parameters_list[i][8] for i in range(len(self.parameters_list)))
            
            # Store the results
            self.y_pred_training = np.append(self.y_pred_training, Output)
            residual = abs(Output - y[k])
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,residual**2)
            
            # Look for a compatible rule
            Center = -1
            compatibility = -1
            for i in range(len(self.parameters_list)):
                chistat = self.M_Distance(X[k,].reshape(1, X.shape[1]), i)
                if self.parameters_list[i][1] < self.Ta and chistat < self.Tp:
                    if self.parameters_list[i][2] > compatibility:
                        compatibility = self.parameters_list[i][2]
                        Center = i
            
            # Check if it is necessary to create or update a rule
            if Center == -1:
                # Initialize the new rule
                self.NewRule(x, y[k])
                # Update the most compatible center
                Center = len(self.parameters_list) - 1
                # Update the RLS for the new rule
                self.RLS_NewRule(xk, y[k], Center)              
            else:
                self.UpdateRule(x, y[k], Center)
            
            # Update the consequent parameters
            for i in range(len(self.parameters_list)):
                self.RLS(xk, y[k], i)
            
            # Check if it is necessary to remove a rule
            if len(self.parameters_list) > 1:
                self.Merging_Rules(x, Center)
                
            # Update the number of rules
            self.rules.append(len(self.parameters_list))
            
        # Save the rules to a dataframe
        self.parameters = pd.DataFrame(self.parameters_list, columns=['Center', 'ArousalIndex', 'CompatibilityMeasure', 'NumObservations', 'Sigma', 'o', 'Theta', 'Q', 'NormCompatibilityMeasure'])
            
    def predict(self, X):
        
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
        
        # Reshape X
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center'].shape[1])
        
        # Preallocate space for the outputs for better performance
        self.y_pred_test = np.zeros((X.shape[0]))
        
        for k in range(X.shape[0]):
            
            xk = np.insert(X[k,], 0, 1, axis=0).reshape(1,X.shape[1]+1)
            
            # Compute the compatibility measure and the arousal index for all rules
            sumCompatibility = 0
            for i in range(len(self.parameters_list)):
                
                # Compute the values of the compatibility measure
                self.CompatibilityMeasure(X[k,], i)
                
                # Total value of the compatibility measure
                sumCompatibility = sumCompatibility + self.parameters_list[i][2]
                
            # Compute the normalized compatibility measure
            for i in range(len(self.parameters_list)):
                
                # Compute the values of the normalized compatibility measure
                self.parameters_list[i][8] = self.parameters_list[i][2] / sumCompatibility
                
            # Output
            Output = sum(xk @ self.parameters_list[i][6] * self.parameters_list[i][8] for i in range(len(self.parameters_list)))
                
            # Store the results
            self.y_pred_test[k,] = Output.item()
            
        return self.y_pred_test
    
    def validate_vector(self, u, dtype=None):
        # XXX Is order='c' really necessary?
        u = np.asarray(u, dtype=dtype, order='c')
        if u.ndim == 1:
            return u

        # Ensure values such as u=1 and u=[1] still return 1-D arrays.
        u = np.atleast_1d(u.squeeze())
        if u.ndim > 1:
            raise ValueError("Input vector should be 1-D.")
        warnings.warn(
            "scipy.spatial.distance metrics ignoring length-1 dimensions is "
            "deprecated in SciPy 1.7 and will raise an error in SciPy 1.9.",
            DeprecationWarning)
        return u


    def mahalanobis(self, u, v, VI):
        """
        Compute the Mahalanobis distance between two 1-D arrays.

        The Mahalanobis distance between 1-D arrays `u` and `v`, is defined as

        .. math::

           \\sqrt{ (u-v) V^{-1} (u-v)^T }

        where ``V`` is the covariance matrix.  Note that the argument `VI`
        is the inverse of ``V``.

        Parameters
        ----------
        u : (N,) array_like
            Input array.
        v : (N,) array_like
            Input array.
        VI : array_like
            The inverse of the covariance matrix.

        Returns
        -------
        mahalanobis : double
            The Mahalanobis distance between vectors `u` and `v`.

        Examples
        --------
        >>> from scipy.spatial import distance
        >>> iv = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
        >>> distance.mahalanobis([1, 0, 0], [0, 1, 0], iv)
        1.0
        >>> distance.mahalanobis([0, 2, 0], [0, 1, 0], iv)
        1.0
        >>> distance.mahalanobis([2, 0, 0], [0, 1, 0], iv)
        1.7320508075688772

        """
        u = self.validate_vector(u)
        v = self.validate_vector(v)
        VI = np.atleast_2d(VI)
        delta = u - v
        m = np.dot(np.dot(delta, VI), delta)
        return m
    
    def NewRule(self, x, y, isFirst = False):
        
        Q = self.omega * np.eye(x.shape[1] + 1)
        Theta = np.insert(np.zeros((x.shape[1],1)), 0, y, axis=0)
        # List of parameters
        self.parameters_list.append([x, 0., 1., 1., self.Sigma_init, np.array([]), Theta, Q, 0.])
        
    def M_Distance(self, x, i):
        dist = self.mahalanobis(x, self.parameters_list[i][0], np.linalg.inv(self.parameters_list[i][4]))
        return dist
       
    def CompatibilityMeasure(self, x, i):
        x = x.reshape(1, x.shape[0])
        dist = self.M_Distance(x, i)
        compat = math.exp(-0.5 * dist)
        if abs(compat) < (10 ** -100):
            compat = (10 ** -100)
        self.parameters_list[i][2] = compat
        return compat
            
    def Arousal_Index(self, x, i):
        x = x.reshape(1, x.shape[0])
        chistat = self.M_Distance(x, i)
        self.parameters_list[i][5] = np.append(self.parameters_list[i][5], 1) if chistat < self.Tp else np.append(self.parameters_list[i][5], 0)
        arousal = binom.cdf(sum(self.parameters_list[i][5][-self.w:]), self.w, self.lambda1) if self.parameters_list[i][3] > self.w else 0.
        self.parameters_list[i][1] = arousal
        return arousal
    
    def UpdateRule(self, x, y, i):
        # Update the number of observations in the rule
        self.parameters_list[i][3] += 1
        # Compute G
        G = (self.alpha * (self.parameters_list[i][2])**(1 - self.parameters_list[i][1]))
        # Update the cluster Center
        self.parameters_list[i][0] += G * (x - self.parameters_list[i][0])
        # Updating the dispersion matrix
        self.parameters_list[i][4] = (1 - G) * (self.parameters_list[i][4] - G * (x - self.parameters_list[i][0]).T @ (x - self.parameters_list[i][0]))
        
    def Membership_Function(self, x, i):
        dist = self.mahalanobis(x, self.parameters_list[i][0], np.linalg.inv(self.parameters_list[i][4]))
        return math.sqrt(dist)
                
    def Merging_Rules(self, x, MaxCompatibilityIdx):
        for i in range(len(self.parameters_list)):
            if MaxCompatibilityIdx != i:
                dist1 = self.M_Distance(self.parameters_list[MaxCompatibilityIdx][0], i)
                dist2 = self.M_Distance(self.parameters_list[i][0], MaxCompatibilityIdx)
                if dist1 < self.Tp or dist2 < self.Tp:
                    self.parameters_list[MaxCompatibilityIdx][0] = np.mean(np.array([self.parameters_list[i][0], self.parameters_list[MaxCompatibilityIdx][0]]), axis=0)
                    self.parameters_list[MaxCompatibilityIdx][4] = self.Sigma_init
                    self.parameters_list[MaxCompatibilityIdx][7] = self.omega * np.eye(x.shape[1] + 1)
                    self.parameters_list[MaxCompatibilityIdx][6] = (self.parameters_list[MaxCompatibilityIdx][6] * self.parameters_list[MaxCompatibilityIdx][2] + self.parameters_list[i][6] * self.parameters_list[i][2]) / (self.parameters_list[MaxCompatibilityIdx][2] + self.parameters_list[i][2])
                    self.parameters_list.pop(i)
                    
                    # Stop the loop
                    break
    
                    # # Stop creating new rules when the model exclude the first rule
                    # self.ExcludedRule = 1

    def RLS(self, xk, y, i):
        self.parameters_list[i][7] -= ((self.parameters_list[i][2] * self.parameters_list[i][7] @ xk.T @ xk @ self.parameters_list[i][7]) / (1 + self.parameters_list[i][2] * xk @ self.parameters_list[i][7] @ xk.T))
        self.parameters_list[i][6] += self.parameters_list[i][7] @ xk.T * self.parameters_list[i][2] * (y - xk @ self.parameters_list[i][6])
    
    def RLS_NewRule(self, xk, y, i):
        self.parameters_list[i][7] -= ((self.parameters_list[i][2] * self.parameters_list[i][7] @ xk.T @ xk @ self.parameters_list[i][7]) / (1 + self.parameters_list[i][2] * xk @ self.parameters_list[i][7] @ xk.T))
        self.parameters_list[i][6] += self.parameters_list[i][7] @ xk.T * self.parameters_list[i][2]

class ePL(base):
    
    def __init__(self, alpha = 0.001, beta = 0.5, lambda1 = 0.35, tau = None, s = 1000, sigma = 0.25):
        
        # Call __init__ of the base class
        super().__init__()
        
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be a float between 0 and 1.")
        if not (0 <= beta <= 1):
            raise ValueError("beta must be a float between 0 and 1.")
        if not (0 <= lambda1 <= 1):
            raise ValueError("lambda1 must be a float between 0 and 1.")
        if not (tau is None or (isinstance(tau, float) and (0 <= tau <= 1))):  # tau can be NaN or in [0, 1]
            raise ValueError("tau must be a float between 0 and 1, or None.")
        if not (isinstance(s, int) and s > 0):
            raise ValueError("s must be a positive integer.")
        if not (sigma > 0):
            raise ValueError("sigma must be a positive float.")
            
        # Hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.tau = beta if tau is None else tau
        self.s = s
        self.sigma = sigma
        
    def get_params(self, deep=True):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'lambda1': self.lambda1,
            'tau': self.tau,
            's': self.s,
            'sigma': self.sigma,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def is_numeric_and_finite(self, array):
        return np.isfinite(array).all() and np.issubdtype(np.array(array).dtype, np.number)
    
    def show_rules(self):
        rules = []
        for i in self.parameters.index:
            rule = f"Rule {i}"
            for j in range(self.parameters.loc[i,"Center"].shape[0]):
                rule = f'{rule} - {self.parameters.loc[i, "Center"][j].item():.2f} ({self.sigma:.2f})'
            print(rule)
            rules.append(rule)
        
        return rules
    
    def plot_rules(self, num_points=100):
        # Set plot-wide configurations only once
        plt.rc('font', size=13)
        plt.rc('axes', titlesize=15)
        # plt.figure(figsize=(19.20, 10.80))
    
        # Determine the number of rules (rows) and attributes per rule
        num_rules = len(self.parameters.index)
        num_attributes = self.parameters.loc[self.parameters.index[0], "Center"].shape[0]
    
        # Create a figure with subplots (one per rule)
        fig, axes = plt.subplots(num_rules, 1, figsize=(8, num_rules * 4), squeeze=False, sharey=True)
    
        # Iterate through rules
        for i, rule_idx in enumerate(self.parameters.index):
            ax = axes[i, 0]  # Select the subplot for the rule
            
            # Iterate through all attributes and plot them in the same subplot
            for j in range(num_attributes):
                center = self.parameters.loc[rule_idx, "Center"][j]
                x_vals = np.linspace(center - 3 * self.sigma, center + 3 * self.sigma, num_points)
                y_vals = np.exp(-((x_vals - center) ** 2) / (2 * self.sigma ** 2))
                
                ax.plot(x_vals, y_vals, linewidth=3, label=f'Attr {j}: μ={center.item():.2f}, σ={self.sigma:.2f}')
            
            # Subplot settings
            ax.set_title(f'Rule {rule_idx}')
            ax.legend(loc="lower center", ncol=2)
            ax.grid(False)
    
        # Set a single y-label for the entire figure
        fig.supylabel("Membership")
        fig.supxlabel("Values")
    
        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()
    
    def plot_gaussians(self, num_points=100):
        # Set plot-wide configurations only once
        plt.rc('font', size=30)
        plt.rc('axes', titlesize=30)
        
        # Iterate through rules and attributes
        for i in self.parameters.index:
            for j in range(self.parameters.loc[i,"Center"].shape[0]):
                
                # Generate x values for smooth curve
                x_vals = np.linspace(self.parameters.loc[i,"Center"][j] - 3*self.sigma, self.parameters.loc[i,"Center"][j] + 3*self.sigma, num_points)
                y_vals = np.exp(-((x_vals - self.parameters.loc[i,"Center"][j])**2) / (2 * self.sigma**2))
                
                # Create and configure the plot
                plt.figure(figsize=(19.20, 10.80))
                plt.plot(x_vals, y_vals, color='blue', linewidth=3, label=f'Gaussian (μ={self.parameters.loc[i,"Center"][j].item():.2f}, σ={self.sigma:.2f})')
                plt.title(f'Rule {i} - Attribute {j}')
                plt.xlabel('Values')
                plt.ylabel('Membership')
                plt.legend()
                plt.grid(False)
                
                # Display the plot
                plt.show()
         
    def fit(self, X, y):
        
        # Shape of X and y
        X_shape = X.shape
        y_shape = y.shape
        
        # Correct format X to 2d
        if len(X_shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y_shape) > 1 and y_shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y_shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X_shape[0] != y_shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
            
        # Prepare the first input vector
        x = X[0,].reshape((1,-1)).T
        # Compute xe
        xe = np.insert(x.T, 0, 1, axis=1).T
        
        # Preallocate space for the outputs for better performance
        self.y_pred_training = np.zeros((y_shape))
        self.ResidualTrainingPhase = np.zeros((y_shape))
        
        # Initialize outputs
        self.y_pred_training[0,] = y[0]
        self.ResidualTrainingPhase[0,] = 0.
        
        # Initialize the first rule
        self.NewRule(x, y[0], 0, True)
                
        # Update the consequent parameters of the fist rule
        self.RLS(x, y[0], xe)
        
        for k in range(1, X.shape[0]):
            
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            
            # Compute the compatibility measure and the arousal index for all rules
            MinArousal, MaxCompatibility, MaxCompatibilityIdx = (np.inf, 0, 0)
            for i in range(len(self.parameters_list)):
                
                # Update the compatibility measure and the respective arousal index (inside compatibility measure function)
                self.CompatibilityMeasure(x, i)
                
                # Find the minimum arousal index
                if self.parameters_list[i][3] < MinArousal:
                    MinArousal = self.parameters_list[i][3]
            
                # Find the maximum compatibility measure
                if self.parameters_list[i][4] > MaxCompatibility:
                    MaxCompatibility = self.parameters_list[i][4]
                    MaxCompatibilityIdx = i
            
            # Verifying the needing to creating a new rule
            if MinArousal > self.tau:
                self.NewRule(x, y[k], k+1, False)
            else:
                self.UpdateRule(x, y[k], MaxCompatibilityIdx)
            if len(self.parameters_list) > 1:
                self.SimilarityIndex()
                
            # Compute the number of rules at the current iteration
            self.rules.append(len(self.parameters_list))
            
            # Update the consequent parameters of the fist rule
            self.RLS(x, y[k], xe)
            
            # Compute the normalized firing degree
            self.Lambda(x)
            
            # Compute the output
            Output = sum([self.parameters_list[row][7] * xe.T @ self.parameters_list[row][2] for row in range(len(self.parameters_list))])
            
            # Store the results
            self.y_pred_training[k,] = Output.item()
            residual = abs(Output - y[k])
            self.ResidualTrainingPhase[k,] = residual ** 2
            
        # Save the rules to a dataframe
        self.parameters = pd.DataFrame(self.parameters_list, columns=['Center', 'P', 'Theta', 'ArousalIndex', 'CompatibilityMeasure', 'TimeCreation', 'NumObservations', 'Lambda', 'Tau'])
    
    def evolve(self, X, y):
        
        # Be sure that X is with a correct shape
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center'].shape[0])
        
        # Check the format of y
        if not isinstance(y, (np.ndarray)):
            y = np.array(y, ndmin=1)
            
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
        
        for k in range(X.shape[0]):
            
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            
            # Compute the compatibility measure and the arousal index for all rules
            MinArousal, MaxCompatibility, MaxCompatibilityIdx = (np.inf, 0, 0)
            for i in range(len(self.parameters_list)):
                
                # Update the compatibility measure and the respective arousal index (inside compatibility measure function)
                self.CompatibilityMeasure(x, i)
                
                # Find the minimum arousal index
                if self.parameters_list[i][3] < MinArousal:
                    MinArousal = self.parameters_list[i][3]
            
                # Find the maximum compatibility measure
                if self.parameters_list[i][4] > MaxCompatibility:
                    MaxCompatibility = self.parameters_list[i][4]
                    MaxCompatibilityIdx = i
            
            # Verifying the needing to creating a new rule
            if MinArousal > self.tau:
                self.NewRule(x, y[k], k+1, False)
            else:
                self.UpdateRule(x, y[k], MaxCompatibilityIdx)
            if len(self.parameters_list) > 1:
                self.SimilarityIndex()
                
            # Compute the number of rules at the current iteration
            self.rules.append(len(self.parameters_list))
            
            # Update the consequent parameters of the fist rule
            self.RLS(x, y[k], xe)
            
            # Compute the normalized firing degree
            self.Lambda(x)
            
            # Compute the output
            Output = sum([self.parameters_list[row][7] * xe.T @ self.parameters_list[row][2] for row in range(len(self.parameters_list))])
            
            # Store the results
            self.y_pred_training = np.append(self.y_pred_training, Output)
            residual = abs(Output - y[k])
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, residual**2)
        
        # Save the rules to a dataframe
        self.parameters = pd.DataFrame(self.parameters_list, columns=['Center', 'P', 'Theta', 'ArousalIndex', 'CompatibilityMeasure', 'TimeCreation', 'NumObservations', 'Lambda', 'Tau'])
    
    def predict(self, X):
        
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
        
        # Reshape X
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center'].shape[0])
        
        # Preallocate space for the outputs for better performance
        self.y_pred_test = np.zeros((X.shape[0]))
        
        for k in range(X.shape[0]):
            
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T
            
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            
            # Compute the normalized firing level
            self.Lambda(x)
            
            # Compute the output
            Output = sum([self.parameters_list[row][7] * xe.T @ self.parameters_list[row][2] for row in range(len(self.parameters_list))])
            
            # Store the results
            self.y_pred_test[k,] = Output.item()
            
        return self.y_pred_test
    
    def NewRule(self, x, y, k=1., isFirst = False):
        
        if isFirst:
            
            # List of parameters
            self.parameters_list.append([x, self.s * np.eye(x.shape[0] + 1), np.zeros((x.shape[0] + 1, 1)), 0., 1., 1., 1., 1., 1.])
        
        else:
            
            # List of parameters
            self.parameters_list.append([x, self.s * np.eye(x.shape[0] + 1), np.zeros((x.shape[0] + 1, 1)), 0., 1., k, 1., 1., 1.])
    
    def CompatibilityMeasure(self, x, i):
        
        # The compatibility measure can be lower than 0 if the input data is not normalized
        # Compute the norm used to calculate the compatibility measure
        norm_cm = (np.linalg.norm(x - self.parameters_list[i][0]) / x.shape[0])
        # Compute the compatibility measure without the correlation
        CompatibilityMeasure = 1 - norm_cm if norm_cm < 1. else 0.
        
        # Update the compatibility measure
        self.parameters_list[i][4] = CompatibilityMeasure
        
        # Update the respective arousal index
        self.Arousal_Index(i)
        
    def Arousal_Index(self, i):
        
        # Atualização para todas as regras no DataFrame
        self.parameters_list[i][3] += self.beta * (1 - self.parameters_list[i][4] - self.parameters_list[i][3])
    
    def GaussianMF(self, v1, v2):
        mf = np.zeros((v1.shape))
        for j in range(v1.shape[0]):
            denominator = (2 * self.sigma ** 2)
            if denominator != 0:
                mf[j,0] = math.exp( - ( v1[j,0] - v2[j,0] ) ** 2 / denominator )
            else:
                mf[j,0] = math.exp( - ( v1[j,0] - v2[j,0] ) ** 2 / 2 )
        return mf.prod()
    
    def Tau(self, x):
        # Compute tau
        for row in range(len(self.parameters_list)):
            tau = self.GaussianMF(x, self.parameters_list[row][0])
            # Evoid tau with values zero
            if abs(tau) < (10 ** -100):
                tau = (10 ** -100)
            self.parameters_list[row][8] = tau
            
    def Lambda(self, x):
        
        # Update the values of Tau
        self.Tau(x)
        
        # Compute the sum of tau
        tau_sum = 0
        for i in range(len(self.parameters_list)):
            tau_sum += self.parameters_list[i][8]
            
        for row in range(len(self.parameters_list)):
            self.parameters_list[row][7] = self.parameters_list[row][8] / tau_sum
            
    def UpdateRule(self, x, y, i):
        
        # Update the number of observations in the rule
        self.parameters_list[i][6] += 1
        # Update the cluster Center
        self.parameters_list[i][0] += (self.alpha*(self.parameters_list[i][4])**(1 - self.alpha))*(x - self.parameters_list[i][0])
        
    def SimilarityIndex(self):
        
        # List of rows to remove
        remove = []
        
        # Look for indexes to remove
        for i in range(len(self.parameters_list) - 1):
            for j in range(i + 1, len(self.parameters_list)):
                vi, vj = (self.parameters_list[i][0], self.parameters_list[j][0])
                compat_ij = (1 - ((np.linalg.norm(vi - vj))))
                if compat_ij >= self.lambda1:
                    self.parameters_list[j][0] = ( (self.parameters_list[i][0] + self.parameters_list[j][0]) / 2)
                    self.parameters_list[j][1] = ( (self.parameters_list[i][1] + self.parameters_list[j][1]) / 2)
                    self.parameters_list[j][2] = np.array((self.parameters_list[i][2] + self.parameters_list[j][2]) / 2)
                    remove.append(int(i))

        # Remove rules
        if not remove:
            
            self.parameters_list = [item for i, item in enumerate(self.parameters_list) if i not in remove]
    

    def RLS(self, x, y, xe):
        for row in range(len(self.parameters_list)):
            self.parameters_list[row][1] -= (( self.parameters_list[row][1] @ xe @ xe.T @ self.parameters_list[row][1])/(1 + xe.T @ self.parameters_list[row][1] @ xe))
            self.parameters_list[row][2] += (self.parameters_list[row][1] @ xe * (y - xe.T @ self.parameters_list[row][2]))
            

class exTS(base):
    
    def __init__(self, omega = 1000, mu = 1/3, epsilon = 0.01, rho = 1/2):
        
        # Call __init__ of the base class
        super().__init__()
        
        if not (isinstance(omega, int) and omega > 0):
            raise ValueError("omega must be a positive integer.")
        if not (isinstance(mu, (float,int)) and mu > 0):
            raise ValueError("mu must be greater than 0.")
        if not (0 <= epsilon <= 1):
            raise ValueError("epsilon must be a float between 0 and 1.")
        if not (0 <= rho <= 1):
            raise ValueError("rho must be a float between 0 and 1.")
            
        # Hyperparameters
        self.omega = omega
        self.mu = mu
        self.epsilon = epsilon
        self.rho = rho
        
        # Model's parameters
        self.parameters = pd.DataFrame(columns = ['Center_Z', 'Center_X', 'C', 'Theta', 'Potential', 'TimeCreation', 'NumPoints', 'mu', 'Tau', 'Lambda', 'r', 'sigma', 'increment_Center_x'])
        self.InitialPotential = 1.
        self.DataPotential = 0.
        self.InitialTheta = 0.
        self.InitialPi = 0.
        self.Beta = 0.
        self.Sigma = 0.
        self.z_last = None
    
    def get_params(self, deep=True):
        return {
            'omega': self.omega,
            'mu': self.mu,
            'epsilon': self.epsilon,
            'rho': self.rho,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def is_numeric_and_finite(self, array):
        return np.isfinite(array).all() and np.issubdtype(np.array(array).dtype, np.number)
         
    def fit(self, X, y):
        
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
            
        # Prepare the first input vector
        x = X[0,].reshape((1,-1)).T
        # Compute xe
        xe = np.insert(x.T, 0, 1, axis=1).T
        # Compute z
        z = np.concatenate((x.T, y[0].reshape(-1,1)), axis=1).T
        
        # Initialize the first rule
        self.Initialize_First_Cluster(x, y[0], z)
        # Update lambda of the first rule
        self.Update_Lambda(x)
        # Update the consequent parameters of the fist rule
        self.RLS(x, y[0], xe)
        
        for k in range(1, X.shape[0]):
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Store the previously z
            z_prev = z
            # Compute the new z
            z = np.concatenate((x.T, y[k].reshape(-1,1)), axis=1).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            
            # Compute the potential for all rules
            for i in self.parameters.index:
                self.Update_Rule_Potential(z, i, k+1)
                
            # Compute the data potential
            self.Update_Data_Potential(z_prev, z, i, k+1)
            Greater_Zero = ((self.DataPotential.item() - self.parameters['Potential']) > 0).all()
            Lower_Zero = ((self.DataPotential - self.parameters['Potential']) < 0).all()
            # Verifying the needing to creating a new rule
            if Greater_Zero == True or Lower_Zero == True:
                self.Compute_mu(x)
                mu_onethird = self.parameters['mu'].apply(lambda x: np.all(np.array(x) > self.mu)).any()
                for row in self.parameters.index:
                    if (self.parameters.loc[row, 'mu'] > self.mu).all():
                        mu_onethird = 1
                if mu_onethird == 1:                            
                    # Update an existing rule
                    self.UpdateRule(x, z)
                else:
                    # Create a new rule
                    self.Initialize_Cluster(x, z, k+1, i)
                    
            # Remove unecessary rules
            if self.parameters.shape[0] > 1:
                self.Remove_Rule(k+1)
                
            # Update consequent parameters
            self.RLS(x, y[k], xe)
            
            # Compute the number of rules at the current iteration
            self.rules.append(self.parameters.shape[0])
            
            # Compute and store the output
            Output = sum(
                self.parameters.loc[row, 'Lambda'] * xe.T @ self.parameters.loc[row, 'Theta']
                for row in self.parameters.index
            )
            # Store the output in the array
            self.y_pred_training = np.append(self.y_pred_training, Output)
            # Compute the square residual of the current iteration
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
        
        # Save the last z
        self.z_last = z
    
    def evolve(self, X, y):
        
        # Be sure that X is with a correct shape
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center_X'].shape[0])
        
        # Check the format of y
        if not isinstance(y, (np.ndarray)):
            y = np.array(y, ndmin=1)
            
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
        
        # Recover the last z
        z = self.z_last
        
        for k in range(1, X.shape[0]):
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Store the previously z
            z_prev = z
            # Compute the new z
            z = np.concatenate((x.T, y[k].reshape(-1,1)), axis=1).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Compute the potential for all rules
            
            for i in self.parameters.index:
                self.Update_Rule_Potential(z, i, k+1)
                
            # Compute the data potential
            self.Update_Data_Potential(z_prev, z, i, k+1)
            Greater_Zero = ((self.DataPotential.item() - self.parameters['Potential']) > 0).all()
            Lower_Zero = ((self.DataPotential - self.parameters['Potential']) < 0).all()
            
            # Verifying the needing to creating a new rule
            if Greater_Zero == True or Lower_Zero == True:
                self.Compute_mu(x)
                mu_onethird = 0
                for row in self.parameters.index:
                    if (self.parameters.loc[row, 'mu'] > self.mu).all():
                        mu_onethird = 1
                if mu_onethird == 1:                            
                    # Update an existing rule
                    self.UpdateRule(x, z)
                else:
                    # Create a new rule
                    self.Initialize_Cluster(x, z, k+1, i)
                    
            # Remove unecessary rules
            if self.parameters.shape[0] > 1:
                self.Remove_Rule(k+1)
                
            # Update consequent parameters
            self.RLS(x, y[k], xe)
            
            # Compute the number of rules at the current iteration
            self.rules.append(self.parameters.shape[0])
            
            # Compute and store the output
            Output = sum(
                self.parameters.loc[row, 'Lambda'] * xe.T @ self.parameters.loc[row, 'Theta']
                for row in self.parameters.index
            )
            
            # Store the output in the array
            self.y_pred_training = np.append(self.y_pred_training, Output)
            # Compute the square residual of the current iteration
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
        
        self.z_last = z
            
    def predict(self, X):
        
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center_X'].shape[0])
        
        for k in range(X.shape[0]):
            
            x = X[k,].reshape((1,-1)).T
            xe = np.insert(x.T, 0, 1, axis=1).T
            
            # Update lambda of all rules
            self.Update_Lambda(x)
            
            # Verify if lambda is nan
            if np.isnan(self.parameters['Lambda']).any():
                self.parameters['Lambda'] = 1 / self.parameters.shape[0]
                    
            # Compute and store the output
            Output = sum(
                self.parameters.loc[row, 'Lambda'] * xe.T @ self.parameters.loc[row, 'Theta']
                for row in self.parameters.index
            )
            
            # Store the output in the array
            self.y_pred_test = np.append(self.y_pred_test, Output)
            
        return self.y_pred_test[-X.shape[0]:]
        
    def Initialize_First_Cluster(self, x, y, z):
        self.parameters = pd.DataFrame([{
            'Center_Z': z,
            'Center_X': x,
            'C': self.omega * np.eye(x.shape[0] + 1),
            'Theta': np.zeros((x.shape[0] + 1, 1)),
            'Potential': self.InitialPotential,
            'TimeCreation': 1.,
            'NumPoints': 1.,
            'mu': np.zeros([x.shape[0], 1]),
            'Tau': 1.,
            'r': np.ones([x.shape[0], 1]),
            'sigma': np.ones([x.shape[0], 1]),
            'increment_Center_x': np.zeros([x.shape[0], 1])
        }])
        
        self.y_pred_training = np.append(self.y_pred_training, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0)
    
    def Initialize_Cluster(self, x, z, k, i):
        Theta = np.zeros((x.shape[0] + 1, 1))
        # Update the lambda value for all rules
        self.Update_Lambda(x)
        sigma = np.zeros([x.shape[0], 1])
        for row in self.parameters.index:
            sigma = sigma + self.parameters.loc[row, 'sigma'] 
            Theta = Theta + self.parameters.loc[row, 'Lambda'] * self.parameters.loc[row, 'Theta']
        sigma = sigma / self.parameters.shape[0]
        NewRow = pd.DataFrame([[z, x, self.omega * np.eye(x.shape[0] + 1), Theta, self.InitialPotential, k, 1., np.zeros([x.shape[0], 1]), 1., np.ones([x.shape[0], 1]), sigma, np.zeros([x.shape[0], 1])]], columns = ['Center_Z', 'Center_X', 'C', 'Theta', 'Potential', 'TimeCreation', 'NumPoints', 'mu', 'Tau', 'r', 'sigma', 'increment_Center_x'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)
    
    def Update_Rule_Potential(self, z, i, k):
        # Vectorized potential update
        numerator = (k - 1) * self.parameters.loc[i, 'Potential']
        denominator = k - 2 + self.parameters.loc[i, 'Potential'] + self.parameters.loc[i, 'Potential'] * self.Distance(z.T, self.parameters.loc[i, 'Center_Z'].T) ** 2
        self.parameters.at[i, 'Potential'] = numerator / denominator
        
    def Distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)
    
    def Update_Data_Potential(self, z_prev, z, i, k):
        self.Beta += z_prev
        self.Sigma += np.sum(z_prev ** 2)
        varTheta = np.sum(z ** 2)
        upsilon = np.sum(z * self.Beta)
        self.DataPotential = (k - 1) / ((k - 1) * (varTheta + 1) + self.Sigma - 2 * upsilon)

    def Minimum_Distance(self, z):
        distances = np.linalg.norm(self.parameters['Center_Z'].values - z, axis=1)
        return np.min(distances)
                           
    def UpdateRule(self, x, z):
        dist = []
        idx = []
        for row in self.parameters.index:
            dist.append(np.linalg.norm(self.parameters.loc[row, 'Center_Z'] - z))
            idx.append(row)
        index = idx[dist.index(min(dist))]
        
        self.parameters.at[index, 'NumPoints'] += 1
        # Efficiently update increment_Center_x and sigma
        diff_Center_x = self.parameters.at[index, 'Center_X'] - x
        self.parameters.at[index, 'increment_Center_x'] += diff_Center_x ** 2
        self.parameters.at[index, 'sigma'] = np.sqrt(self.parameters.loc[index, 'increment_Center_x'] / self.parameters.loc[index, 'NumPoints'])
        self.parameters.at[index, 'r'] = self.rho * self.parameters.loc[index, 'r'] + (1 - self.rho) * self.parameters.loc[index, 'sigma']
        
        # Update rule parameters
        self.parameters.at[index, 'Center_Z'] = z
        self.parameters.at[index, 'Center_X'] = x
        self.parameters.at[index, 'Potential'] = self.DataPotential
            
    def Update_Lambda(self, x):
        # Vectorized update of Lambda values
        self.Compute_mu(x)
        Total_Tau = np.sum(self.parameters['Tau'].values)
        if Total_Tau == 0:
            self.parameters['Lambda'] = 1.0 / len(self.parameters)
        else:
            self.parameters['Lambda'] = self.parameters['Tau'] / Total_Tau
            
    def Compute_mu(self, x):
        for row in self.parameters.index:
            mu = np.zeros([x.shape[0], 1])
            for j in range(x.shape[0]):
                mu[j,0] = math.exp( - np.linalg.norm( x[j,0] - self.parameters.loc[row, 'Center_X'][j,0] )**2 / ( 2 * self.parameters.loc[row, 'r'][j,0] ** 2 ) )
            self.parameters.at[row, 'mu'] = mu
            self.parameters.at[row, 'Tau'] = np.prod(mu)
    
    def Remove_Rule(self, k):
        N_total = np.sum(self.parameters['NumPoints'].values)
        remove = self.parameters.index[self.parameters['NumPoints'] / N_total < self.epsilon]
        if len(remove) > 0:
            self.parameters = self.parameters.drop(remove)
    
    def RLS(self, x, y, xe):
        self.Update_Lambda(x)
        for row in self.parameters.index:
            
            # Extract frequently used values to avoid repeated lookups
            lambda_val = self.parameters.loc[row, 'Lambda']
            C = self.parameters.loc[row, 'C']
            Theta = self.parameters.loc[row, 'Theta']
            
            # Compute intermediate values once
            xe_T_C = xe.T @ C
            denominator = 1 + lambda_val * xe_T_C @ xe
            
            # Update the matrix C
            C -= (lambda_val * C @ xe @ xe_T_C) / denominator
            
            # Update Theta
            residual = y - xe.T @ Theta
            Theta += (C @ xe * lambda_val * residual)
            
            # Save updated values back into the DataFrame
            self.parameters.at[row, 'C'] = C
            self.parameters.at[row, 'Theta'] = Theta
            
    
class Simpl_eTS(base):
    
    def __init__(self, omega = 1000, r = 0.1):
        
        # Call __init__ of the base class
        super().__init__()
        
        if not (isinstance(omega, int) and omega > 0):
            raise ValueError("omega must be a positive integer.")
        if not (isinstance(r, (float,int)) and r > 0):
            raise ValueError("r must be greater than 0.")
        
        # Hyperparameters
        self.omega = omega
        self.r = r
        
        # Model's parameters
        self.parameters = pd.DataFrame(columns = ['Center_Z', 'Center_X', 'C', 'Theta', 'Scatter', 'TimeCreation', 'NumPoints', 'Tau', 'Lambda'])
        self.ThresholdRemoveRules = 0.01
        self.InitialScatter = 0.
        self.DataScatter = 0.
        self.InitialTheta = 0.
        self.InitialPi = 0.
        self.Beta = 0.
        self.Sigma = 0.
        self.z_last = None
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.y_pred_training = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.y_pred_test = np.array([])
    
    def get_params(self, deep=True):
        return {
            'omega': self.omega,
            'r': self.r,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def is_numeric_and_finite(self, array):
        return np.isfinite(array).all() and np.issubdtype(np.array(array).dtype, np.number)
         
    def fit(self, X, y):
        
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
            
        # Prepare the first input vector
        x = X[0,].reshape((1,-1)).T
        # Compute xe
        xe = np.insert(x.T, 0, 1, axis=1).T
        # Compute z
        z = np.concatenate((x.T, y[0].reshape(-1,1)), axis=1).T
        # Initialize the first rule
        self.Initialize_First_Cluster(x, y[0], z)
        # Update lambda of the first rule
        self.Update_Lambda(x)
        # Update the consequent parameters of the fist rule
        self.RLS(x, y[0], xe)
        for k in range(1, X.shape[0]):
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Store the previously z
            z_prev = z
            # Compute the new z
            z = np.concatenate((x.T, y[k].reshape(-1,1)), axis=1).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Compute the scatter for all rules
            for i in self.parameters.index:
                self.Update_Rule_Scatter(z, z_prev, i, k+1)
            # Compute the data scatter
            self.Update_Data_Scatter(z_prev, z, k+1)
            # Find the rule with the minimum and maximum scatter
            IdxMinScatter = self.parameters['Scatter'].astype('float64').idxmin()
            IdxMaxScatter = self.parameters['Scatter'].astype('float64').idxmax()
            # Compute minimum delta
            Delta = self.Minimum_Distance(z)
            # Verifying the needing to creating a new rule
            if (self.DataScatter.item() < self.parameters.loc[IdxMinScatter, 'Scatter'] or self.DataScatter.item() > self.parameters.loc[IdxMaxScatter, 'Scatter']) and Delta < 0.5 * self.r:
                # Update an existing rule
                self.UpdateRule(x, z)
            elif self.DataScatter.item() < self.parameters.loc[IdxMinScatter, 'Scatter'] or self.DataScatter.item() > self.parameters.loc[IdxMaxScatter, 'Scatter']:
                # Create a new rule
                self.Initialize_Cluster(x, z, k+1, i)
                #self.parameters = self.parameters.append(self.Initialize_Cluster(x, z, k+1, i), ignore_index = True)
            elif Delta > 0.5 * self.r:
                # Update num points
                self.Update_Num_Points(z)
            # Remove unecessary rules
            if self.parameters.shape[0] > 1:
                self.Remove_Rule(k+1)
            # Update consequent parameters
            self.RLS(x, y[k], xe)
            # Compute the number of rules at the current iteration
            self.rules.append(self.parameters.shape[0])
            
            # Compute and store the output
            Output = sum(
                self.parameters.loc[row, 'Lambda'] * xe.T @ self.parameters.loc[row, 'Theta']
                for row in self.parameters.index
            )
            
            # Store the output in the array
            self.y_pred_training = np.append(self.y_pred_training, Output)
            # Compute the square residual of the current iteration
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
        
        self.z_last = z
    
    def evolve(self, X, y):
        
        # Be sure that X is with a correct shape
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center_X'].shape[0])
        
        # Check the format of y
        if not isinstance(y, (np.ndarray)):
            y = np.array(y, ndmin=1)
            
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
        
        # Recover the last z
        z = self.z_last
        
        for k in range(1, X.shape[0]):
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Store the previously z
            z_prev = z
            # Compute the new z
            z = np.concatenate((x.T, y[k].reshape(-1,1)), axis=1).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Compute the scatter for all rules
            for i in self.parameters.index:
                self.Update_Rule_Scatter(z, z_prev, i, k+1)
            # Compute the data scatter
            self.Update_Data_Scatter(z_prev, z, k+1)
            # Find the rule with the minimum and maximum scatter
            IdxMinScatter = self.parameters['Scatter'].astype('float64').idxmin()
            IdxMaxScatter = self.parameters['Scatter'].astype('float64').idxmax()
            # Compute minimum delta
            Delta = self.Minimum_Distance(z)
            # Verifying the needing to creating a new rule
            if (self.DataScatter.item() < self.parameters.loc[IdxMinScatter, 'Scatter'] or self.DataScatter.item() > self.parameters.loc[IdxMaxScatter, 'Scatter']) and Delta < 0.5 * self.r:
                # Update an existing rule
                self.UpdateRule(x, z)
            elif self.DataScatter.item() < self.parameters.loc[IdxMinScatter, 'Scatter'] or self.DataScatter.item() > self.parameters.loc[IdxMaxScatter, 'Scatter']:
                # Create a new rule
                self.Initialize_Cluster(x, z, k+1, i)
                #self.parameters = self.parameters.append(self.Initialize_Cluster(x, z, k+1, i), ignore_index = True)
            elif Delta > 0.5 * self.r:
                # Update num points
                self.Update_Num_Points(z)
            # Remove unecessary rules
            if self.parameters.shape[0] > 1:
                self.Remove_Rule(k+1)
            # Update consequent parameters
            self.RLS(x, y[k], xe)
            # Compute the number of rules at the current iteration
            self.rules.append(self.parameters.shape[0])
            
            # Compute and store the output
            Output = sum(
                self.parameters.loc[row, 'Lambda'] * xe.T @ self.parameters.loc[row, 'Theta']
                for row in self.parameters.index
            )
            
            # Store the output in the array
            self.y_pred_training = np.append(self.y_pred_training, Output)
            # Compute the square residual of the current iteration
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
        
        self.z_last = z
            
    def predict(self, X):
        
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center_X'].shape[0])
        
        for k in range(X.shape[0]):
            x = X[k,].reshape((1,-1)).T
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Update lambda of all rules
            self.Update_Lambda(x)
            
            # Compute and store the output
            Output = sum(
                self.parameters.loc[row, 'Lambda'] * xe.T @ self.parameters.loc[row, 'Theta']
                for row in self.parameters.index
            )
            
            # Store the output in the array
            self.y_pred_test = np.append(self.y_pred_test, Output)
        return self.y_pred_test[-X.shape[0]:]
        
    def Initialize_First_Cluster(self, x, y, z):
        n_features = x.shape[0]
        cluster_data = {
            "Center_Z": [z],
            "Center_X": [x],
            "C": [self.omega * np.eye(n_features + 1)],
            "Theta": [np.zeros((n_features + 1, 1))],
            "Scatter": [self.InitialScatter],
            "TimeCreation": [1.],
            "NumPoints": [1.]
        }
        self.parameters = pd.DataFrame(cluster_data)
        self.y_pred_training = np.append(self.y_pred_training, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0)  # Residual (y - y)^2 is always 0 for the first cluster
    
    def Initialize_Cluster(self, x, z, k, i):
        Theta = np.zeros((x.shape[0] + 1, 1))
        # Update the lambda value for all rules
        self.Update_Lambda(x)
        for row in self.parameters.index:
            Theta = Theta + self.parameters.loc[row, 'Lambda'] * self.parameters.loc[row, 'Theta']
        NewRow = pd.DataFrame([[z, x, self.omega * np.eye(x.shape[0] + 1), Theta, self.DataScatter.item(), k, 1.]], columns = ['Center_Z', 'Center_X', 'C', 'Theta', 'Scatter', 'TimeCreation', 'NumPoints'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)
      
    def Update_Rule_Scatter(self, z, z_prev, i, k):
        scatter = self.parameters.at[i, 'Scatter']
        self.parameters.at[i, 'Scatter'] = scatter * ((k - 2) / (k - 1)) + np.sum((z - z_prev)**2)
        
    def Distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)
    
    def Update_Data_Scatter(self, z_prev, z, k):
        self.Beta += z_prev
        self.Theta = self.Sigma + sum(z_prev**2)
        self.DataScatter = (1 / ((k - 1) * (z.shape[0]))) * ((k - 1) * sum(z**2) - 2 * sum(z * self.Beta) + self.Theta)
        
    def Minimum_Distance(self, z):
        distances = self.parameters['Center_Z'].apply(lambda Center: np.linalg.norm(Center - z))
        return distances.min()
                              
    def UpdateRule(self, x, z):
        distances = self.parameters['Center_Z'].apply(lambda Center: np.linalg.norm(Center - z))
        index = distances.idxmin()
        self.parameters.at[index, 'NumPoints'] += 1
        self.parameters.at[index, 'Center_Z'] = z
        self.parameters.at[index, 'Center_X'] = x
            
    def Update_Num_Points(self, z):
        distances = self.parameters['Center_Z'].apply(lambda Center: np.linalg.norm(Center - z))
        index = distances.idxmin()
        self.parameters.at[index, 'NumPoints'] += 1
        
    def Update_Lambda(self, x):
        self.parameters['Tau'] = self.parameters['Center_X'].apply(lambda Center: self.mu(Center, x))
        total_tau = self.parameters['Tau'].sum()
        self.parameters['Lambda'] = self.parameters['Tau'] / total_tau
    
    def mu(self, Center_X, x):
        squared_diff = (2 * (x - Center_X) / self.r)**2
        tau = np.prod(1 + squared_diff)
        return 1 / tau
    
    def Remove_Rule(self, k):
        N_total = 0
        for i in self.parameters.index:
            N_total = N_total + self.parameters.loc[i, 'NumPoints']
        remove = []
        for i in self.parameters.index:
            if self.parameters.loc[i, 'NumPoints'] / N_total < self.ThresholdRemoveRules:
                remove.append(i)
        if len(remove) > 0 and len(remove) < self.parameters.shape[0]:    
            self.parameters = self.parameters.drop(remove)
            
    def RLS(self, x, y, xe):
        self.Update_Lambda(x)
        for row in self.parameters.index:
            
            # Extract frequently used values to avoid repeated lookups
            lambda_val = self.parameters.loc[row, 'Lambda']
            C = self.parameters.loc[row, 'C']
            Theta = self.parameters.loc[row, 'Theta']
            
            # Compute intermediate values once
            xe_T_C = xe.T @ C
            denominator = 1 + lambda_val * xe_T_C @ xe
            
            # Update the matrix C
            C -= (lambda_val * C @ xe @ xe_T_C) / denominator
            
            # Update Theta
            residual = y - xe.T @ Theta
            Theta += (C @ xe * lambda_val * residual)
            
            # Save updated values back into the DataFrame
            self.parameters.at[row, 'C'] = C
            self.parameters.at[row, 'Theta'] = Theta


class eTS(base):
    
    def __init__(self, omega = 1000, r = 0.1):
        
        # Call __init__ of the base class
        super().__init__()
        
        if not (isinstance(omega, int) and omega > 0):
            raise ValueError("omega must be a positive integer.")
        if not (isinstance(r, (float,int)) and r > 0):
            raise ValueError("r must be greater than 0.")
            
        # Hyperparameters
        self.omega = omega
        self.r = r
        
        # Parameters
        self.parameters = pd.DataFrame(columns = ['Center_Z', 'Center_X', 'C', 'Theta', 'Potential', 'TimeCreation', 'NumPoints', 'Tau', 'Lambda'])
        self.InitialPotential = 1.
        self.DataPotential = 0.
        self.InitialTheta = 0.
        self.InitialPi = 0.
        self.Beta = 0.
        self.Sigma = 0.
        self.z_last = None
        # Store k for the evolving phase
        self.k = 1
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.y_pred_training = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.y_pred_test = np.array([])
    
    def get_params(self, deep=True):
        return {
            'omega': self.omega,
            'r': self.r,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def is_numeric_and_finite(self, array):
        return np.isfinite(array).all() and np.issubdtype(np.array(array).dtype, np.number)
         
    def fit(self, X, y):
        
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
        
        # Number of samples
        n_samples = X.shape[0]
        self.rules = []
        
        # Prepare the first input vector
        x = X[0,].reshape((1,-1)).T
        # Compute xe
        xe = np.insert(x.T, 0, 1, axis=1).T
        # Compute z
        z = np.concatenate((x.T, y[0].reshape(-1,1)), axis=1).T
        
        # Initialize the first rule
        self.Initialize_First_Cluster(x, y[0], z)
        self.Update_Lambda(x)  # Update lambda of the first rule
        self.RLS(x, y[0], xe)  # Update the consequent parameters of the first rule
        
        for k in range(1, n_samples):
            # Update self k
            self.k += 1
            # Prepare the k-th input vector
            x = X[k, :].reshape(-1, 1)
            
            # Store the previously z
            z_prev = z
            # Compute the new z
            z = np.concatenate((x.T, y[k].reshape(-1,1)), axis=1).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            
            # Compute the potential for all rules
            for i in self.parameters.index:
                self.Update_Rule_Potential(z, i)
                
            # Compute the data potential
            self.Update_Data_Potential(z_prev, z, i)
            # Find the rule with the maximum potential
            IdxMaxPotential = self.parameters['Potential'].astype('float64').idxmax()
            
            # Compute minimum delta
            Delta = self.Minimum_Distance(z)
            DataPotentialRatio = self.DataPotential.item() / self.parameters.loc[IdxMaxPotential, 'Potential']
            
            if self.DataPotential.item() > self.parameters.loc[IdxMaxPotential, 'Potential'] and DataPotentialRatio - Delta / self.r >= 1.:
                self.UpdateRule(x, z)  # Update an existing rule
            elif self.DataPotential > self.parameters.loc[IdxMaxPotential, 'Potential']:
                self.Initialize_Cluster(x, z, i)  # Create a new rule
                
            # Update consequent parameters
            self.RLS(x, y[k], xe)
            # Compute the number of rules at the current iteration
            self.rules.append(self.parameters.shape[0])
            
            # Compute and store the output
            Output = sum(
                self.parameters.loc[row, 'Lambda'] * xe.T @ self.parameters.loc[row, 'Theta']
                for row in self.parameters.index
            )
            
            # Store the output in the array
            self.y_pred_training = np.append(self.y_pred_training, Output)
            # Compute the square residual of the current iteration
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
        
        self.z_last = z
    
    def evolve(self, X, y):
        
        # Be sure that X is with a correct shape
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'Center_X'].shape[0])
        
        # Check the format of y
        if not isinstance(y, (np.ndarray)):
            y = np.array(y, ndmin=1)
            
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        
        # Check wheather y is 1d
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise TypeError(
                "This algorithm does not support multiple outputs. "
                "Please, give only single outputs instead."
            )
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Check wheather y is 1d
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "The number of samples of X are not compatible with the number of samples in y. "
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(y):
            raise ValueError(
                "y contains incompatible values."
                " Check y for non-numeric or infinity values"
            )
        
        # Recover the last z
        z = self.z_last
        
        for k in range(X.shape[0]):
            
            # Update k
            self.k += 1
            # Prepare the k-th input vector
            x = X[k, :].reshape(-1, 1)
            
            # Store the previously z
            z_prev = z
            # Compute the new z
            z = np.concatenate((x.T, y[k].reshape(-1,1)), axis=1).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            
            # Compute the potential for all rules
            for i in self.parameters.index:
                self.Update_Rule_Potential(z, i)
                
            # Compute the data potential
            self.Update_Data_Potential(z_prev, z, i)
            # Find the rule with the maximum potential
            IdxMaxPotential = self.parameters['Potential'].astype('float64').idxmax()
            # Compute minimum delta
            Delta = self.Minimum_Distance(z)
            DataPotentialRatio = self.DataPotential.item() / self.parameters.loc[IdxMaxPotential, 'Potential']
            
            if self.DataPotential.item() > self.parameters.loc[IdxMaxPotential, 'Potential'] and DataPotentialRatio - Delta / self.r >= 1.:
                self.UpdateRule(x, z)  # Update an existing rule
            elif self.DataPotential > self.parameters.loc[IdxMaxPotential, 'Potential']:
                self.Initialize_Cluster(x, z, i)  # Create a new rule
                
            # Update consequent parameters
            self.RLS(x, y[k], xe)
            # Compute the number of rules at the current iteration
            self.rules.append(self.parameters.shape[0])
            
            # Compute and store the output
            Output = sum(
                self.parameters.loc[row, 'Lambda'] * xe.T @ self.parameters.loc[row, 'Theta']
                for row in self.parameters.index
            )
            
            # Store the output in the array
            self.y_pred_training = np.append(self.y_pred_training, Output)
            # Compute the square residual of the current iteration
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
    
        self.z_last = z
            
    def predict(self, X):
        
        # Correct format X to 2d
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
            
        # Check if the inputs contain valid numbers
        if not self.is_numeric_and_finite(X):
            raise ValueError(
                "X contains incompatible values."
                " Check X for non-numeric or infinity values"
            )
            
        # Reshape X to match the dimension of the cluster Centers
        expected_shape = self.parameters.loc[self.parameters.index[0], 'Center_X'].shape[0]
        if X.shape[1] != expected_shape:
            X = X.reshape(-1, expected_shape)
        
        # Preallocate output array for efficiency
        y_pred_test = np.zeros(X.shape[0])

        for k in range(X.shape[0]):
            x = X[k, :].reshape(-1, 1)  # Prepare the input vector
            xe = np.insert(x.T, 0, 1, axis=1).T
            
            # Update lambda of all rules
            self.Update_Lambda(x)
            
            # Verify if lambda is nan
            if np.isnan(self.parameters['Lambda']).any():
                self.parameters['Lambda'] = 1 / self.parameters.shape[0]
                
            # Compute the output as a dot product
            Output = sum(
                self.parameters.loc[row, 'Lambda'] * xe.T @ self.parameters.loc[row, 'Theta']
                for row in self.parameters.index
            )
            
            # Store the output in the array
            y_pred_test[k] = Output
        
        # Update the class variable and return the recent outputs
        self.y_pred_test = np.append(self.y_pred_test, y_pred_test)
            
        return y_pred_test
        
    def Initialize_First_Cluster(self, x, y, z):
        self.parameters = pd.DataFrame([{
            'Center_Z': z,
            'Center_X': x,
            'C': self.omega * np.eye(x.shape[0] + 1),
            'Theta': np.zeros((x.shape[0] + 1, 1)),
            'Potential': self.InitialPotential,
            'TimeCreation': 1.0,
            'NumPoints': 1
        }])
        self.y_pred_training = np.append(self.y_pred_training, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0)
    
    def Initialize_Cluster(self, x, z, i):
        Theta = np.sum(
            [self.parameters.loc[row, 'Lambda'] * self.parameters.loc[row, 'Theta']
             for row in self.parameters.index], axis=0
        )
        new_row = {
            'Center_Z': z,
            'Center_X': x,
            'C': self.omega * np.eye(x.shape[0] + 1),
            'Theta': Theta,
            'Potential': self.InitialPotential,
            'TimeCreation': self.k,
            'NumPoints': 1
        }
        self.parameters = pd.concat([self.parameters, pd.DataFrame([new_row])], ignore_index=True)

    def Update_Rule_Potential(self, z, i):
        dist = self.Distance(z.T, self.parameters.loc[i, 'Center_Z'].T)
        numerator = (self.k - 1) * self.parameters.loc[i, 'Potential']
        denominator = (self.k - 2 + self.parameters.loc[i, 'Potential'] +
                       self.parameters.loc[i, 'Potential'] * dist**2)
        self.parameters.at[i, 'Potential'] = numerator / denominator
        
    def Distance(self, p1, p2):
        distance = np.linalg.norm(p1 - p2)
        return distance
    
    def Update_Data_Potential(self, z_prev, z, i):
        self.Beta = self.Beta + z_prev
        self.Sigma = self.Sigma + sum(z_prev**2)
        varTheta = sum(z**2)
        upsilon = sum(z*self.Beta)
        self.DataPotential = ((self.k - 1)) / ((self.k - 1) * (varTheta + 1) + self.Sigma - 2*upsilon)
        
    def Minimum_Distance(self, z):
        distances = np.linalg.norm(np.stack(self.parameters['Center_Z']) - z, axis=1)
        return np.min(distances)
                           
    def UpdateRule(self, x, z):
        distances = np.linalg.norm(np.stack(self.parameters['Center_Z']) - z, axis=1)
        index = np.argmin(distances)
        self.parameters.at[index, 'NumPoints'] += 1
        self.parameters.at[index, 'Center_Z'] = z
        self.parameters.at[index, 'Center_X'] = x
        self.parameters.at[index, 'Potential'] = self.DataPotential
            
    def Update_Lambda(self, x):
        self.parameters['Tau'] = self.parameters['Center_X'].apply(
            lambda Center_x: self.mu(Center_x, x)
        )
        total_tau = np.sum(self.parameters['Tau'])
        if total_tau == 0:
            self.parameters['Lambda'] = 1.0 / len(self.parameters)
        else:
            self.parameters['Lambda'] = self.parameters['Tau'] / total_tau
    
    def mu(self, Center_X, x):
        distances = np.linalg.norm(Center_X - x, axis=0)**2
        tau = np.exp(-4 / self.r**2 * distances).prod()
        return tau
    
    def RLS(self, x, y, xe):
        self.Update_Lambda(x)
        for row in self.parameters.index:
            
            # Extract frequently used values to avoid repeated lookups
            lambda_val = self.parameters.loc[row, 'Lambda']
            C = self.parameters.loc[row, 'C']
            Theta = self.parameters.loc[row, 'Theta']
            
            # Compute intermediate values once
            xe_T_C = xe.T @ C
            denominator = 1 + lambda_val * xe_T_C @ xe
            
            # Update the matrix C
            C -= (lambda_val * C @ xe @ xe_T_C) / denominator
            
            # Update Theta
            residual = y - xe.T @ Theta
            Theta += (C @ xe * lambda_val * residual)
            
            # Save updated values back into the DataFrame
            self.parameters.at[row, 'C'] = C
            self.parameters.at[row, 'Theta'] = Theta