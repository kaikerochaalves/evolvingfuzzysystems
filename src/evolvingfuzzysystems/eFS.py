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

# Importing libraries
# Binomial cumulative distribution function
from scipy.stats import binom
# Inverse chi square
from scipy.stats.distributions import chi2

class base():
    
    def n_rules(self):
        return self.rules[-1]
    
    def output_training(self):
        return self.OutputTrainingPhase

class ePL_KRLS_DISCO(base):
    
    def __init__(self, alpha = 0.001, beta = 0.05, lambda1 = 0.0000001, sigma = 0.5, tau = 0.05, omega = 1, e_utility = 0.05):
        
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
        
        # Parameters
        self.parameters = pd.DataFrame(columns = ['center', 'dictionary', 'nu', 'P', 'Q', 'theta','arousal_index', 'utility', 'sum_lambda', 'time_creation', 'compatibility_measure', 'old_center', 'tau', 'lambda'])
        # Parameters used to calculate the utility measure
        self.epsilon = []
        self.eTil = [0.]
        # Monitoring if some rule was excluded
        self.ExcludedRule = 0
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
    
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
        # Initialize the first rule
        self.Initialize_First_Cluster(x, y[0])
        
        for k in range(1, X.shape[0]):
            
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            
            # Compute the compatibility measure and the arousal index for all rules
            for i in self.parameters.index:
                self.Compatibility_Measure(x, i)
                self.Arousal_Index(i)
            
            # Find the minimum arousal index
            MinIndexArousal = self.parameters['arousal_index'].astype('float64').idxmin()
            
            # Find the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['compatibility_measure'].astype('float64').idxmax()
            
            # Verifying the needing to creating a new rule
            if self.parameters.loc[MinIndexArousal, 'arousal_index'] > self.tau and self.ExcludedRule == 0 and self.epsilon != []:
                self.Initialize_Cluster(x, y[k], k+1, MaxIndexCompatibility)
            else:
                self.Rule_Update(x, y[k], MaxIndexCompatibility)
                self.KRLS(x, y[k], MaxIndexCompatibility)
            self.Lambda(x)
            
            if self.parameters.shape[0] > 1:
                self.Utility_Measure(X[k,], k+1)
            self.rules.append(self.parameters.shape[0])
            
            # Finding the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['compatibility_measure'].astype('float64').idxmax()
            
            # Computing the output
            Output = 0
            for ni in range(self.parameters.loc[MaxIndexCompatibility, 'dictionary'].shape[1]):
                Output = Output + self.parameters.loc[MaxIndexCompatibility, 'theta'][ni] * self.Kernel_Gaussiano(self.parameters.loc[MaxIndexCompatibility, 'dictionary'][:,ni].reshape(-1,1), x)
                
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
            
            # Updating epsilon and e_til
            quociente = math.exp(-0.8 * self.eTil[-1] - abs(Output - y[k]))
            
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
    
    def evolve(self, X, y):
        
        # Be sure that X is with a correct shape
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'center'].shape[0])
        
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
            
            # Compute the compatibility measure and the arousal index for all rules
            for i in self.parameters.index:
                self.Compatibility_Measure(x, i)
                self.Arousal_Index(i)
            
            # Find the minimum arousal index
            MinIndexArousal = self.parameters['arousal_index'].astype('float64').idxmin()
            
            # Find the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['compatibility_measure'].astype('float64').idxmax()
            
            # Verifying the needing to creating a new rule
            if self.parameters.loc[MinIndexArousal, 'arousal_index'] > self.tau and self.ExcludedRule == 0 and self.epsilon != []:
                self.Initialize_Cluster(x, y[k], k+1, MaxIndexCompatibility)
            else:
                self.Rule_Update(x, y[k], MaxIndexCompatibility)
                self.KRLS(x, y[k], MaxIndexCompatibility)
            self.Lambda(x)
            
            if self.parameters.shape[0] > 1:
                self.Utility_Measure(X[k,], k+1)
            self.rules.append(self.parameters.shape[0])
            
            # Finding the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['compatibility_measure'].astype('float64').idxmax()
            
            # Computing the output
            Output = 0
            for ni in range(self.parameters.loc[MaxIndexCompatibility, 'dictionary'].shape[1]):
                Output = Output + self.parameters.loc[MaxIndexCompatibility, 'theta'][ni] * self.Kernel_Gaussiano(self.parameters.loc[MaxIndexCompatibility, 'dictionary'][:,ni].reshape(-1,1), x)
                
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
            
            # Updating epsilon and e_til
            quociente = math.exp(-0.8 * self.eTil[-1] - abs(Output - y[k]))
            
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
            
        for k in range(X.shape[0]):
            
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            
            # Computing the compatibility measure
            for i in self.parameters.index:
                self.Compatibility_Measure(x, i)
                
            # Finding the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['compatibility_measure'].astype('float64').idxmax()
            
            # Computing the output
            Output = 0
            for ni in range(self.parameters.loc[MaxIndexCompatibility, 'dictionary'].shape[1]):
                Output = Output + self.parameters.loc[MaxIndexCompatibility, 'theta'][ni] * self.Kernel_Gaussiano(self.parameters.loc[MaxIndexCompatibility, 'dictionary'][:,ni].reshape(-1,1), x)
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
            
        return self.OutputTestPhase
        
    def Initialize_First_Cluster(self, x, y):
        kernel_value = self.Kernel_Gaussiano(x, x)
        Q = np.linalg.inv(np.ones((1,1)) * (self.lambda1 + kernel_value))
        theta = Q*y
        self.parameters = pd.DataFrame([{
            'center': x,
            'dictionary': x,
            'nu': float(self.sigma),
            'P': np.ones((1,1)),
            'Q': Q,
            'theta': theta,
            'arousal_index': 0.,
            'utility': 1.,
            'sum_lambda': 0.,
            'num_observations': 1.,
            'time_creation': 1.,
            'compatibility_measure': 1.,
            'old_center': np.zeros((x.shape[0],1)),
            'tau': 1.
        }])
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,0.)
    
    def Initialize_Cluster(self, x, y, k, i):
        kernel_value = self.Kernel_Gaussiano(x, x)
        Q = np.linalg.inv(np.ones((1,1)) * (self.lambda1 + kernel_value))
        theta = Q*y
        # Compute nu
        distance = np.linalg.norm(x - self.parameters.loc[i, 'center'])
        log_epsilon = math.sqrt(-2 * np.log(max(self.epsilon)))
        nu = float(distance / log_epsilon)
        NewRow = pd.DataFrame([{
            'center': x,
            'dictionary': x,
            'nu': nu,
            'P': np.ones((1,1)),
            'Q': Q,
            'theta': theta,
            'arousal_index': 0.,
            'utility': 1.,
            'sum_lambda': 0.,
            'num_observations': 1.,
            'time_creation': k,
            'compatibility_measure': 1.,
            'old_center': np.zeros((x.shape[0],1)),
            'tau': 1.
        }])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)
    
    def Kernel_Gaussiano(self, Vector1, Vector2):
        distance = np.linalg.norm(Vector1 - Vector2)**2
        return math.exp(-distance / (2 * self.sigma**2))
    
    def Compatibility_Measure(self, x, i):
        
        
        # Verificar se a correlação é NaN
        if (not np.all(np.isfinite(x)) or np.std(x, axis=0).min() == 0) or (not np.all(np.isfinite(self.parameters.loc[i, 'center'])) or np.std(self.parameters.loc[i, 'center'], axis=0).min() == 0):
            compatibility_measure = 1 - (np.linalg.norm(x - self.parameters.loc[i, 'center']) / x.shape[0])
        else:
            # Calcular a correlação uma vez
            correlation = np.corrcoef(self.parameters.loc[i, 'center'].T, x.T)[0, 1]
            compatibility_measure = (1 - (np.linalg.norm(x - self.parameters.loc[i, 'center']) / x.shape[0])) * ((correlation + 1) / 2)
        
        # Atualizar a medida de compatibilidade
        self.parameters.at[i, 'compatibility_measure'] = compatibility_measure
 
    def Arousal_Index(self, i):
        # Atualização para todas as regras no DataFrame
        self.parameters['arousal_index'] += self.beta * (1 - self.parameters['compatibility_measure'] - self.parameters['arousal_index'])


    
    def Rule_Update(self, x, y, i):
        # Incrementar o número de observações
        self.parameters.loc[i, 'num_observations'] += 1
        
        # Atualizar o centro antigo e o centro atual de forma direta
        old_center = self.parameters.loc[i, 'center']
        compatibility_adjustment = self.alpha * (self.parameters.loc[i, 'compatibility_measure']) ** (1 - self.parameters.loc[i, 'arousal_index'])
        new_center = old_center + compatibility_adjustment * (x - old_center)
        
        self.parameters.loc[i, ['old_center', 'center']] = [old_center, new_center]
                       
    def Lambda(self, x):
        # Calculando o somatório de 'tau' uma vez
        tau_sum = self.parameters['tau'].sum()
        
        # Atualizando a coluna 'lambda' vetorizadamente
        self.parameters['lambda'] = self.parameters['tau'] / tau_sum
        
        # Atualizando 'sum_lambda' acumulativamente
        self.parameters['sum_lambda'] += self.parameters['lambda']
            
    def Utility_Measure(self, x, k):
        # Calcular o tempo desde a criação
        time_diff = k - self.parameters['time_creation']
    
        # Evitar divisões por zero
        with np.errstate(divide='ignore', invalid='ignore'):
            self.parameters['utility'] = np.where(
                time_diff == 0, 
                1,  # Caso em que o tempo de criação é igual a k
                self.parameters['sum_lambda'] / time_diff
            )
    
        # Identificar regras com utilidade menor que o limite
        remove = self.parameters[self.parameters['utility'] < self.e_utility].index
    
        # Remover as regras inadequadas
        if not remove.empty:
            self.parameters.drop(index=remove, inplace=True)
    
            # Indicar que uma regra foi excluída
            self.ExcludedRule = 1
            
    def KRLS(self, x, y, i):
        
        # Validar número de observações
        num_obs = self.parameters.loc[i, 'num_observations']
        if num_obs <= 0:
            raise ValueError("Número de observações deve ser maior que zero para evitar divisão por zero.")
    
        # Atualizar 'nu' (kernel size)
        center = self.parameters.loc[i, 'center']
        old_center = self.parameters.loc[i, 'old_center']
        nu = self.parameters.loc[i, 'nu']
        norm_diff = np.linalg.norm(x - center)
        center_shift = np.linalg.norm(center - old_center)
    
        self.parameters.at[i, 'nu'] = math.sqrt(
            nu**2 + (norm_diff**2 - nu**2) / num_obs + (num_obs - 1) * center_shift**2 / num_obs
        )
    
        # Calcular vetor g
        dictionary = self.parameters.loc[i, 'dictionary']
        g = np.array([self.Kernel_Gaussiano(dictionary[:, ni].reshape(-1, 1), x) for ni in range(dictionary.shape[1])]).reshape(-1, 1)
    
        # Calcular z, r, erro estimado
        z = np.matmul(self.parameters.loc[i, 'Q'], g)
        r = self.lambda1 + 1 - np.matmul(z.T, g).item()
        estimated_error = y - np.matmul(g.T, self.parameters.loc[i, 'theta'])
    
        # Calcular distâncias
        distance = np.linalg.norm(dictionary - x, axis=0)
        min_distance = np.min(distance)
    
        # Critério de novidade
        if min_distance > 0.1 * self.parameters.loc[i, 'nu']:
            # Atualizar dicionário
            self.parameters.at[i, 'dictionary'] = np.hstack([dictionary, x])
    
            # Atualizar Q
            Q = self.parameters.loc[i, 'Q']
            sizeQ = Q.shape[0]
            new_Q = np.zeros((sizeQ + 1, sizeQ + 1))
            new_Q[:sizeQ, :sizeQ] = Q + (1 / r) * np.matmul(z, z.T)
            new_Q[-1, -1] = (1 / r) * self.omega
            new_Q[:sizeQ, -1] = new_Q[-1, :sizeQ] = -(1 / r) * z.flatten()
            self.parameters.at[i, 'Q'] = new_Q
    
            # Atualizar P
            P = self.parameters.loc[i, 'P']
            new_P = np.zeros((P.shape[0] + 1, P.shape[1] + 1))
            new_P[:P.shape[0], :P.shape[1]] = P
            new_P[-1, -1] = self.omega
            self.parameters.at[i, 'P'] = new_P
    
            # Atualizar theta
            theta = self.parameters.loc[i, 'theta']
            self.parameters.at[i, 'theta'] = np.vstack([theta - (z * (1 / r) * estimated_error), (1 / r) * estimated_error])
        else:
            # Atualizar P e theta (caso de baixa novidade)
            P = self.parameters.loc[i, 'P']
            q = np.matmul(P, z) / (1 + np.matmul(np.matmul(z.T, P), z))
            self.parameters.at[i, 'P'] = P - np.matmul(q, np.matmul(z.T, P))
            self.parameters.at[i, 'theta'] += np.matmul(self.parameters.loc[i, 'Q'], q) * estimated_error

            
class ePL_plus(base):
    
    def __init__(self, alpha = 0.001, beta = 0.1, lambda1 = 0.35, tau = None, omega = 1000, sigma = 0.25, e_utility = 0.05, pi = 0.5):
        
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
        
        # Model's parameters
        self.parameters = pd.DataFrame(columns = ['center', 'P', 'Gamma', 'ArousalIndex', 'CompatibilityMeasure', 'TimeCreation', 'NumObservations', 'tau', 'lambda', 'SumLambda', 'Utility', 'sigma', 'support', 'z', 'diff_z', 'local_scatter'])
        # Monitoring if some rule was excluded
        self.ExcludedRule = 0
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
    
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
            for i in self.parameters.index:
                self.Compatibility_Measure(x, i)
                self.Arousal_Index(i)
            # Find the minimum arousal index
            MinIndexArousal = self.parameters['ArousalIndex'].astype('float64').idxmin()
            # Find the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['CompatibilityMeasure'].astype('float64').idxmax()
            # Verifying the needing to creating a new rule
            if self.parameters.loc[MinIndexArousal, 'ArousalIndex'] > self.tau:
                self.Initialize_Cluster(x, y[k], z, k+1)
            else:
                self.Rule_Update(x, y[k], z, MaxIndexCompatibility)
            if self.parameters.shape[0] > 1:
                self.Similarity_Index()
            # Compute the number of rules at the current iteration
            self.rules.append(self.parameters.shape[0])
            # Update the consequent parameters of the fist rule
            self.RLS(x, y[k], xe)
            # Compute activation degree
            self.Compute_tau(x)
            # Compute the normalized firing level
            self.Lambda(x)
            # Utility Measure
            if self.parameters.shape[0] > 1:
                self.Utility_Measure(X[k,], k+1)
            # Compute the output
            Output = 0
            for row in self.parameters.index:
                Output = Output + self.parameters.loc[row, 'lambda'] * xe.T @ self.parameters.loc[row, 'Gamma']
            Output = Output / sum(self.parameters['lambda'])
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
    
    def evolve(self, X, y):
        
        # Be sure that X is with a correct shape
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'center'].shape[0])
        
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
            for i in self.parameters.index:
                self.Compatibility_Measure(x, i)
                self.Arousal_Index(i)
            # Find the minimum arousal index
            MinIndexArousal = self.parameters['ArousalIndex'].astype('float64').idxmin()
            # Find the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['CompatibilityMeasure'].astype('float64').idxmax()
            # Verifying the needing to creating a new rule
            if self.parameters.loc[MinIndexArousal, 'ArousalIndex'] > self.tau:
                self.Initialize_Cluster(x, y[k], z, k+1)
            else:
                self.Rule_Update(x, y[k], z, MaxIndexCompatibility)
            if self.parameters.shape[0] > 1:
                self.Similarity_Index()
            # Compute the number of rules at the current iteration
            self.rules.append(self.parameters.shape[0])
            # Update the consequent parameters of the fist rule
            self.RLS(x, y[k], xe)
            # Compute activation degree
            self.Compute_tau(x)
            # Compute the normalized firing level
            self.Lambda(x)
            # Utility Measure
            if self.parameters.shape[0] > 1:
                self.Utility_Measure(X[k,], k+1)
            # Compute the output
            Output = 0
            for row in self.parameters.index:
                Output = Output + self.parameters.loc[row, 'lambda'] * xe.T @ self.parameters.loc[row, 'Gamma']
            Output = Output / sum(self.parameters['lambda'])
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
            
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
            
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'center'].shape[0])
        for k in range(X.shape[0]):
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Compute activation degree
            self.Compute_tau(x)
            # Compute the normalized firing level
            self.Lambda(x)
            # Compute the output
            Output = 0
            for row in self.parameters.index:
                Output = Output + self.parameters.loc[row, 'lambda'] * xe.T @ self.parameters.loc[row, 'Gamma']
            Output = Output / sum(self.parameters['lambda'])
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
        return self.OutputTestPhase[-X.shape[0]:]
        
    def Initialize_First_Cluster(self, x, y, z):
        self.parameters = pd.DataFrame([[x, self.omega * np.eye(x.shape[0] + 1), np.zeros((x.shape[0] + 1, 1)), 0., 1., 1., 1., 0., 0., 1., self.sigma * np.ones((x.shape[0] + 1, 1)), 1., z, np.zeros((x.shape[0] + 1, 1)), np.zeros((x.shape[0], 1))]], columns = ['center', 'P', 'Gamma', 'ArousalIndex', 'CompatibilityMeasure', 'TimeCreation', 'NumObservations', 'lambda', 'SumLambda', 'Utility', 'sigma', 'support', 'z', 'diff_z', 'local_scatter'])
        Output = y
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y)**2)
    
    def Initialize_Cluster(self, x, y, z, k):
        NewRow = pd.DataFrame([[x, self.omega * np.eye(x.shape[0] + 1), np.zeros((x.shape[0] + 1, 1)), 0., 1., k, 1., 0., 0., 1., self.sigma * np.ones((x.shape[0] + 1, 1)), 1., z, np.zeros((x.shape[0] + 1, 1)), np.zeros((x.shape[0] + 1, 1))]], columns = ['center', 'P', 'Gamma', 'ArousalIndex', 'CompatibilityMeasure', 'TimeCreation', 'NumObservations', 'lambda', 'SumLambda', 'Utility', 'sigma', 'support', 'z', 'diff_z', 'local_scatter'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)

    def Compatibility_Measure(self, x, i):
        self.parameters.at[i, 'CompatibilityMeasure'] = (1 - (np.linalg.norm(x - self.parameters.loc[i, 'center']))/x.shape[0] )
            
    def Arousal_Index(self, i):
        self.parameters.at[i, 'ArousalIndex'] = self.parameters.loc[i, 'ArousalIndex'] + self.beta*(1 - self.parameters.loc[i, 'CompatibilityMeasure'] - self.parameters.loc[i, 'ArousalIndex'])
    
    def mu(self, x1, x2, row, j):
        if ( 2 * self.parameters.loc[row, 'sigma'][j,0] ** 2 ) != 0:
            mu = math.exp( - ( x1 - x2 ) ** 2 / ( 2 * self.parameters.loc[row, 'sigma'][j,0] ** 2 ) )
        else:
            mu = math.exp( - ( x1 - x2 ) ** 2 / ( 2 ) )
        return mu
    
    def Compute_tau(self, x):
        for row in self.parameters.index:
            tau = 1
            for j in range(x.shape[0]):
                tau = tau * self.mu(x[j], self.parameters.loc[row, 'center'][j,0], row, j)
            # Evoid mtau with values zero
            if abs(tau) < (10 ** -100):
                tau = (10 ** -100)
            self.parameters.at[row, 'tau'] = tau
            
    def Lambda(self, x):
        for row in self.parameters.index:
            self.parameters.at[row, 'lambda'] = self.parameters.loc[row, 'tau'] / sum(self.parameters['tau'])
            self.parameters.at[row, 'SumLambda'] = self.parameters.loc[row, 'SumLambda'] + self.parameters.loc[row, 'lambda']
            
    def Utility_Measure(self, x, k):
        # Calculating the utility
        remove = []
        for i in self.parameters.index:
            if (k - self.parameters.loc[i, 'TimeCreation']) == 0:
                self.parameters.at[i, 'Utility'] = 1
            else:
                self.parameters.at[i, 'Utility'] = self.parameters.loc[i, 'SumLambda'] / (k - self.parameters.loc[i, 'TimeCreation'])
            if self.parameters.loc[i, 'Utility'] < self.e_utility:
                remove.append(i)
        if len(remove) > 0:    
            self.parameters = self.parameters.drop(remove)  
           
    def Rule_Update(self, x, y, z, i):
        # Update the number of observations in the rule
        self.parameters.loc[i, 'NumObservations'] = self.parameters.loc[i, 'NumObservations'] + 1
        # Update the cluster center
        self.parameters.at[i, 'center'] = self.parameters.loc[i, 'center'] + (self.alpha*(self.parameters.loc[i, 'CompatibilityMeasure'])**(1 - self.alpha))*(x - self.parameters.loc[i, 'center'])
        # Update the cluster support
        self.parameters.at[i, 'support'] = self.parameters.loc[i, 'support'] + 1
        # Update the cluster diff z
        self.parameters.at[i, 'diff_z'] = self.parameters.loc[i, 'diff_z'] + ( self.parameters.loc[i, 'z'] - z ) ** 2
        # Update the cluster local scatter
        self.parameters.at[i, 'local_scatter'] = (self.parameters.loc[i, 'diff_z'] / ( self.parameters.loc[i, 'support'] - 1 )) ** (1/2)
        # Update the cluster radius
        self.parameters.at[i, 'sigma'] = self.pi * self.parameters.loc[i, 'sigma'] + ( 1 - self.pi) * self.parameters.at[i, 'local_scatter']
        
    def Similarity_Index(self):
        l = []
        for i in range(self.parameters.shape[0] - 1):
			#if i < len(self.clusters) - 1:
            for j in range(i + 1, self.parameters.shape[0]):
                vi, vj = self.parameters.iloc[i,0], self.parameters.iloc[j,0]
                compat_ij = (1 - ((np.linalg.norm(vi - vj))))
                if compat_ij >= self.lambda1:
                    self.parameters.at[self.parameters.index[j], 'center'] = ( (self.parameters.loc[self.parameters.index[i], 'center'] + self.parameters.loc[self.parameters.index[j], 'center']) / 2)
                    self.parameters.at[self.parameters.index[j], 'P'] = ( (self.parameters.loc[self.parameters.index[i], 'P'] + self.parameters.loc[self.parameters.index[j], 'P']) / 2)
                    self.parameters.at[self.parameters.index[j], 'Gamma'] = np.array((self.parameters.loc[self.parameters.index[i], 'Gamma'] + self.parameters.loc[self.parameters.index[j], 'Gamma']) / 2)
                    l.append(int(i))

        self.parameters.drop(index=self.parameters.index[l,], inplace=True)

    def RLS(self, x, y, xe):
        for row in self.parameters.index:
            self.parameters.at[row, 'P'] = self.parameters.loc[row, 'P'] - (( self.parameters.loc[row, 'lambda'] * self.parameters.loc[row, 'P'] @ xe @ xe.T @ self.parameters.loc[row, 'P'])/(1 + self.parameters.loc[row, 'lambda'] * xe.T @ self.parameters.loc[row, 'P'] @ xe))
            self.parameters.at[row, 'Gamma'] = self.parameters.loc[row, 'Gamma'] + (self.parameters.loc[row, 'P'] @ xe * self.parameters.loc[row, 'lambda'] * (y - xe.T @ self.parameters.loc[row, 'Gamma']))
            
            
class eMG(base):
    
    def __init__(self, alpha = 0.01, lambda1 = 0.1, w = 10, sigma = 0.05, omega = 10^2, maximum_rules = 200):
        
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
        
        # Model's parameters
        self.parameters = pd.DataFrame(columns = ['center', 'ArousalIndex', 'CompatibilityMeasure', 'NumObservations', 'Sigma', 'o', 'Gamma', 'Q', 'LocalOutput'])
        # Defining the initial dispersion matrix
        self.Sigma_init = np.array([])
        # Defining the threshold for the compatibility measure
        self.Tp = None
        # Defining the threshold for the arousal index
        self.Ta = 1 - self.lambda1
        # Monitoring if some rule was excluded
        self.ExcludedRule = 0
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
        
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
            
        # Initializing the initial dispersion matrix
        self.Sigma_init = self.sigma * np.eye(X.shape[1])
        # Initializing the threshold for the compatibility measure
        self.Tp = chi2.ppf(1 - self.lambda1, df=X.shape[1])
        # Initialize the first rule
        self.Initialize_First_Cluster(X[0,], y[0])
        
        for k in range(X.shape[0]):
            
            xk = np.insert(X[k,], 0, 1, axis=0).reshape(1,X.shape[1]+1)
            
            # Compute the compatibility measure and the arousal index for all rules
            Output = 0
            sumCompatibility = 0
            for i in self.parameters.index:
                self.Compatibility_Measure(X[k,], i)
                # Local output
                self.parameters.at[i, 'LocalOutput'] = xk @ self.parameters.loc[i, 'Gamma']
                Output = Output + self.parameters.at[i, 'LocalOutput'] * self.parameters.loc[i, 'CompatibilityMeasure']
                sumCompatibility = sumCompatibility + self.parameters.loc[i, 'CompatibilityMeasure']
            # Global output
            if sumCompatibility == 0:
                Output = 0
            else:
                Output = Output/sumCompatibility
                
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            # Residual
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
            center = -1
            compatibility = -1
            for i in self.parameters.index:
                chistat = self.M_Distance(X[k,].reshape(1, X.shape[1]), i)
                if self.parameters.loc[i, 'ArousalIndex'] < self.Ta and chistat < self.Tp:
                    if self.parameters.loc[i, 'CompatibilityMeasure'] > compatibility:
                        compatibility = self.parameters.loc[i, 'CompatibilityMeasure']
                        center = i
            if center == -1:
                self.Initialize_Cluster(X[k,], y[k])
                center = self.parameters.last_valid_index()               
            else:
                self.Rule_Update(X[k,], y[k], center)
            for i in self.parameters.index:
                self.Update_Consequent_Parameters(xk, y[k], i)
            if self.parameters.shape[0] > 1:
                self.Merging_Rules(X[k,], center)
            self.rules.append(self.parameters.shape[0])
    
    def evolve(self, X, y):
        
        # Be sure that X is with a correct shape
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'center'].shape[1])
        
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
            
            xk = np.insert(X[k,], 0, 1, axis=0).reshape(1,X.shape[1]+1)
            
            # Compute the compatibility measure and the arousal index for all rules
            Output = 0
            sumCompatibility = 0
            for i in self.parameters.index:
                self.Compatibility_Measure(X[k,], i)
                # Local output
                self.parameters.at[i, 'LocalOutput'] = xk @ self.parameters.loc[i, 'Gamma']
                Output = Output + self.parameters.at[i, 'LocalOutput'] * self.parameters.loc[i, 'CompatibilityMeasure']
                sumCompatibility = sumCompatibility + self.parameters.loc[i, 'CompatibilityMeasure']
            # Global output
            if sumCompatibility == 0:
                Output = 0
            else:
                Output = Output/sumCompatibility
                
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            # Residual
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
            center = -1
            compatibility = -1
            for i in self.parameters.index:
                chistat = self.M_Distance(X[k,].reshape(1, X.shape[1]), i)
                if self.parameters.loc[i, 'ArousalIndex'] < self.Ta and chistat < self.Tp:
                    if self.parameters.loc[i, 'CompatibilityMeasure'] > compatibility:
                        compatibility = self.parameters.loc[i, 'CompatibilityMeasure']
                        center = i
            if center == -1:
                self.Initialize_Cluster(X[k,], y[k])
                center = self.parameters.last_valid_index()               
            else:
                self.Rule_Update(X[k,], y[k], center)
            for i in self.parameters.index:
                self.Update_Consequent_Parameters(xk, y[k], i)
            if self.parameters.shape[0] > 1:
                self.Merging_Rules(X[k,], center)
            self.rules.append(self.parameters.shape[0])
            
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
            
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'center'].shape[1])
        
        for k in range(X.shape[0]):
            
            xk = np.insert(X[k,], 0, 1, axis=0).reshape(1,X.shape[1]+1)
            
            # Compute the compatibility measure and the arousal index for all rules
            Output = 0
            sumCompatibility = 0
            for i in self.parameters.index:
                self.Compatibility_Measure(X[k,], i)
                # Local output
                self.parameters.at[i, 'LocalOutput'] = xk @ self.parameters.loc[i, 'Gamma']
                Output = Output + self.parameters.at[i, 'LocalOutput'] * self.parameters.loc[i, 'CompatibilityMeasure']
                sumCompatibility = sumCompatibility + self.parameters.loc[i, 'CompatibilityMeasure']
            # Global output
            if sumCompatibility == 0:
                Output = 0
            else:
                Output = Output/sumCompatibility
                
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
        return self.OutputTestPhase[-X.shape[0]:]
    
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
        
    def Initialize_First_Cluster(self, x, y):
        x = x.reshape(1, x.shape[0])
        Q = self.omega * np.eye(x.shape[1] + 1)
        Gamma = np.insert(np.zeros((x.shape[1],1)), 0, y, axis=0)
        self.parameters = pd.DataFrame([[x, 0., 1., 1., self.Sigma_init, np.array([]), Gamma, Q, 0.]], columns = ['center', 'ArousalIndex', 'CompatibilityMeasure', 'NumObservations', 'Sigma', 'o', 'Gamma', 'Q', 'LocalOutput'])
    
    def Initialize_Cluster(self, x, y):
        x = x.reshape(1, x.shape[0])
        Q = self.omega * np.eye(x.shape[1] + 1)
        Gamma = np.insert(np.zeros((x.shape[1],1)), 0, y, axis=0)
        NewRow = pd.DataFrame([[x, 0., 1., 1., self.Sigma_init, np.array([]), Gamma, Q, 0.]], columns = ['center', 'ArousalIndex', 'CompatibilityMeasure', 'NumObservations', 'Sigma', 'o', 'Gamma', 'Q', 'LocalOutput'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)
    
    def M_Distance(self, x, i):
        dist = self.mahalanobis(x, self.parameters.loc[i, 'center'], np.linalg.inv(self.parameters.loc[i, 'Sigma']))
        return dist
       
    def Compatibility_Measure(self, x, i):
        x = x.reshape(1, x.shape[0])
        dist = self.M_Distance(x, i)
        compat = math.exp(-0.5 * dist)
        self.parameters.at[i, 'CompatibilityMeasure'] = compat
        return compat
            
    def Arousal_Index(self, x, i):
        x = x.reshape(1, x.shape[0])
        chistat = self.M_Distance(x, i)
        self.parameters.at[i, 'o'] = np.append(self.parameters.loc[i, 'o'], 1) if chistat < self.Tp else np.append(self.parameters.loc[i, 'o'], 0)
        arousal = binom.cdf(sum(self.parameters.loc[i,'o'][-self.w:]), self.w, self.lambda1) if self.parameters.loc[i,'NumObservations'] > self.w else 0.
        self.parameters.at[i, 'ArousalIndex'] = arousal
        return arousal
    
    def Rule_Update(self, x, y, MaxIndexCompatibility):
        # Update the number of observations in the rule
        self.parameters.loc[MaxIndexCompatibility, 'NumObservations'] = self.parameters.loc[MaxIndexCompatibility, 'NumObservations'] + 1
        # Store the old cluster center
        # Oldcenter = self.parameters.loc[MaxIndexCompatibility, 'center']
        G = (self.alpha * (self.parameters.loc[MaxIndexCompatibility, 'CompatibilityMeasure'])**(1 - self.parameters.loc[MaxIndexCompatibility, 'ArousalIndex']))
        # Update the cluster center
        self.parameters.at[MaxIndexCompatibility, 'center'] = self.parameters.loc[MaxIndexCompatibility, 'center'] + G * (x - self.parameters.loc[MaxIndexCompatibility, 'center'])
        # Updating the dispersion matrix
        self.parameters.at[MaxIndexCompatibility, 'Sigma'] = (1 - G) * (self.parameters.loc[MaxIndexCompatibility, 'Sigma'] - G * (x - self.parameters.loc[MaxIndexCompatibility, 'center']) @ (x - self.parameters.loc[MaxIndexCompatibility, 'center']).T)
        
    def Membership_Function(self, x, i):
        dist = self.mahalanobis(x, self.parameters.loc[i, 'center'], np.linalg.inv(self.parameters.loc[i, 'Sigma']))
        return math.sqrt(dist)
        
    def Update_Consequent_Parameters(self, xk, y, i):
        self.parameters.at[i, 'Q'] = self.parameters.loc[i, 'Q'] - ((self.parameters.loc[i, 'CompatibilityMeasure'] * self.parameters.loc[i, 'Q'] @ xk.T @ xk @ self.parameters.loc[i, 'Q']) / (1 + self.parameters.loc[i, 'CompatibilityMeasure'] * xk @ self.parameters.loc[i, 'Q'] @ xk.T))
        self.parameters.at[i, 'Gamma'] = self.parameters.loc[i, 'Gamma'] + self.parameters.loc[i, 'Q'] @ xk.T * self.parameters.loc[i, 'CompatibilityMeasure'] * (y - xk @ self.parameters.loc[i, 'Gamma'])
                        
    def Merging_Rules(self, x, MaxIndexCompatibility):
        for i in self.parameters.index:
            if MaxIndexCompatibility != i:
                dist1 = self.M_Distance(self.parameters.loc[MaxIndexCompatibility, 'center'], i)
                dist2 = self.M_Distance(self.parameters.loc[i, 'center'], MaxIndexCompatibility)
                if dist1 < self.Tp or dist2 < self.Tp:
                    self.parameters.at[MaxIndexCompatibility, 'center'] = np.mean(np.array([self.parameters.loc[i, 'center'], self.parameters.loc[MaxIndexCompatibility, 'center']]), axis=0)
                    self.parameters.at[MaxIndexCompatibility, 'Sigma'] = [self.Sigma_init]
                    self.parameters.at[MaxIndexCompatibility, 'Q'] = self.omega * np.eye(x.shape[0] + 1)
                    self.parameters.at[MaxIndexCompatibility, 'Gamma'] = (self.parameters.loc[MaxIndexCompatibility, 'Gamma'] * self.parameters.loc[MaxIndexCompatibility, 'CompatibilityMeasure'] + self.parameters.loc[i, 'Gamma'] * self.parameters.loc[i, 'CompatibilityMeasure']) / (self.parameters.loc[MaxIndexCompatibility, 'CompatibilityMeasure'] + self.parameters.loc[i, 'CompatibilityMeasure'])
                    self.parameters = self.parameters.drop(i)
                    # Stop creating new rules when the model exclude the first rule
                    self.ExcludedRule = 1



class ePL(base):
    def __init__(self, alpha = 0.001, beta = 0.1, lambda1 = 0.35, tau = None, s = 1000, r = 0.25):
        
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
        if not (r > 0):
            raise ValueError("r must be a positive float.")
            
        # Hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.tau = beta if tau is None else tau
        self.s = s
        self.r = r
        
        self.parameters = pd.DataFrame(columns = ['center', 'P', 'Gamma', 'ArousalIndex', 'CompatibilityMeasure', 'TimeCreation', 'NumObservations', 'mu'])
        # Monitoring if some rule was excluded
        self.ExcludedRule = 0
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
    
    def get_params(self, deep=True):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'lambda1': self.lambda1,
            'tau': self.tau,
            's': self.s,
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
        
        # Initialize the first rule
        self.Initialize_First_Cluster(x, y[0])
        # Update the consequent parameters of the fist rule
        self.RLS(x, y[0], xe)
        
        for k in range(1, X.shape[0]):
            
            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            
            # Compute the compatibility measure and the arousal index for all rules
            for i in self.parameters.index:
                self.Compatibility_Measure(x, i)
                self.Arousal_Index(i)
            
            # Find the minimum arousal index
            MinIndexArousal = self.parameters['ArousalIndex'].astype('float64').idxmin()
            # Find the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['CompatibilityMeasure'].astype('float64').idxmax()
            
            # Verifying the needing to creating a new rule
            if self.parameters.loc[MinIndexArousal, 'ArousalIndex'] > self.tau:
                self.Initialize_Cluster(x, y[k], k+1)
            else:
                self.Rule_Update(x, y[k], MaxIndexCompatibility)
            if self.parameters.shape[0] > 1:
                self.Similarity_Index()
                
            # Compute the number of rules at the current iteration
            self.rules.append(self.parameters.shape[0])
            # Update the consequent parameters of the fist rule
            self.RLS(x, y[k], xe)
            # Compute firing degree
            self.mu(x)
            
            # Compute the output
            mu = np.array(self.parameters['mu'])
            Gamma = np.stack(self.parameters['Gamma'].values, axis=2)  # Stack Gamma vectors into a 3D array
            weighted_outputs = mu * np.einsum('ij,jik->i', xe.T, Gamma)
            Output = np.sum(weighted_outputs) / np.sum(mu)
            
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
    
    def evolve(self, X, y):
        
        # Be sure that X is with a correct shape
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'center'].shape[0])
        
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
            
            # Compute the compatibility measure and the arousal index for all rules
            for i in self.parameters.index:
                self.Compatibility_Measure(x, i)
                self.Arousal_Index(i)
                
            # Find the minimum arousal index
            MinIndexArousal = self.parameters['ArousalIndex'].astype('float64').idxmin()
            # Find the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['CompatibilityMeasure'].astype('float64').idxmax()
            
            # Verifying the needing to creating a new rule
            if self.parameters.loc[MinIndexArousal, 'ArousalIndex'] > self.tau:
                self.Initialize_Cluster(x, y[k], k+1)
            else:
                self.Rule_Update(x, y[k], MaxIndexCompatibility)
            if self.parameters.shape[0] > 1:
                self.Similarity_Index()
                
            # Compute the number of rules at the current iteration
            self.rules.append(self.parameters.shape[0])
            # Update the consequent parameters of the fist rule
            self.RLS(x, y[k], xe)
            # Compute firing degree
            self.mu(x)
            
            # Compute the output
            mu = np.array(self.parameters['mu'])
            Gamma = np.stack(self.parameters['Gamma'].values, axis=2)  # Stack Gamma vectors into a 3D array
            weighted_outputs = mu * np.einsum('ij,jik->i', xe.T, Gamma)
            Output = np.sum(weighted_outputs) / np.sum(mu)
            
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
            
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
            
        X = X.reshape(-1,self.parameters.loc[self.parameters.index[0],'center'].shape[0])
        
        for k in range(X.shape[0]):
            
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T
            # Compute xe
            xe = np.insert(x.T, 0, 1, axis=1).T
            # Compute firing degree
            self.mu(x)
            
            # Compute the output
            mu = np.array(self.parameters['mu'])
            Gamma = np.stack(self.parameters['Gamma'].values, axis=2)  # Stack Gamma vectors into a 3D array
            weighted_outputs = mu * np.einsum('ij,jik->i', xe.T, Gamma)
            Output = np.sum(weighted_outputs) / np.sum(mu)
            
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
            
        return self.OutputTestPhase[-X.shape[0]:]
        
    def Initialize_First_Cluster(self, x, y):
        self.parameters = pd.DataFrame([[x, self.s * np.eye(x.shape[0] + 1), np.zeros((x.shape[0] + 1, 1)), 0., 1., 1., 1., 1.]], columns = ['center', 'P', 'Gamma', 'ArousalIndex', 'CompatibilityMeasure', 'TimeCreation', 'NumObservations', 'mu'])
        Output = y
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y)**2)
    
    def Initialize_Cluster(self, x, y, k):
        NewRow = pd.DataFrame([[x, self.s * np.eye(x.shape[0] + 1), np.zeros((x.shape[0] + 1, 1)), 0., 1., k, 1., 1.]], columns = ['center', 'P', 'Gamma', 'ArousalIndex', 'CompatibilityMeasure', 'TimeCreation', 'NumObservations', 'mu'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)

    def Compatibility_Measure(self, x, i):
        self.parameters.at[i, 'CompatibilityMeasure'] = (1 - (np.linalg.norm(x - self.parameters.loc[i, 'center']))/x.shape[0] )
            
    def Arousal_Index(self, i):
        self.parameters.at[i, 'ArousalIndex'] += self.beta*(1 - self.parameters.loc[i, 'CompatibilityMeasure'] - self.parameters.loc[i, 'ArousalIndex'])
    
    def mu(self, x):
        for row in self.parameters.index:
            self.parameters.at[row, 'mu'] = math.exp( - self.r * np.linalg.norm(self.parameters.loc[row, 'center'] - x ) )
           
    def Rule_Update(self, x, y, i):
        # Update the number of observations in the rule
        self.parameters.loc[i, 'NumObservations'] += 1
        # Update the cluster center
        self.parameters.at[i, 'center'] += (self.alpha*(self.parameters.loc[i, 'CompatibilityMeasure'])**(1 - self.alpha))*(x - self.parameters.loc[i, 'center'])
          
        
    def Similarity_Index(self):
        for i in range(self.parameters.shape[0] - 1):
            l = []
			#if i < len(self.clusters) - 1:
            for j in range(i + 1, self.parameters.shape[0]):
                vi, vj = self.parameters.iloc[i,0], self.parameters.iloc[j,0]
                compat_ij = (1 - ((np.linalg.norm(vi - vj))))
                if compat_ij >= self.lambda1:
                    self.parameters.at[self.parameters.index[j], 'center'] = ( (self.parameters.loc[self.parameters.index[i], 'center'] + self.parameters.loc[self.parameters.index[j], 'center']) / 2)
                    self.parameters.at[self.parameters.index[j], 'P'] = ( (self.parameters.loc[self.parameters.index[i], 'P'] + self.parameters.loc[self.parameters.index[j], 'P']) / 2)
                    self.parameters.at[self.parameters.index[j], 'Gamma'] = np.array((self.parameters.loc[self.parameters.index[i], 'Gamma'] + self.parameters.loc[self.parameters.index[j], 'Gamma']) / 2)
                    l.append(int(i))

        self.parameters.drop(index=self.parameters.index[l,], inplace=True)

    def RLS(self, x, y, xe):
        for row in self.parameters.index:
            self.parameters.at[row, 'P'] -= ((self.parameters.loc[row, 'P'] @ xe @ xe.T @ self.parameters.loc[row, 'P'])/(1 + xe.T @ self.parameters.loc[row, 'P'] @ xe))
            self.parameters.at[row, 'Gamma'] += (self.parameters.loc[row, 'P'] @ xe * (y - xe.T @ self.parameters.loc[row, 'Gamma']))


class exTS(base):
    def __init__(self, omega = 1000, mu = 1/3, epsilon = 0.01, rho = 1/2):
        
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
        self.parameters = pd.DataFrame(columns = ['Center_Z', 'Center_X', 'C', 'Theta', 'Potential', 'TimeCreation', 'NumPoints', 'mu', 'Tau', 'Lambda', 'r', 'sigma', 'increment_center_x'])
        self.InitialPotential = 1.
        self.DataPotential = 0.
        self.InitialTheta = 0.
        self.InitialPi = 0.
        self.Beta = 0.
        self.Sigma = 0.
        self.z_last = None
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
    
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
                    self.Rule_Update(x, z)
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
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
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
                    self.Rule_Update(x, z)
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
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
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
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
            
        return self.OutputTestPhase[-X.shape[0]:]
        
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
            'increment_center_x': np.zeros([x.shape[0], 1])
        }])
        
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, y)
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
        NewRow = pd.DataFrame([[z, x, self.omega * np.eye(x.shape[0] + 1), Theta, self.InitialPotential, k, 1., np.zeros([x.shape[0], 1]), 1., np.ones([x.shape[0], 1]), sigma, np.zeros([x.shape[0], 1])]], columns = ['Center_Z', 'Center_X', 'C', 'Theta', 'Potential', 'TimeCreation', 'NumPoints', 'mu', 'Tau', 'r', 'sigma', 'increment_center_x'])
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
        vartheta = np.sum(z ** 2)
        upsilon = np.sum(z * self.Beta)
        self.DataPotential = (k - 1) / ((k - 1) * (vartheta + 1) + self.Sigma - 2 * upsilon)

    def Minimum_Distance(self, z):
        distances = np.linalg.norm(self.parameters['Center_Z'].values - z, axis=1)
        return np.min(distances)
                           
    def Rule_Update(self, x, z):
        dist = []
        idx = []
        for row in self.parameters.index:
            dist.append(np.linalg.norm(self.parameters.loc[row, 'Center_Z'] - z))
            idx.append(row)
        index = idx[dist.index(min(dist))]
        
        self.parameters.at[index, 'NumPoints'] += 1
        # Efficiently update increment_center_x and sigma
        diff_center_x = self.parameters.at[index, 'Center_X'] - x
        self.parameters.at[index, 'increment_center_x'] += diff_center_x ** 2
        self.parameters.at[index, 'sigma'] = np.sqrt(self.parameters.loc[index, 'increment_center_x'] / self.parameters.loc[index, 'NumPoints'])
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
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
    
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
                self.Rule_Update(x, z)
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
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
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
                self.Rule_Update(x, z)
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
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
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
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
        return self.OutputTestPhase[-X.shape[0]:]
        
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
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, y)
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
        self.Gamma = self.Sigma + sum(z_prev**2)
        self.DataScatter = (1 / ((k - 1) * (z.shape[0]))) * ((k - 1) * sum(z**2) - 2 * sum(z * self.Beta) + self.Gamma)
        
    def Minimum_Distance(self, z):
        distances = self.parameters['Center_Z'].apply(lambda center: np.linalg.norm(center - z))
        return distances.min()
                              
    def Rule_Update(self, x, z):
        distances = self.parameters['Center_Z'].apply(lambda center: np.linalg.norm(center - z))
        index = distances.idxmin()
        self.parameters.at[index, 'NumPoints'] += 1
        self.parameters.at[index, 'Center_Z'] = z
        self.parameters.at[index, 'Center_X'] = x
            
    def Update_Num_Points(self, z):
        distances = self.parameters['Center_Z'].apply(lambda center: np.linalg.norm(center - z))
        index = distances.idxmin()
        self.parameters.at[index, 'NumPoints'] += 1
        
    def Update_Lambda(self, x):
        self.parameters['Tau'] = self.parameters['Center_X'].apply(lambda center: self.mu(center, x))
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
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
    
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
                self.Rule_Update(x, z)  # Update an existing rule
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
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
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
                self.Rule_Update(x, z)  # Update an existing rule
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
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
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
            
        # Reshape X to match the dimension of the cluster centers
        expected_shape = self.parameters.loc[self.parameters.index[0], 'Center_X'].shape[0]
        if X.shape[1] != expected_shape:
            X = X.reshape(-1, expected_shape)
        
        # Preallocate output array for efficiency
        OutputTestPhase = np.zeros(X.shape[0])

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
            OutputTestPhase[k] = Output
        
        # Update the class variable and return the recent outputs
        self.OutputTestPhase = np.append(self.OutputTestPhase, OutputTestPhase)
            
        return OutputTestPhase
        
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
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, y)
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
        vartheta = sum(z**2)
        upsilon = sum(z*self.Beta)
        self.DataPotential = ((self.k - 1)) / ((self.k - 1) * (vartheta + 1) + self.Sigma - 2*upsilon)
        
    def Minimum_Distance(self, z):
        distances = np.linalg.norm(np.stack(self.parameters['Center_Z']) - z, axis=1)
        return np.min(distances)
                           
    def Rule_Update(self, x, z):
        distances = np.linalg.norm(np.stack(self.parameters['Center_Z']) - z, axis=1)
        index = np.argmin(distances)
        self.parameters.at[index, 'NumPoints'] += 1
        self.parameters.at[index, 'Center_Z'] = z
        self.parameters.at[index, 'Center_X'] = x
        self.parameters.at[index, 'Potential'] = self.DataPotential
            
    def Update_Lambda(self, x):
        self.parameters['Tau'] = self.parameters['Center_X'].apply(
            lambda center_x: self.mu(center_x, x)
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