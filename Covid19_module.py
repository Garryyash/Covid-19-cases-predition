# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:19:50 2022

@author: caron
"""

import matplotlib.pyplot as plt

class EDA():
    def __init__(self):
        pass
        
    def plot_graph(self,df):
        plt.figure()
        plt.plot(df['cases_new'])
        plt.plot(df['cases_recovered'])
        plt.legend(['cases_new','cases_recovered'])
        plt.show

    
class model_evaluation():
    def plot_predicted_graph(self,test_df,predicted,mms):
        plt.figure()
        plt.plot(test_df,'b',label='actual cases')
        plt.plot(predicted,'r',label='predicted cases')
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.plot(mms.inverse_transform(test_df),'b',label='actual cases')
        plt.plot(mms.inverse_transform(predicted),'r',label='predicted cases')
        plt.legend()
        plt.show()