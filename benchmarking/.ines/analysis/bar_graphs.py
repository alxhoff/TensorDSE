import matplotlib.pyplot as plt 
import csv 
import numpy as np 
import pandas as pd 
from scipy.optimize import curve_fit
import csv 
import seaborn as sns
import os 

'''
Different functions to plot bar graphs to analyze the data.
'''
    

def averageInference_vs_HW():
    '''
    Function that plots bar graphs of the average inference time over different parameters versus the HW.
    The HW here includes (Desktop(CPU+GPU), NCS, and the coral)
    '''
    #filepath = input("give filepath of test results: ")
    filepath = "ressources/outcsv_all_16_Sep_test.csv"
    data = pd.read_csv(filepath)
    pivot = pd.pivot_table(data, values='inference_time(us)', index='Hardware', columns= ['op_name'], margins=False, aggfunc=np.mean )
    locations=pivot.index.get_level_values(0).unique()
    #print(pivot)
    pivot.plot.bar()
    plt.xlabel('Hardware', fontsize=8, fontweight='bold')
    plt.ylabel('Average Execution time in \u03BCs', fontsize=8, fontweight='bold')
    plt.xticks(rotation=360, fontsize=8)
    plt.legend(fontsize=7)
    plt.show()

#averageInference_vs_HW()

def ncs_sep_shaves():
    '''
    Function to plot the bar graph of average inference time over different parameters versus the maximum number of SHAVEs pf the NCS.
    '''
    #filepath = 'ressources/out_isize_s_chan.csv'
    filepath = 'ressources/outcsv_all_16_Sep_test.csv'
    data = pd.read_csv(filepath)
    pivot = pd.pivot_table(data, values='inference_time(us)', index=['num_threads'], columns= ['Hardware'], margins=False, aggfunc=np.mean )
    locations=pivot.index.get_level_values(0).unique()
    fig, ax = plt.subplots()
    bars = ax.bar(locations, pivot['NCS2'])
    plt.xticks(locations)
    plt.xticks(rotation=360, fontsize=12)
    plt.xlabel('Number of SHAVEs', fontsize=15, fontweight='bold')
    plt.ylabel('Average Execution time in \u03BCs', fontsize=15, fontweight='bold')
    plt.yticks(fontsize=12)
    plt.title( 'Average inference time in \u03BCs on the Neural Compute Stick (NCS) for different SHAVEs', fontsize=16, fontweight='bold')
    for br in bars:
        ht = br.get_height()
        ax.annotate('{:6.1f}'.format(ht), xy= (br.get_x()+ 0.35, ht), ha='center', va='bottom', fontsize=12)
    plt.show()

#ncs_sep_shaves()

def ncs_sep_shaves_sep_ops():
    '''
    Function to plot the bar graph of average inference time over different parameters 
    versus the maximum number of SHAVEs pf the NCS for each of the tested operations seperately.
    '''
    filepath = 'ressources/outcsv_all_16_Sep_test.csv' 
    data = pd.read_csv(filepath)
    pivot = pd.pivot_table(data, values='inference_time(us)', index=['op_name','num_threads'], columns= ['Hardware'], margins=False, aggfunc=np.mean )
    locations=pivot.index.get_level_values(0).unique()
    for column in pivot.columns:
        for location in locations:
            y = pivot.loc[location][column]
            x = y.index.get_level_values(0).unique()
            if (column=="NCS2"):
                fig, ax = plt.subplots() 
                bars = ax.bar(x, y[x], width=0.5)
                plt.xticks(x)
                plt.xticks(rotation=360, fontsize=8)
                plt.xlabel('Number of SHAVEs', fontsize=8, fontweight='bold')
                plt.ylabel('Average Execution time in \u03BCs', fontsize=8, fontweight='bold')
                plt.title( 'Average inference time of operation model ' + location + ' on the NCS for different values of input shaves', fontsize=8, fontweight='bold')
                
                for br in bars:
                    ht = br.get_height()
                    ax.annotate('{:6.2f}'.format(ht), xy= (br.get_x()+ 0.25, ht), ha='center', va='bottom', fontsize=7)
                plt.show()

#ncs_sep_shaves_sep_ops()

