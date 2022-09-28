import matplotlib.pyplot as plt 
import csv 
import numpy as np 
import pandas as pd 
from scipy.optimize import curve_fit
import csv 
import seaborn as sns


def pivot_table_from_csv():
    #filepath = input("give filepath of test results: ")
    filepath = "ressources/outcsv_all_16_Sep_test.csv" #the fittings were applied to this CSV file
    
    data = pd.read_csv(filepath) 
    pivot = pd.pivot_table(data, values='inference_time(us)', index =['op_name' , 'input_type'], columns= ['Hardware'], margins=False, aggfunc=np.mean )
    locations=pivot.index.get_level_values(0).unique()
    
    for column in pivot.columns: 
        i = 1
        j = locations.size 
        
        for location in locations:
            if column != "All":    
                y = pivot.loc[location][column]
                x = y.index.get_level_values(0).unique()
                try:
                    fig = plt.figure()
                    fig.suptitle(' hardware: ' + column + ' op_name: ' + location)
                    plt.bar(x, y[x], label =' hardware: ' + column + ' op_name: ' + location )
                    i=i +1                 
                except ValueError as e:
                    print(column, location, e) 
                    continue   
    plt.legend()
    plt.show()

pivot_table_from_csv()