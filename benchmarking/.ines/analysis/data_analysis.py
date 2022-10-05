import matplotlib.pyplot as plt 
import csv 
import numpy as np 
import pandas as pd 
from scipy.optimize import curve_fit
import csv 

def func(x, a, b): 
    '''
    Defining the function to which the data is fitted to. In this case linear regresion. 
    '''
    return a * x + b

def inf_vs_inputsize():
    '''
    In this function, the average inference time (over different parameters) is analyzed according to the input size.
    In deed, it is fitted to a linear regression for each operation and for the different presented Hardware. 
    Correlations are also calculated here to support this linear regression model. 
    '''
    data = pd.read_csv("ressources/results_reduced.csv")
    pivot = pd.pivot_table(data, values='inference_time(us)', index =['op_name' , 'input_type', 'num_threads', 'input_size'], columns= ['Hardware'], margins=False, aggfunc=np.mean )
    locations=pivot.index.get_level_values(0).unique()
    columns = ['Operation name', 'num_threads', 'Type', 'Hardware', 'Correlation between Measurement data and regression cost model' ]
    correlation_table = pd.DataFrame(columns=columns)
    color = ['Black']
    
    for column in pivot.columns:
        for location in locations:  
            machin1 = pivot.loc[location][column]
            bidules1 = machin1.index.get_level_values(0).unique()
            for location_1 in bidules1: 
                machin2 = machin1.loc[location_1]
                bidules2 = machin2.index.get_level_values(0).unique()
                colorindex=0
                for location_2 in bidules2: 
                    if column != "All":
                        machin = machin2.loc[location_2]
                        bidules = machin.index.get_level_values(0).unique()
                    test = bidules.values*3
                    try:
                        
                        popt, pcov = curve_fit(func, bidules.values, machin[bidules].values)
                        
                        if location== "convx":
                            location = 'Conv2D'

                        plt.plot(bidules.values, machin[bidules].values, '+', label ='Inference Time in \u03BCs in function of the number of elements of the input: Measurements Data with \n Hardware= ' + column + ', Operation name= ' + location + ', Input type= ' + location_1 + ', Number of shaves= ' + str(location_2) )
                        plt.plot(bidules.values, func(bidules.values, *popt), '--', label='fit: y=%5.5f x + %5.5f' %tuple(popt) )

                        plt.grid(True)
                        plt.ylabel('Inference time (\u03BCs)')
                        plt.xlabel('Input size')
                        plt.legend(fontsize=7)

                        print("measurements: ", machin[bidules].values )
                        print("lineat regression: ", func(bidules.values, *popt))
                        print("correlation", np.corrcoef(np.array((machin[bidules].values, func(bidules.values, *popt))))[0,1] )
                        corr_calc  = np.corrcoef(np.array((machin[bidules].values, func(bidules.values, *popt))))[0,1]
                        corr_df = pd.DataFrame({'Operation name':[location], 'num_threads':[location_2], 'Type':[location_1], 'Hardware':[column], 'Correlation between Measurement data and regression cost model':[corr_calc]})
                        correlation_table = correlation_table.append(corr_df, ignore_index=True)
                        with open('costs_file.csv', 'a') as csvfile: 
                                    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                                    filewriter.writerow([location, column, location_1, location_2, tuple(popt)[0], tuple(popt)[1]])

                    except ValueError as e: 
                        continue
            plt.show()
    
    correlation_table.to_csv("correlations2.csv")

inf_vs_inputsize()
