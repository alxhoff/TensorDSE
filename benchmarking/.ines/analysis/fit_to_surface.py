import matplotlib.pyplot as plt 
import csv 
import numpy as np 
import pandas as pd 
from scipy.optimize import curve_fit
import csv 
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D 
import scipy.interpolate


'''
In this script, I was trying to fit the data to a surface
However noticing that the coefficients were negative, I did not continue towards this direction.
May be this could be improved..
'''
 
def curvefit_func(M, a, b, c): 
    y = M[0]
    x = M[1]
    return (a * x + b)/(c*y)

def pivot_table_from_csv():
    filepath = "ressources/outcsv_all_16_Sep_test.csv"
    data = pd.read_csv(filepath)
    pivot = pd.pivot_table(data, values='inference_time(us)', index =['op_name', 'num_threads' , 'input_shape'], columns= ['Hardware'], margins=False, aggfunc=np.mean , fill_value=0)
    locations=pivot.index.get_level_values(0).unique()
    
    for column in pivot.columns: 
        for location in locations: 
            tableau = pivot.loc[location][column]
            # x = input shapes and y number of threads 
            x = tableau.index.get_level_values(1).unique()
            y = tableau.index.get_level_values(0).unique()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            levels = tableau.loc[y]
            xline =np.array([])
            yline =np.array([])
            zline =np.array([])
            for row in levels.iteritems(): 
                xline = np.append(xline, row[0][0])
                yline = np.append(yline, row[0][1])
                zline = np.append(zline, row[1])
            
            Y,X, Z = np.meshgrid(y, x, zline)
            #XX = X.flatten()
            #YY = Y.flatten()
            XX = X.ravel()
            YY = Y.ravel()
            ZZ = Z.flatten()
            M = np.vstack((YY, XX))
            #M = (YY, XX)
            try :
                popt2, pcov2 = curve_fit(curvefit_func, M , ZZ)
                ax.plot_trisurf(YY, XX, ZZ)
                plot_label = 'fit: a=' + str(popt2[0]) + ' b=' + str(popt2[1])+ ' c=' + str(popt2[2])
                ax.plot_trisurf(YY, XX, curvefit_func(M, *popt2))
                ax.set_title(label ='Hardware: ' + column + ' op_name: ' + location + " --" + plot_label)
            except(RuntimeError):
                ax.set_title(label ='hardware: ' + column + ' op_name: ' + location)
                ax.plot_trisurf(YY, XX, ZZ)
                
            plt.legend()
            plt.show()
            
pivot_table_from_csv()