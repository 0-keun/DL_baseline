from custom_plot import draw_single_subplot, draw_fault_subplot
import numpy as np
import pandas as pd
from pandas import read_csv

def split_data(data):
    data1 = []
    data2 = []
    data0 = []
    for row in data:
        data0.append(row[0])
        data1.append(row[1])
        data2.append(row[2])
    return data0, data1, data2

feature_name  = ['Vo:Measured voltage','IL:Measured current','Vin:Measured voltage']
input_file    = "./dataset/dataset_test_added/data_f_3_t_1.102649.csv"

df = read_csv(input_file, index_col=None)
data = df[feature_name].values
time_list = df['Time / s'].values
fault_list = df['SHORT'].values
pred = np.insert(fault_list[:-1], 0, 0)

Vo, I_L, Vin = split_data(data)
# plot_custom(t_list[125], Vo, I_L, Vin, Vo_n, I_L_n, Vin_n, Vo_g, I_L_g, Vin_g, t_list[570], s_Vo, s_I_L, s_Vin,s_Vo_n, s_I_L_n, s_Vin_n, s_Vo_g, s_I_L_g, s_Vin_g)
draw_single_subplot(time_list, Vo, I_L, Vin, "Data")
draw_fault_subplot(time_list[240:270], fault_list[240:270], pred[240:270], "Predicted results")