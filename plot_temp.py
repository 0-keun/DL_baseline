from custom_plot import draw_single_subplot, draw_fault_subplot
import numpy as np
import pandas as pd
from pandas import read_csv

# def split_data(data):
#     data1 = []
#     data2 = []
#     data0 = []
#     for row in data:
#         data0.append(row[0])
#         data1.append(row[1])
#         data2.append(row[2])
#     return data0, data1, data2

# feature_name  = ['y0_pred_v']
# input_file    = "./dataset/dataset_test_added/data_f_0_t_1.100444.csv"

df_o = read_csv("./dataset/dataset_OOD_single_load_400.csv", index_col=None)
df_v = read_csv("./predicted_vanilla.csv", index_col=None)
df_p = read_csv("./predicted.csv", index_col=None)

df_origin  = df_o['sw_p1'].values
df_vanilla = df_v['y0_pred_v'].values
df_prop    = df_p['y0_pred'].values

x_list = []
for i in range(len(df_origin)):
    x_list.append(i)

# print(df_origin)

# Vo, I_L, Vin = split_data(data)
# plot_custom(t_list[125], Vo, I_L, Vin, Vo_n, I_L_n, Vin_n, Vo_g, I_L_g, Vin_g, t_list[570], s_Vo, s_I_L, s_Vin,s_Vo_n, s_I_L_n, s_Vin_n, s_Vo_g, s_I_L_g, s_Vin_g)
draw_single_subplot(x_list, df_origin, df_vanilla, df_prop, "Data")

# draw_fault_subplot(time_list[240:270], fault_list[240:270], pred[240:270], "Predicted results")