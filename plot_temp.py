from utils.custom_plot import draw_single_subplot, draw_single_subplot_for_prop
import numpy as np
import pandas as pd
from pandas import read_csv
from utils.utils import load_json

p = load_json('./params.json')

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

df_o = read_csv(p.test_data_dir, index_col=None)
df_v = read_csv("./results/predicted_vanilla.csv", index_col=None)
df_p_1 = read_csv("./results/dklgp/predicted.csv", index_col=None)
df_p_2 = read_csv("./results/sn_dklgp/predicted.csv", index_col=None)
df_p_3 = read_csv("./results/dklgp_svd/predicted.csv", index_col=None)
df_p_4 = read_csv("./results/sn_dklgp_svd/predicted.csv", index_col=None)

df_origin  = df_o['sw_p1'].values
df_vanilla   = np.abs(df_v['y0_pred_v'].values - df_origin)
df_prop_1    = np.abs(df_p_1['y0_pred'].values - df_origin)
df_prop_2    = np.abs(df_p_2['y0_pred'].values - df_origin)
df_prop_3    = np.abs(df_p_3['y0_pred'].values - df_origin)
df_prop_4    = np.abs(df_p_4['y0_pred'].values - df_origin)

x_list = []
for i in range(len(df_origin)):
    x_list.append(i)

# print(df_origin)

# Vo, I_L, Vin = split_data(data)
# plot_custom(t_list[125], Vo, I_L, Vin, Vo_n, I_L_n, Vin_n, Vo_g, I_L_g, Vin_g, t_list[570], s_Vo, s_I_L, s_Vin,s_Vo_n, s_I_L_n, s_Vin_n, s_Vo_g, s_I_L_g, s_Vin_g)
draw_single_subplot_for_prop(x_list, df_vanilla, df_prop_1,df_prop_2,df_prop_4,df_prop_4, "Error_comparison")

# draw_fault_subplot(time_list[240:270], fault_list[240:270], pred[240:270], "Predicted results")
