import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def dist_for_dataset(dataset, print_flag=False):
    '''
    dataset: ndarray 2D (N,dim)
    return: ndarray 1D (NC2) combination function
    '''
    N_DATA = len(dataset)
    dis_list = []
    for i in range(N_DATA):
        for j in range(i+1,N_DATA):
            # print(f"(i, j) = ({i, j})")
            d = euclidean_distance(dataset[i], dataset[j])
            dis_list.append(d)

    if print_flag == True:
        print(np.array(dis_list))

    return np.array(dis_list)

def plot_comp_dist(true_list, edited_list):
    MAX_VALUE = max(max(true_list),max(edited_list))
    
    plt.plot(true_list, edited_list, 'bo', [0,MAX_VALUE],[0,MAX_VALUE],'r') # bo is distance and r is the ref line
    plt.xlabel('Distance of Actual')
    plt.ylabel('Distance of Extracted Data')
    plt.show()

def extract_ith_data(dataset, ith):
    '''
    When dataset is 2D, this function extract ith column.
    input: (N,f)
    output: (N,1)
    '''
    return dataset.T[ith].T.reshape(len(dataset),1)

def comp_X_Z_plot(X_data, all_Z, feature_num):
    '''
    X_data: (N, i_d)
    all_Z: (o_d, N, h_d)
    '''
    for i in range(feature_num):
        # X_data_i = extract_ith_data(X_data, i)

        actual_dist_list = dist_for_dataset(all_Z[i])
        edited_dist_list = dist_for_dataset(X_data)

        print(len(actual_dist_list))
        print(len(edited_dist_list))

        plot_comp_dist(actual_dist_list, edited_dist_list)

if __name__ == "__main__":
    dataset = []
    for i in range(32):
        dataset.append([i*0.1+0.4,i*0.2+0.7])
    dataset = np.array(dataset)

    dataset_2 = []
    for i in range(32):
        dataset_2.append([i*1.15+1.4,i*0.15+0.7])
    dataset_2 = np.array(dataset_2)

    dist_list = dist_for_dataset(dataset=dataset, print_flag=False)
    dist_list2 = dist_for_dataset(dataset=dataset_2, print_flag=False)

    plot_comp_dist(dist_list, dist_list2)