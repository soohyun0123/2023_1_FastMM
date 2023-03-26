import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

def make_train_data(filename):
    f = open(filename, 'r')
    data_list = []
    for line in f :
        cur = line.strip().split(', ')
        try:
            x1 = int(cur[0])    # mat size
            x0 = int(cur[1])    # block size
            y = float(cur[2])   # latency
            data_list.append([x0, x1, y])
        except:
            pass
    df = pd.DataFrame(data_list, columns=['x0(block size)', 'x1(mat size)', 'y(latency)'])
    df.to_csv('./data/exp3-1/cost_model_dataframe_0319.csv')

def preprocess_train_data(filter, std, fig_show):
    def _get_dataframe(filename):
        f = open(filename, 'r')
        l = []
        col = []
        for line in f:
            cur = line.strip().split(', ')
            try:
                cur[0] = int(cur[0])
                cur[1] = float(cur[1])
                l.append(cur)
            except:
                col = cur
        f.close()
        return pd.DataFrame(l, columns=col)
    def _plot_with_block_size(df, mat_size, fig_save):
        x = df['Block size']
        y = df['Time elapsed[sec]']
        plt.plot(x, y)
        plt.title('MM latency with block size (Mat. size=' + str(mat_size) + ')')
        plt.xlabel('Block size')
        plt.ylabel('Latency [sec]')
        if fig_save:
            plt.savefig('./graph/[exp3-3]training_data_original_' + str(mat_size) + '.png')
        else:
            plt.show()

    file_list = os.listdir('./data/exp3-3')
    is_first = True
    for filename in file_list:
        if not filename.startswith('cost_model_data_mat_size') :
            continue
        df = _get_dataframe('./data/exp3-3/' + filename)
        mat_size = int(filename.strip('.txt').split('_')[-1])
        #_plot_with_block_size(df, mat_size, fig_save=1)
        x, y_origin = np.array(df['Block size']), np.array(df['Time elapsed[sec]'])
        if filter:
            y = np.hstack((y_origin[:4], gaussian_filter1d(y_origin[4:], std)))
        else:
            y = y_origin
        m = np.ones(len(x)) * mat_size
        x, y, m = x.reshape(-1, 1), y.reshape(-1, 1), m.reshape(-1, 1)
        cur = np.hstack((m, x, y))
        if fig_show:
            plt.figure()
            plt.plot(x, y_origin, label='original data', color='blue')
            plt.plot(x, y, label='filtered data', color='skyblue')
            plt.title('Mat size ' + str(mat_size))
            plt.legend()
            plt.show()
        if is_first:
            training_data_arr = cur
            is_first = False
        else:
            training_data_arr = np.vstack((training_data_arr, cur))
    df = pd.DataFrame(training_data_arr, columns=['x1(mat size)', 'x0(block size)', 'y(latency)'])
    df.to_csv('./data/exp3-3/cost_model_training_data_0326.csv')


def curve_simple(X, p0, p1) :
    """
        X[0]: block size, mc
        X[1]: matrix size, M
    """
    flops_task = np.multiply(np.power(X[0], 2), X[1])
    data_task = 2 * np.multiply(X[0], X[1]) + np.power(X[0], 2)
    cost_seq = p0 * flops_task + p1 * data_task
    cost_comp = (1/8) * np.multiply(np.power(X[1], 2), np.power(1/X[0], 2)) * cost_seq
    cost_total = cost_comp
    return cost_total

def curve_ver2(X, p0, p1, p2, p3, p4) :
    """
            X[0]: block size, mc
            X[1]: matrix size, M
        """
    flops_task = np.multiply(np.power(X[0], 2), X[1])
    data_task = 2 * np.multiply(X[0], X[1]) + np.power(X[0], 2)
    cost_seq = p0 * flops_task + p1 * data_task
    cost_comp = (1 / 8) * np.multiply(np.power(X[1], 2), np.power(1 / X[0], 2)) * cost_seq
    cost_balance = p2 * cost_seq
    cost_sched = p3 * np.multiply(np.power(X[1], 2), np.power(1 / X[0], 2))
    cost_total = cost_comp + cost_balance + cost_sched + p4
    return cost_total

def train_cost_model(input_filename, output_filename, is_save, ver):
    df = pd.read_csv(input_filename, index_col=0)
    #print(df[:10])
    ydata = df['y(latency)']
    ydata = np.array(ydata)     # shape: (467, )
    xdata = df[['x0(block size)', 'x1(mat size)']]
    xdata = np.array(xdata)     # shape: (467, 2)
    xdata = xdata.transpose()
    xdata = xdata.astype('float64')
    #print(curve_simple(xdata, 0.1, 0.2)[:10])
    if ver == 1 :
        popt, pcov = curve_fit(curve_simple, xdata, ydata)
    elif ver == 2:
        popt, pcov = curve_fit(curve_ver2, xdata, ydata)
    print('Coefficient: ', popt)
    if is_save:
        f = open(output_filename, 'w')
        for i in range(popt.shape[0]) :
            f.write(str(popt[i]) + ' ')
        f.close()
    return popt

def load_cost_model(filename):
    f = open(filename, 'r')
    param = f.readline().strip().split(' ')
    f.close()
    for i in range(len(param)) :
        param[i] = float(param[i])
    return param

def test_cost_model(filename, param, ver):
    df = pd.read_csv(filename, index_col=0)
    ydata = df['y(latency)']
    ydata = np.array(ydata)  # shape: (467, )
    xdata = df[['x0(block size)', 'x1(mat size)']]
    xdata = np.array(xdata)  # shape: (467, 2)
    xdata = xdata.transpose()
    xdata = xdata.astype('float64')
    if ver == 1:
        ypred = curve_simple(xdata, param[0], param[1])
    elif ver == 2:
        ypred = curve_ver2(xdata, param[0], param[1], param[2], param[3], param[4])

    x_axis = [0, 25]
    #plt.plot(x_axis, ydata)
    #plt.plot(x_axis, ypred)
    plt.scatter(ydata, ypred)
    plt.plot(x_axis, x_axis)
    plt.show()

def predict_with_mat_size(param, block, filename_output, is_save, ver) :
    def _get_dataframe(filename) :
        f = open(filename, 'r')
        l = []
        col = []
        for line in f:
            cur = line.strip().split(', ')
            try:
                cur[0] = int(cur[0])
                cur[1] = float(cur[1])
                l.append(cur)
            except:
                col = cur
        f.close()
        return pd.DataFrame(l, columns=col)
    ############## Generate test data
    X_test = []
    for M in range(1, 1601) :
        if M <= block:
            mc = M
        else :
            mc = block
        X_test.append([mc, M])
    X_test = np.array(X_test)
    X_test = np.transpose(X_test)
    if ver == 1:
        y_pred = curve_simple(X_test, param[0], param[1])
    elif ver == 2:
        y_pred = curve_ver2(X_test, param[0], param[1], param[2], param[3], param[4])
    ################# Load experiment data
    df = _get_dataframe('./data/exp1/multithread_MM_0318_1600.txt')

    ################ Figure
    plt.figure()
    plt.plot(df['Matrix size'], df['Time elapsed[sec]'], 'c', label='Experiment data')
    plt.plot(X_test[1], y_pred, 'b', label='Predicted data', linestyle='dotted')
    plt.legend()
    plt.title('MM latency with different matrix size')
    plt.xlabel('Matrix size')
    plt.ylabel('Latency [sec]')
    if is_save:
        plt.savefig(filename_output)
    else:
        plt.show()

def predict_with_block_size(param, mat_size, block_max, filename_input, filename_output, is_save, ver) :
    def _get_dataframe(filename) :
        f = open(filename, 'r')
        l = []
        col = []
        for line in f:
            cur = line.strip().split(', ')
            try:
                cur[0] = int(cur[0])
                cur[1] = float(cur[1])
                l.append(cur)
            except:
                col = cur
        f.close()
        return pd.DataFrame(l, columns=col)

    ############## Generate test data
    X_test = []
    for mc in range(4, block_max + 1):
        X_test.append([mc, mat_size])
    X_test = np.array(X_test)
    X_test = np.transpose(X_test)
    if ver == 1:
        y_pred = curve_simple(X_test, param[0], param[1])
    elif ver == 2:
        y_pred = curve_ver2(X_test, param[0], param[1], param[2], param[3], param[4])

    ################# Load experiment data
    df = _get_dataframe(filename_input)

    ################ Figure
    plt.figure()
    plt.plot(df['Block size'], df['Time elapsed[sec]'], 'c', label='Experiment data')
    plt.plot(X_test[0], y_pred, 'b', label='Predicted data', linestyle='dotted')
    plt.legend()
    plt.title('MM latency with different block size (Total size=' + str(mat_size) +')')
    plt.xlabel('Block size')
    plt.ylabel('Latency [sec]')
    if is_save:
        plt.savefig(filename_output)
    else:
        plt.show()

def predict_with_block_size_multiple(param, mat_size, block_max, filename_base, iter, is_save, ver) :
    def _get_dataframe(filename) :
        f = open(filename, 'r')
        l = []
        col = []
        for line in f:
            cur = line.strip().split(', ')
            try:
                cur[0] = int(cur[0])
                cur[1] = float(cur[1])
                l.append(cur)
            except:
                col = cur
        f.close()
        return pd.DataFrame(l, columns=col)

    ############## Generate test data
    X_test = []
    for mc in range(1, block_max + 1):
        X_test.append([mc, mat_size])
    X_test = np.array(X_test)
    X_test = np.transpose(X_test)
    if ver == 1:
        y_pred = curve_simple(X_test, param[0], param[1])
    elif ver == 2:
        y_pred = curve_ver2(X_test, param[0], param[1], param[2], param[3], param[4])

    color_list = ['blue', 'cyan', 'skyblue', 'lime', 'green']
    plt.figure()
    ################# Load experiment data
    for i in range(iter):
        filename = filename_base + str(i + 1) + '.txt'
        color = color_list[i]
        df = _get_dataframe(filename)
        plt.plot(df['Block size'], df['Time elapsed[sec]'], color=color, label=str(i + 1) + 'th')

    plt.plot(X_test[0], y_pred, color='gray', label='Predicted data', linestyle='dotted')

    plt.title('MM latency with different block size (Total size=' + str(mat_size) + ')')
    plt.xlabel('Block size')
    plt.ylabel('Latency [sec]')
    plt.legend()

    if is_save:
        plt.savefig('./graph/[exp3-2] Block_size_' + str(mat_size) + 'ver_' + str(ver) +'.png')
    else:
        plt.show()

#make_train_data('./data/exp3-1/cost_model_data_0319.txt')
#param = train_cost_model('./data/exp3-1/cost_model_dataframe_0319.csv', is_save=1, ver=2)
#param = load_cost_model('./data/exp3-1/cost_model_param_ver2_0322.txt')
#print(param)
#test_cost_model('./data/exp3-1/cost_model_dataframe_0319.csv', param, ver=2)
#predict_with_mat_size(param, 32, is_save=1, ver=2)
#predict_with_block_size(param, 512, 250, './data/exp2-2/latency_with_block_size_512_0321_2.txt', is_save=1, ver=2)
#predict_with_block_size(param, 1024, 200, './data/exp2-2/latency_with_block_size_1024_0318.txt', is_save=1, ver=2)
#predict_with_block_size(param, 2048, 25, './data/exp2-2/latency_with_block_size_2048_0318.txt', is_save=1, ver=2)
#predict_with_block_size_multiple(param, 512, 300, './data/exp3-2/[exp3-2]Mat_size_512_', iter=5, is_save=0, ver=2)

##################################
## Exp 3-3
##################################
#preprocess_train_data(filter=1, std=1, fig_show=False)
#train_cost_model('./data/exp3-3/cost_model_training_data_0326.csv', './data/exp3-3/cost_model_parameter_0326.txt', is_save=1, ver=2)
param = load_cost_model('./data/exp3-3/cost_model_parameter_0326.txt')
#test_cost_model('./data/exp3-3/cost_model_training_data_0326.csv', param, ver=2)
#predict_with_mat_size(param, 32, filename_output='./graph/[exp3-3] latency_prediction_mat_size_std=1.png', is_save=1, ver=2)
predict_with_block_size(param, 512, 300, './data/exp3-2/[exp3-2]Mat_size_512_1.txt',
                        filename_output='./graph/[exp3-3] latency_prediction_512_std=1.png', is_save=1, ver=2)
predict_with_block_size(param, 1024, 200, './data/exp3-2/[exp3-2]Mat_size_1024_1.txt',
                        filename_output='./graph/[exp3-3] latency_prediction_1024_std=1.png', is_save=1, ver=2)
predict_with_block_size(param, 2048, 40, './data/exp3-3/cost_model_data_mat_size_2048.txt',
                        filename_output='./graph/[exp3-3] latency_prediction_2048_std=1.png', is_save=1, ver=2)