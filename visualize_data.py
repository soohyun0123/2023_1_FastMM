import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.ndimage import gaussian_filter1d

def plot_latency_with_size() :
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
    df1 = _get_dataframe('./data/exp1/naive_MM_0318.txt')
    df2 = _get_dataframe('./data/exp1/multithread_ver1_MM_0318.txt')
    df3 = _get_dataframe('./data/exp1/multithread_ver2_MM_0318.txt')
    t1 = df1[df1['Matrix size'] == 1024]['Time elapsed[sec]']
    t2 = df2[df2['Matrix size'] == 1024]['Time elapsed[sec]']
    t3 = df3[df3['Matrix size'] == 1024]['Time elapsed[sec]']
    print(t1, t2, t3)
    print(round(t1 / t2, 3))
    print(round(t3 / t2, 3))
    """
    plt.figure()
    plt.plot(df1['Matrix size'], df1['Time elapsed[sec]'], 'b', label='Naive MM')
    plt.plot(df3['Matrix size'], df3['Time elapsed[sec]'], 'g', label='Multithread without data copy')
    plt.plot(df2['Matrix size'], df2['Time elapsed[sec]'], 'c', label='Multithread with data copy')
    plt.legend()
    plt.title('MM latency with different matrix size')
    plt.xlabel('Matrix size')
    plt.ylabel('Latency [sec]')
    #plt.show()
    plt.savefig('./Latency_with_size_0318.png')"""

def plot_latency_with_size_multithread(is_save) :
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

    #df = _get_dataframe('./data/exp1/multithread_MM_0318_2500.txt')
    df = _get_dataframe('./data/exp1/multithread_ver1_MM_0318.txt')
    plt.figure()
    plt.plot(df['Matrix size'], df['Time elapsed[sec]'], 'c', label='Multithread with data copy')
    #plt.legend()
    plt.title('MM latency with different matrix size')
    plt.xlabel('Matrix size')
    plt.ylabel('Latency [sec]')
    if is_save:
        plt.savefig('./Latency_with_size_multithread_2.png')
    else :
        plt.show()

def plot_phase_time(filename, is_plot) :
    def _get_dataframe(filename) :
        f = open(filename, 'r')
        l = []
        col = []
        for line in f:
            cur = line.strip().split(' ')
            cur[0] = float(cur[0])
            cur[1] = float(cur[1])
            cur[2] = float(cur[2])
            l.append(cur)
        f.close()
        return pd.DataFrame(l, columns=['Phase1', 'Phase2', 'Phase3'])
    df = _get_dataframe(filename)
    x1 = df['Phase1']
    x2 = df['Phase2']
    x3 = df['Phase3']
    if is_plot:
        axis_max = max(x1.max(), x2.max())
        plt.figure()
        plt.scatter(x1, x2)
        plt.plot([0, axis_max], [0, axis_max])
        plt.title('Data copying vs Computation')
        plt.xlabel('Data copy latency [sec]')
        plt.ylabel('Computation latency [sec]')
        plt.show()
    l1, l2, l3 = x1.mean() * 1000, x2.mean() * 1000, x3.mean() * 1000
    s = l1+l2+l3
    print(l1, l2, l3)
    print(1, l2/l1, l3/l1)
    print((l1/s)*100, (l2/s)*100, (l3/s)*100)

def plot_latency_with_block(filename, mat_size, fig_save):
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

    df = _get_dataframe(filename)
    plt.figure()
    plt.plot(df['Block size'], df['Time elapsed[sec]'], 'b')
    plt.title('MM latency with different block size (Total size=' + str(mat_size) + ')')
    plt.xlabel('Block size')
    plt.ylabel('Latency [sec]')
    if fig_save:
        plt.savefig('./[exp2-2] Latency_with_block_size_' + str(mat_size) + '_0321_2.png')
    else:
        plt.show()

def plot_thread_balance():
    def _get_std(mat_size) :
        f = open('./data/exp2-1/thread_balance_profile_' + str(mat_size) + '.txt', 'r')
        std_list = []
        for line in f:
            t = line.strip().split(' ')
            s = 0
            for i in range(8) :
                t[i] = int(t[i])
                s += t[i]
            t = np.array(t)
            std_list.append(np.std(t))
        total_job = s
        equal_partition = s/8
        for i in range(len(std_list)):
            std_list[i] = (std_list[i]/equal_partition) * 100
        print(total_job)
        return np.array(std_list)
    s1 = _get_std(500)
    s2 = _get_std(1000)
    s3 = _get_std(2000)

    plt.figure()
    plt.boxplot([s1, s2, s3])
    plt.title('Job Imbalance Percentage among 8 Threads')
    plt.xlabel('Matrix Size')
    plt.ylabel('Percentage [%]')
    plt.xticks([1, 2, 3], ['500', '1000', '2000'])
    plt.savefig('./[exp2-1] Thread_job_balance_0319.png')
    #plt.show()


def plot_latency_with_block_multiple(filename_base, iter, mat_size, filter, fig_save):
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
    def _moving_average(x, tap):
        x_filtered = []
        for i in range(len(x)) :
            left = max(0, i-tap)
            right = min(len(x)-1, i+tap)
            avg = 0
            for j in range(left, right+1):
                avg += x[j]
            x_filtered.append(avg/(right-left+1))
        return x_filtered

    color_list = ['blue', 'cyan', 'skyblue', 'lime', 'green']
    plt.figure()
    for i in range(iter):
        filename = filename_base + str(i+1) + '.txt'
        color = color_list[i]
        df =  _get_dataframe(filename)
        x = np.array(df['Block size'])
        y = np.array(df['Time elapsed[sec]'])
        if filter:
            y = _moving_average(y, 2)
            y = gaussian_filter1d(y, 2)
        plt.plot(x, y, color=color, label=str(i+1) + 'th')

    plt.title('MM latency with different block size (Total size=' + str(mat_size) + ')')
    plt.xlabel('Block size')
    plt.ylabel('Latency [sec]')
    plt.legend()

    if fig_save:
        plt.savefig('./graph/[exp3-2] Latency_with_block_size_' + str(mat_size) + '.png')
    else:
        plt.show()

def plot_latency_with_block_filtered(filename, mat_size, fig_save):
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

    def _moving_average(x, tap):
        x_filtered = []
        for i in range(len(x)) :
            left = max(0, i-tap)
            right = min(len(x)-1, i+tap)
            avg = 0
            for j in range(left, right+1):
                avg += x[j]
            x_filtered.append(avg/(right-left+1))
        return x_filtered
    df = _get_dataframe(filename)
    x = df['Block size']
    y = df['Time elapsed[sec]']
    x, y = np.array(x), np.array(y)
    y_filtered = _moving_average(y, 2)
    y_gaussian = gaussian_filter1d(y, 2)
    plt.figure()
    plt.plot(x, y, color='skyblue')
    plt.plot(x, y_filtered, color='blue')
    plt.plot(x, y_gaussian, color='green')
    plt.title('MM latency with different block size (Total size=' + str(mat_size) + ')')
    plt.xlabel('Block size')
    plt.ylabel('Latency [sec]')
    if fig_save:
        plt.savefig('./[exp2-2] Latency_with_block_size_' + str(mat_size) + '_0321_2.png')
    else:
        plt.show()

#plot_latency_with_size()
#plot_latency_with_size_multithread(1)
#plot_phase_time('./data/exp2-3/latency_component_512_32.txt', 0)
#plot_phase_time('./data/exp2-3/latency_component_1024_32.txt', 0)
#plot_phase_time('./data/exp2-3/latency_component_2048_32.txt', 0)
#plot_latency_with_block(512, 1)
#plot_latency_with_block(1024)
#plot_latency_with_block(2048)
#plot_thread_balance()
plot_latency_with_block_multiple('./data/exp3-2/[exp3-2]Mat_size_512_', 5, 512, filter=0, fig_save=0)
#plot_latency_with_block_multiple('./data/exp3-2/[exp3-2]Mat_size_1024_', 5, 1024, fig_save=1)
#plot_latency_with_block_filtered('./data/exp3-2/[exp3-2]Mat_size_512_1.txt', 512, fig_save=0)