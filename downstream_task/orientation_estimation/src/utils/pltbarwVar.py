"""画bar ， 高度是均值，上下线是方差
"""


import matplotlib.pyplot as plt
import numpy as np
import json


# 画出均值和标准差的柱状图
class DataVisualizer:
    def __init__(self, data_dict: dict):
        # data_dict : key 是横轴 ， value 是一维的list， 用于计算均值和标准差
        self.data_dict = data_dict
    
    # 计算均值和标准差
    def calculate_mean_std(self):
        mean_dict = {}
        std_dict = {}
        for key, value in self.data_dict.items():
            mean_dict[key] = np.mean(value)
            std_dict[key] = np.std(value, ddof=1)
        return mean_dict, std_dict
    
    # 画出柱状图
    def plot_bar(self, title, x_label, y_label, save_path):
        '''mean_dict, std_dict = self.calculate_mean_std()
        x = list(mean_dict.keys())
        y = list(mean_dict.values())
        std = list(std_dict.values())
        plt.bar(x, y, yerr=std, capsize=5)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(save_path)
        # plt.show()'''

        mean_dict, std_dict = self.calculate_mean_std()
        
        base = list(mean_dict.keys())
        means = list(mean_dict.values())
        stds = list(std_dict.values())

        # 根据均值对base, means, stds进行排序，降序
        base, means, stds = zip(*sorted(zip(base, means, stds), key=lambda x: x[2], reverse=True))

        # 取用前20个类别
        # base = base[20:17+20]
        # means = means[20:17+20]
        # stds = stds[20:17+20]
        base = base[:17]
        means = means[:17]
        stds = stds[:17]

        # 绘制柱形图
        # 设置图像大小
        # fig = plt.figure(figsize=(12, 6))
        fig, ax = plt.subplots(figsize=(15, 6))
        fig.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.1)  # 调整子图的边界
        bar_width = 0.6
        index = range(len(base))

        # 绘制柱形图
        color_3 = (122 / 255, 182 / 255, 83 / 255)
        color_4 = (192 / 255, 50 / 255,  26 / 255)
        bars = ax.bar(base, means, bar_width, align='center', color=color_3, yerr=stds, capsize=3, error_kw={'ecolor': color_4})  # 修改标准差线条的颜色为color_4

        # 添加标签
        ax.set_xlabel('Categories', fontsize=24)
        ax.set_ylabel('Chamfer distance', fontsize=24)
        


        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

        ax.set_xticks(base)
        ax.set_xticklabels([str(value) for value in base], rotation=40)  # 旋转45度显示x轴刻度标签
        # 设置不显示横轴标签
        ax.set_xticklabels([])
        # ax.legend(['Scores\' Mean'], fontsize=14)

        legend_elements = [(bars, bars.errorbar)]
        labels = ["Mean and S.D."]
        ax.legend(legend_elements, labels, fontsize=24)

        offset_id = 0
        #            0     1     2     3     4     5     10    20    30    40    50    100   200    300   400   500   1000
        x_offsets = [0.20, 0.00, 0.00, 0.10, -0.2, -0.2, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.1, -0.3, 0.1,  -0.1,  -0.3]
        y_offsets = [0.06, 0.09, 0.12, 0.12, 0.10, 0.12, 0.12, 0.13, 0.13, 0.17, 0.15, 0.14, 0.14, 0.13, 0.10, 0.11, 0.10]
        # 在每个柱形上标注Mean的值，并在下一行用红色标注标准差
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2 - x_offsets[offset_id], height + std + 0.06, f'{mean:.2f}', ha='center', va='bottom', fontsize=14, color= 'black')
            ax.text(bar.get_x() + bar.get_width() / 2 - x_offsets[offset_id], height + std + 0.05, f'{std:.2f}', ha='center', va='top', fontsize=14, color='red')  # 标准差以红色显示
            offset_id = offset_id +1
            
        plt.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.20)

        plt.ylim(-0.05, 0.85)
        plt.savefig(save_path)

if __name__ == '__main__':
    data_path = '/data4/jl/project/one-shot/results_org/rebuttal/inter-cate_cds/cate_cds_dict.json'
    with open(data_path, 'r') as f:
        data_dict = json.load(f)
    # # 先只取用其中的10个类别
    # data_dict = {key: value for key, value in data_dict.items() if key in list(data_dict.keys())[:10]}
    # data_dict = {'A':[0.2, 0.3, 0.4, 0.5], 'B':[0.3, 0.4, 0.5, 0.6]}
    dv = DataVisualizer(data_dict)
    dv.plot_bar('Mean and Std', 'Attributes', 'Mean', '/data4/jl/project/one-shot/results/test/bar.png')

    asdf










import sys
# sys.path.append("/home/songxiuqiang/project/AR3DVEngine/ar_utils/python")
# import directory as mydir

from sklearn import preprocessing  
import matplotlib
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from matplotlib.pyplot import MultipleLocator
from matplotlib import ticker
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import laplace
import math
from scipy import stats

from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
    
class HandlerErrorbar(HandlerLine2D):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # 创建一个竖线和两个小横线
        line = Line2D([width / 2, width / 2], [0, height],
                      color=orig_handle.get_color(), linestyle=orig_handle.get_linestyle(), linewidth=1.5)
        capline1 = Line2D([width / 2 - 3, width / 2 + 3], [height, height],
                          color=orig_handle.get_color(), linewidth=1.5)
        capline2 = Line2D([width / 2 - 3, width / 2 + 3], [0, 0],
                          color=orig_handle.get_color(), linewidth=1.5)
        return [line, capline1, capline2]

plt.rcParams['font.family'] = 'Times New Roman'

class User:
    def __init__(self, id, name, gender, age, ar_exp, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q):
        self.id = id
        self.name = name
        self.gender = gender
        self.age = age
        self.ar_exp = ar_exp
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F
        self.G = G
        self.H = H
        self.I = I
        self.J = J
        self.K = K
        self.L = L
        self.M = M
        self.N = N
        self.O = O
        self.P = P
        self.Q = Q
        # 数据合法性检查
        attributes = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G, 'H': H, 'I': I, 'J': J, 'K': K, 'L' : L, 'M' : M, 'N' : N, 'O' : O, 'P' : P, 'Q' : Q}
        for key, value in attributes.items():
            if value < 0.0 or value > 1.0:
                print(f"Warning: {key} is not within the range [0.0, 1.0]")
                
# 用户对象
# user 2 3 4 8 20不是学生
user1  = User(id=1,  name='sxq', gender='男', age=27, ar_exp='True',  A=0.2, B=0.8, C=0.0, D=0.0, E=0.9, F=0.4, G=0.7, H=0.0, I=0.0, J=0.5, K=1.0, L=0.2, M=0.1, N=0.8, O=1.0, P=0.0, Q=0.0)
user2  = User(id=2,  name='ljc', gender='男', age=31, ar_exp='True',  A=0.2, B=0.8, C=0.0, D=0.0, E=0.9, F=0.3, G=0.8, H=0.0, I=0.0, J=0.6, K=1.0, L=0.3, M=0.0, N=0.9, O=1.0, P=0.0, Q=0.0)
user3  = User(id=3,  name='dxx', gender='男', age=26, ar_exp='False', A=0.3, B=0.7, C=0.0, D=0.0, E=0.8, F=0.4, G=0.7, H=0.1, I=0.1, J=0.6, K=0.9, L=0.5, M=0.2, N=0.8, O=1.0, P=0.0, Q=0.1)
user4  = User(id=4,  name='zzx', gender='男', age=23, ar_exp='True',  A=0.3, B=0.8, C=0.0, D=0.0, E=0.9, F=0.4, G=0.7, H=0.0, I=0.0, J=0.5, K=1.0, L=0.2, M=0.1, N=0.8, O=1.0, P=0.0, Q=0.0)
user5  = User(id=5,  name='yj',  gender='女', age=22, ar_exp='False', A=0.3, B=0.9, C=0.0, D=0.0, E=0.9, F=0.3, G=0.7, H=0.0, I=0.0, J=0.6, K=1.0, L=0.1, M=0.2, N=0.8, O=1.0, P=0.0, Q=0.1)
user6  = User(id=6,  name='scx', gender='男', age=27, ar_exp='False', A=0.4, B=0.8, C=0.0, D=0.0, E=0.8, F=0.5, G=0.7, H=0.1, I=0.1, J=0.6, K=0.9, L=0.3, M=0.2, N=0.7, O=1.0, P=0.0, Q=0.1)
user7  = User(id=7,  name='tjk', gender='男', age=25, ar_exp='False', A=0.2, B=0.7, C=0.0, D=0.0, E=0.8, F=0.4, G=0.7, H=0.1, I=0.1, J=0.5, K=1.0, L=0.2, M=0.1, N=0.8, O=1.0, P=0.0, Q=0.1)
user8  = User(id=8,  name='sp',  gender='女', age=35, ar_exp='False', A=0.4, B=0.8, C=0.0, D=0.0, E=0.9, F=0.3, G=0.8, H=0.0, I=0.0, J=0.6, K=0.9, L=0.2, M=0.1, N=0.9, O=1.0, P=0.0, Q=0.0)
user9  = User(id=9,  name='wxl', gender='男', age=24, ar_exp='False', A=0.4, B=0.7, C=0.0, D=0.0, E=0.8, F=0.3, G=0.8, H=0.2, I=0.1, J=0.6, K=1.0, L=0.3, M=0.2, N=0.8, O=1.0, P=0.0, Q=0.1)
user10 = User(id=10, name='zyf', gender='女', age=24, ar_exp='True',  A=0.5, B=0.7, C=0.0, D=0.1, E=0.8, F=0.4, G=0.7, H=0.2, I=0.3, J=0.5, K=1.0, L=0.4, M=0.3, N=0.8, O=1.0, P=0.1, Q=0.3)
user11 = User(id=11, name='thw', gender='男', age=24, ar_exp='False', A=0.3, B=0.8, C=0.0, D=0.0, E=0.8, F=0.4, G=0.6, H=0.1, I=0.1, J=0.7, K=0.9, L=0.4, M=0.2, N=0.7, O=1.0, P=0.0, Q=0.2)
user12 = User(id=12, name='czy', gender='男', age=24, ar_exp='True',  A=0.3, B=0.8, C=0.0, D=0.0, E=0.9, F=0.4, G=0.8, H=0.2, I=0.2, J=0.5, K=0.9, L=0.2, M=0.1, N=0.7, O=1.0, P=0.0, Q=0.2)
user13 = User(id=13, name='zpp', gender='女', age=25, ar_exp='True',  A=0.3, B=0.8, C=0.0, D=0.0, E=0.8, F=0.4, G=0.8, H=0.1, I=0.1, J=0.5, K=1.0, L=0.3, M=0.2, N=0.8, O=1.0, P=0.0, Q=0.2)
user14 = User(id=14, name='lwl', gender='男', age=24, ar_exp='True',  A=0.2, B=0.8, C=0.0, D=0.0, E=0.8, F=0.5, G=0.8, H=0.1, I=0.0, J=0.6, K=0.9, L=0.3, M=0.2, N=0.8, O=1.0, P=0.0, Q=0.1)
user15 = User(id=15, name='lk',  gender='男', age=23, ar_exp='False', A=0.2, B=0.7, C=0.0, D=0.0, E=0.8, F=0.4, G=0.7, H=0.1, I=0.1, J=0.5, K=0.9, L=0.2, M=0.1, N=0.8, O=1.0, P=0.0, Q=0.2)
user16 = User(id=16, name='yc',  gender='男', age=29, ar_exp='False', A=0.5, B=0.8, C=0.1, D=0.2, E=0.9, F=0.7, G=0.7, H=0.3, I=0.4, J=0.7, K=1.0, L=0.5, M=0.5, N=0.8, O=1.0, P=0.2, Q=0.3)
user17 = User(id=17, name='xl',  gender='男', age=23, ar_exp='False', A=0.3, B=0.8, C=0.0, D=0.0, E=0.9, F=0.4, G=0.7, H=0.1, I=0.1, J=0.5, K=1.0, L=0.3, M=0.2, N=0.8, O=1.0, P=0.0, Q=0.1)
user18 = User(id=18, name='lw',  gender='男', age=31, ar_exp='False', A=0.2, B=0.8, C=0.1, D=0.1, E=0.8, F=0.5, G=0.8, H=0.2, I=0.1, J=0.6, K=0.9, L=0.3, M=0.2, N=0.8, O=1.0, P=0.0, Q=0.1)
user19 = User(id=19, name='kf',  gender='女', age=25, ar_exp='True',  A=0.3, B=0.7, C=0.0, D=0.0, E=0.9, F=0.4, G=0.8, H=0.0, I=0.1, J=0.6, K=1.0, L=0.4, M=0.3, N=0.8, O=1.0, P=0.0, Q=0.3)
user20 = User(id=20, name='cyl', gender='男', age=29, ar_exp='False', A=0.3, B=0.8, C=0.0, D=0.1, E=0.9, F=0.4, G=0.6, H=0.2, I=0.1, J=0.7, K=0.9, L=0.4, M=0.4, N=0.8, O=1.0, P=0.1, Q=0.1)
user21 = User(id=21, name='gmc', gender='男', age=23, ar_exp='False', A=0.2, B=0.8, C=0.0, D=0.0, E=0.9, F=0.5, G=0.7, H=0.2, I=0.1, J=0.6, K=1.0, L=0.4, M=0.2, N=0.8, O=1.0, P=0.0, Q=0.1)
user22 = User(id=22, name='grq', gender='男', age=23, ar_exp='False', A=0.3, B=0.6, C=0.0, D=0.1, E=0.8, F=0.5, G=0.7, H=0.2, I=0.2, J=0.5, K=0.9, L=0.3, M=0.3, N=0.8, O=1.0, P=0.1, Q=0.3)
user23 = User(id=23, name='xr',  gender='男', age=24, ar_exp='False', A=0.2, B=0.7, C=0.1, D=0.0, E=0.9, F=0.5, G=0.7, H=0.0, I=0.0, J=0.7, K=1.0, L=0.2, M=0.3, N=0.7, O=1.0, P=0.2, Q=0.0)
user24 = User(id=24, name='xzr', gender='女', age=23, ar_exp='False', A=0.3, B=0.8, C=0.0, D=0.0, E=0.9, F=0.4, G=0.7, H=0.1, I=0.0, J=0.6, K=0.9, L=0.3, M=0.2, N=0.8, O=1.0, P=0.1, Q=0.1)
user25 = User(id=25, name='zsl', gender='男', age=25, ar_exp='False', A=0.2, B=0.7, C=0.0, D=0.0, E=0.8, F=0.4, G=0.7, H=0.1, I=0.1, J=0.5, K=1.0, L=0.2, M=0.1, N=0.8, O=1.0, P=0.0, Q=0.1)
user26 = User(id=26, name='ycl', gender='男', age=26, ar_exp='False', A=0.3, B=0.8, C=0.0, D=0.0, E=0.9, F=0.4, G=0.8, H=0.1, I=0.1, J=0.5, K=1.0, L=0.2, M=0.2, N=0.8, O=1.0, P=0.0, Q=0.1)
user27 = User(id=27, name='swb', gender='男', age=23, ar_exp='True',  A=0.3, B=0.8, C=0.0, D=0.0, E=0.9, F=0.3, G=0.7, H=0.1, I=0.1, J=0.5, K=1.0, L=0.3, M=0.3, N=0.8, O=1.0, P=0.0, Q=0.1)
user28 = User(id=28, name='cx',  gender='男', age=26, ar_exp='True',  A=0.2, B=0.7, C=0.0, D=0.0, E=0.8, F=0.5, G=0.8, H=0.0, I=0.0, J=0.6, K=1.0, L=0.6, M=0.2, N=0.9, O=1.0, P=0.0, Q=0.0)
user29 = User(id=29, name='jl',  gender='男', age=26, ar_exp='True',  A=0.2, B=0.6, C=0.0, D=0.1, E=0.6, F=0.4, G=0.5, H=0.1, I=0.1, J=0.5, K=0.9, L=0.5, M=0.3, N=0.7, O=1.0, P=0.1, Q=0.2)
user30 = User(id=30, name='syx', gender='男', age=18, ar_exp='False', A=0.3, B=0.9, C=0.0, D=0.0, E=0.8, F=0.4, G=0.7, H=0.0, I=0.0, J=0.5, K=1.0, L=0.4, M=0.2, N=0.9, O=1.0, P=0.0, Q=0.0)

all_users = [user1,  user2,  user3,  user4,  user5,  user6,  user7,  user8,  user9,  user10, \
           user11, user12, user13, user14, user15, user16, user17, user18, user19, user20, \
           user21, user22, user23, user24, user25, user26, user27, user28, user29, user30]
		   
letter_to_format_num = {'A':'30   ', 'B':'3    ', 'C':'1000 ', 'D':'400  ', 'E':'2    ', 'F':'20   ', 'G':'5    ', 'H':'300  ', 'I':'200  ', 'J':'10   ', \
                        'K':'1    ', 'L':'40   ', 'M':'50   ', 'N':'4    ', 'O':'0    ', 'P':'500  ', 'Q':'100  '}


# 计算所有用户的指定属性的均值和标准差
def calculate_mean_std(users, attribute):
    attribute_values = [getattr(user, attribute) for user in users]
    attribute_mean = np.mean(attribute_values)
    attribute_std = np.std(attribute_values,ddof=1)
    # print( "均值: {:.2f}".format(attribute_mean), " 标准差: {:.2f}".format(attribute_std))    
    print(letter_to_format_num[attribute] + "Mean: {:.2f}".format(attribute_mean), " Standard deviation: {:.2f}".format(attribute_std))    
    return attribute_mean, attribute_std

# mean_age, std_age = calculate_mean_std(all_users, 'age')

# OKEBNGJFALMQIHDPC
mean_A, std_A = calculate_mean_std(all_users, 'A') # 30
mean_B, std_B = calculate_mean_std(all_users, 'B') # 3
mean_C, std_C = calculate_mean_std(all_users, 'C') # 1000
mean_D, std_D = calculate_mean_std(all_users, 'D') # 400
mean_E, std_E = calculate_mean_std(all_users, 'E') # 2
mean_F, std_F = calculate_mean_std(all_users, 'F') # 20
mean_G, std_G = calculate_mean_std(all_users, 'G') # 5
mean_H, std_H = calculate_mean_std(all_users, 'H') # 300
mean_I, std_I = calculate_mean_std(all_users, 'I') # 200 
mean_J, std_J = calculate_mean_std(all_users, 'J') # 10
mean_K, std_K = calculate_mean_std(all_users, 'K') # 1
mean_L, std_L = calculate_mean_std(all_users, 'L') # 40
mean_M, std_M = calculate_mean_std(all_users, 'M') # 50
mean_N, std_N = calculate_mean_std(all_users, 'N') # 4
mean_O, std_O = calculate_mean_std(all_users, 'O') # 0
mean_P, std_P = calculate_mean_std(all_users, 'P') # 500
mean_Q, std_Q = calculate_mean_std(all_users, 'Q') # 10000

print('Fail  &', 0, '   &', 1, '   &', 2,'   &', 3, '   &', 4, '   &', 5, '   &', 10, '  &', 20, '  &', 30, '  &', 40, '  &', 50, '  &', 100, ' &', 200, ' &', 300, ' &', 400, ' &', 500, ' &', 1000)
# 输出均值和标准差，显示两位小数，并在每个数字前添加'& '
print('mean:', '& {:.2f}'.format(mean_O), '& {:.2f}'.format(mean_K), '& {:.2f}'.format(mean_E), '& {:.2f}'.format(mean_B), '& {:.2f}'.format(mean_N), '& {:.2f}'.format(mean_G), '& {:.2f}'.format(mean_J), '& {:.2f}'.format(mean_F), '& {:.2f}'.format(mean_A), '& {:.2f}'.format(mean_L), '& {:.2f}'.format(mean_M), '& {:.2f}'.format(mean_Q), '& {:.2f}'.format(mean_I), '& {:.2f}'.format(mean_H), '& {:.2f}'.format(mean_D), '& {:.2f}'.format(mean_P), '& {:.2f}'.format(mean_C))
print('std: ', '& {:.2f}'.format(std_O), '& {:.2f}'.format(std_K), '& {:.2f}'.format(std_E), '& {:.2f}'.format(std_B), '& {:.2f}'.format(std_N), '& {:.2f}'.format(std_G), '& {:.2f}'.format(std_J), '& {:.2f}'.format(std_F),  '& {:.2f}'.format(std_A),'& {:.2f}'.format(std_L), '& {:.2f}'.format(std_M), '& {:.2f}'.format(std_Q), '& {:.2f}'.format(std_I), '& {:.2f}'.format(std_H), '& {:.2f}'.format(std_D), '& {:.2f}'.format(std_P), '& {:.2f}'.format(std_C))

l2n = {'A':'30', 'B':'3', 'C':'1000', 'D':'400', 'E':'2', 'F':'20', 'G':'5', 'H':'300', 'I':'200', 'J':'10', 'K':'1', 'L':'40', 'M':'50', 'N':'4', 'O':'0', 'P':'500', 'Q':'100'}

# All the data
base  = [l2n['O'], l2n['K'], l2n['E'], l2n['B'], l2n['N'], l2n['G'], l2n['J'], l2n['F'], l2n['A'], l2n['L'], l2n['M'], l2n['Q'], l2n['I'], l2n['H'], l2n['D'], l2n['P'], l2n['C']]
means = [mean_O,   mean_K,   mean_E,   mean_B,   mean_N,   mean_G,   mean_J,   mean_F,   mean_A,   mean_L,   mean_M,   mean_Q,   mean_I,   mean_H,   mean_D,   mean_P,   mean_C]
stds  = [ std_O,    std_K,    std_E,    std_B,    std_N,    std_G,    std_J,    std_F,    std_A,    std_L,    std_M,    std_Q,    std_I,    std_H,    std_D,    std_P,    std_C]

# base  = [l2n['O'], l2n['K'], l2n['E'], l2n['B'], l2n['N'], l2n['G'], l2n['J'], l2n['F'], l2n['A'], l2n['L'], l2n['M'], l2n['Q']]
# means = [mean_O,   mean_K,   mean_E,   mean_B,   mean_N,   mean_G,   mean_J,   mean_F,   mean_A,   mean_L,   mean_M,   mean_Q  ]
# stds  = [ std_O,    std_K,    std_E,    std_B,    std_N,    std_G,    std_J,    std_F,    std_A,    std_L,    std_M,    std_Q  ]

# 绘制柱形图
fig, ax = plt.subplots()
fig.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.1)  # 调整子图的边界
bar_width = 0.6
index = range(len(base))

# 绘制柱形图
color_3 = (122 / 255, 182 / 255, 83 / 255)
color_4 = (192 / 255, 50 / 255,  26 / 255)
bars = ax.bar(base, means, bar_width, align='center', color=color_3, yerr=stds, capsize=3, error_kw={'ecolor': color_4})  # 修改标准差线条的颜色为color_4

# 添加标签
ax.set_xlabel('Proportion of Failures(‰)', fontsize=16)
ax.set_ylabel('User Scores', fontsize=16)

ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

ax.set_xticks(base)
ax.set_xticklabels([str(value) for value in base], rotation=40)  # 旋转45度显示x轴刻度标签
# ax.legend(['Scores\' Mean'], fontsize=14)

legend_elements = [(bars, bars.errorbar)]
labels = ["Scores' Mean and S.D."]
ax.legend(legend_elements, labels, fontsize=16)

offset_id = 0
#            0     1     2     3     4     5     10    20    30    40    50    100   200    300   400   500   1000
x_offsets = [0.20, 0.00, 0.00, 0.10, -0.2, -0.2, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.1, -0.3, 0.1,  -0.1,  -0.3]
y_offsets = [0.06, 0.09, 0.12, 0.12, 0.10, 0.12, 0.12, 0.13, 0.13, 0.17, 0.15, 0.14, 0.14, 0.13, 0.10, 0.11, 0.10]
# 在每个柱形上标注Mean的值，并在下一行用红色标注标准差
for bar, mean, std in zip(bars, means, stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2 - x_offsets[offset_id], height + std + 0.06, f'{mean:.2f}', ha='center', va='bottom', fontsize=10, color= 'black')
    ax.text(bar.get_x() + bar.get_width() / 2 - x_offsets[offset_id], height + std + 0.05, f'{std:.2f}', ha='center', va='top', fontsize=10, color='red')  # 标准差以红色显示
    offset_id = offset_id +1
    
plt.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.20)

plt.ylim(-0.05, 1.15)

# 显示图形
plt.show()
asfd


# 定义数据点
x_data = np.array([0,    1,    2,    3,    4,    5,    10,   20,   30,   40,   50,   100,   200,   300,  400,  500,  1000])
y_data = np.array([1.00, 0.96, 0.84, 0.76, 0.80, 0.72, 0.57, 0.42, 0.29, 0.31, 0.21, 0.12,  0.09,  0.10, 0.02, 0.03, 0.01])

# x_data = np.array([0,    1,    2,    3,    4,    5,    10,   20,   30,   40,   50,   100])
# y_data = np.array([1.00, 0.96, 0.84, 0.76, 0.80, 0.72, 0.57, 0.42, 0.29, 0.31, 0.21, 0.12])


# 定义指数衰减函数
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

# 使用curve_fit进行函数拟合
params, covariance = curve_fit(exp_decay, x_data, y_data, p0=[0.9, 0.01, 0])

# 拟合得到的参数
a, b, c = params
print(f"Fitting parameters: a={a:.2f}, b={b:.2f}, c={c:.2f}")

# 使用拟合参数生成y值
x_fit = np.linspace(0, 100, 400)
y_fit = exp_decay(x_fit, *params)

# 绘制数据点和拟合曲线
plt.figure(figsize=(7, 5))
plt.scatter(x_data, y_data, color=color_3, label='Scores\' Mean')  # 绘制数据点
plt.plot(x_fit, y_fit, 'b-', color = 'green', label='$g(FailureRate)$')  # 拟合曲线
# plt.plot(x_fit, y_fit, 'b-', color = 'green', label=f'$g = {a:.2f}e^{{-{b:.2f} (F.R.)}} + {c:.2f}$')  # 拟合曲线

plt.xlabel('Failures Rate(‰)',fontsize=18)
plt.ylabel('Values',fontsize=18)

# Change the font size of the tick labels
plt.tick_params(axis='both', which='major', labelsize=16)

plt.legend(fontsize=14)

# 设置x轴和y轴的显示范围，留出余量
plt.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.12)
plt.xlim(-10, 110)
plt.ylim(min(y_data) - 0.1, 1.1)  # 留出余量显示y值及其标准差
plt.show()

