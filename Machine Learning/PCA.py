import sklearn.preprocessing
from sklearn.decomposition import PCA
from numpy import *
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn.preprocessing import MinMaxScaler   # 归一化

from sklearn import preprocessing
plt.rc('font', family='Arial', size='20')
# sns.set(style='whitegrid')
plt.style.use('seaborn')

cm = plt.cm.get_cmap('jet')

data = pd.read_csv(r'./mofs_cofs.csv',sep=',',encoding='gbk')
Qst = data['Qst']
datas= data.loc[:,['LCD','density','ΔW','-<ΔadsH>']]
dataMat = np.array(datas)
dataMat = preprocessing.MinMaxScaler().fit_transform(dataMat)  # 归一化


data = open(f'PCA/pcating.dat', 'w+')
for i in range(len(dataMat)):
    output_data = dataMat[i]
    data.write(str(output_data) + '\n')

pca = PCA(n_components=2)  # 初始化保留前4个主成分
pca.fit(dataMat)  # 进行数据的拟合
new_X = pca.transform(dataMat)  # 将数据X转换成降维后的数据
X = pca.inverse_transform(new_X)  # 将降维后的数据转换成原始数据
# new_X = preprocessing.MinMaxScaler().fit_transform(new_X)
#print(X)


output = open('PCA/PCA.data', 'w+')
output.write(str(pca.components_) + '\n')
output.write(str(pca.explained_variance_ratio_))

out = open('PCA/PCA_a.dat', 'w+')
for i in range(len(new_X)):
    output_data = new_X[i]
    out.write(str(output_data) + '\n')
print(pca.components_)  # 返回具有最大方差的成分
print(pca.explained_variance_ratio_) # 返回所保留的n个成分各自的方差百分


def draw_vector(v0, v1, name):
    label_f1 = name
    ax1.annotate('', xy=(v0, v1), xytext=(0, 0), color='black',
                 arrowprops=dict(arrowstyle='->', linewidth=2)
                 )
    ax1.text(v0, v1, label_f1, fontsize=18, verticalalignment="bottom", horizontalalignment="left")

MOFs_X = new_X[:, 0]
MOFs_Y = new_X[:, 1]

plt.rc('font', size=14,family= 'Arial')
fig, axs = plt.subplots()
ax1 = axs
sc1 = ax1.scatter(MOFs_X, MOFs_Y, c=Qst, s=70,  cmap=cm, norm=matplotlib.colors.LogNorm())
# sc1.set_facecolor("black")
bar2 = plt.colorbar(sc1, ax=ax1)
bar2.ax.set_title('$\mathrm{Q_{ed}}$ (J/g)', y=1.02, fontsize=20)
bar2.ax.tick_params(labelsize=16)



ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax1.set_xticks(np.linspace(-1, 1,9, endpoint=True))
ax1.set_yticks(np.linspace(-1, 1, 9, endpoint=True))

# draw_vector(pca.components_[0][0], pca.components_[0][1], '1')
# draw_vector(pca.components_[1][0], pca.components_[1][1], 'V$\mathrm{_{a}}$')

# draw_vector(pca.components_[3][0], pca.components_[3][1], r'$\mathrm{\Delta{W}}$')
ax1.annotate('', xy=(pca.components_[0][0], pca.components_[1][0]), xytext=(0, 0), color='black',
             arrowprops=dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0))
ax1.text(pca.components_[0][0], pca.components_[1][0] - 0.12, 'LCD', fontsize=18,
         verticalalignment="bottom", horizontalalignment="left")

ax1.annotate('', xy=(pca.components_[0][1], pca.components_[1][1]), xytext=(0, 0), color='black',
             arrowprops=dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0))
ax1.text(pca.components_[0][1] - 0.12, pca.components_[1][1] - 0.12, 'density', fontsize=18,
         verticalalignment="bottom", horizontalalignment="left")



draw_vector(pca.components_[0][2]-0.02, pca.components_[1][2], 'ΔW')


ax1.annotate('', xy=(pca.components_[0][3], pca.components_[1][3]), xytext=(0, 0), color='black',
             arrowprops=dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0))
ax1.text(pca.components_[0][3] - 0.12, pca.components_[1][3]+0.05, '<$\Delta$$\mathrm{_{ads}}$H>', fontsize=18,
         verticalalignment="bottom", horizontalalignment="left")
         


# draw_vector(pca.components_[5][0], pca.components_[5][1], '6')


fig.set_size_inches(10, 7.2)
plt.savefig('PCA.pdf', format='pdf', dpi=600)
plt.savefig('PCA.png', format='png', dpi=600)
plt.show()
#'''
