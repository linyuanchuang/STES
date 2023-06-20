import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

data = pd.read_csv(r'./mofs_cofs.csv',sep=',',encoding='gbk')
datas= np.array(data.loc[:,['density','Qst']])



title_font = {'family': 'Arial',
        'size': 20}    
marker_size = 150
plt.rc('font', size=18,family= 'Arial')

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3,init='k-means++',n_init=10,max_iter=500,tol=0.0001,verbose=0,random_state=None,copy_x=True,algorithm='auto') 
model.fit(datas) #开始聚类
label_pred = model.labels_ #获取聚类标签
print(metrics.silhouette_score(datas,label_pred))   #所有样本的平均轮廓系数
#绘制k-means结果
x0 = datas[label_pred == 0]
x1 = datas[label_pred == 1]
x2 = datas[label_pred == 2]


fig = plt.figure(figsize=(8, 8), dpi=80)
ax1 = fig.add_subplot(1,1,1)

ax1.scatter(x2[:, 0], x2[:, 1], c = "gold", marker='.',  s=marker_size, label='high')
ax1.scatter(x1[:, 0], x1[:, 1], c = "forestgreen", marker='.',  s=marker_size, label='medium')
ax1.scatter(x0[:, 0], x0[:, 1], c = "royalblue", marker='.',  s=marker_size, label='low')

ax1.set_xlim(0, 4.5)
ax1.set_ylim(0 ,800)
yticks = np.linspace(0 ,800,9)
xticks = np.linspace(0, 4.5, 10)

ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
plt.ylabel('$\mathrm{Q_{ed}}$ (J/g)',  fontdict=title_font)
plt.xlabel(chr(961)+' (kg/m$^3$)',  fontdict=title_font)
plt.legend(loc='best')



plt.savefig('clustering.pdf', format='pdf', dpi=600)
plt.savefig('clustering.png', format='png', dpi=600)

plt.show()
'''
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=3,linkage='ward')
model.fit(datas)  # 训练模型
label_pred = model.labels_  # 输出模型结果 
print(metrics.silhouette_score(datas,label_pred))   #所有样本的平均轮廓系数
#绘制层次聚类hierarchical clustering结果
x0 = datas[label_pred == 0]
x1 = datas[label_pred == 1]
x2 = datas[label_pred == 2]
ax2 = fig.add_subplot(2,2,2)
if x2[2, 1] > 200:
    ax2.scatter(x2[:, 0], x2[:, 1], c = "red", marker='.',  s=marker_size, label='high')
elif x2[2, 1] < 50:
    ax2.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='.',  s=marker_size, label='low')
else:
    ax2.scatter(x2[:, 0], x2[:, 1], c = "green", marker='.',  s=marker_size, label='medium')

if x1[2, 1] > 200:
    ax2.scatter(x1[:, 0], x1[:, 1], c = "red", marker='.',  s=marker_size, label='high')
elif x1[2, 1] < 50:
    ax2.scatter(x1[:, 0], x1[:, 1], c = "blue", marker='.',  s=marker_size, label='low')
else:
    ax2.scatter(x1[:, 0], x1[:, 1], c = "green", marker='.',  s=marker_size, label='medium')
    
if x0[2, 1] > 200:
    ax2.scatter(x0[:, 0], x0[:, 1], c = "red", marker='.',  s=marker_size, label='high')
elif x0[2, 1] < 50:
    ax2.scatter(x0[:, 0], x0[:, 1], c = "blue", marker='.',  s=marker_size, label='low')
else:
    ax2.scatter(x0[:, 0], x0[:, 1], c = "green", marker='.',  s=marker_size, label='medium') 
ax2.set_xlim(0, 4.5)
ax2.set_ylim(0 ,800)
yticks = np.linspace(0 ,800,9)
xticks = np.linspace(0, 4.5, 10)
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)




from sklearn.cluster import Birch
model= Birch(n_clusters =3)
model.fit(datas)   #开始聚类
label_pred = model.labels_ #获取聚类标签
print(metrics.silhouette_score(datas,label_pred))   #所有样本的平均轮廓系数
#绘制高斯混合模型（GMM）结果
x0 = datas[label_pred == 0]
x1 = datas[label_pred == 1]
x2 = datas[label_pred == 2]
ax6 = fig.add_subplot(2,2,3)
if x2[2, 1] > 300:
    ax6.scatter(x2[:, 0], x2[:, 1], c = "red", marker='.',  s=marker_size, label='high')
elif x2[2, 1] < 150:
    ax6.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='.',  s=marker_size, label='low')
else:
    ax6.scatter(x2[:, 0], x2[:, 1], c = "green", marker='.',  s=marker_size, label='medium')

if x1[2, 1] > 300:
    ax6.scatter(x1[:, 0], x1[:, 1], c = "red", marker='.',  s=marker_size, label='high')
elif x1[2, 1] < 150:
    ax6.scatter(x1[:, 0], x1[:, 1], c = "blue", marker='.',  s=marker_size, label='low')
else:
    ax6.scatter(x1[:, 0], x1[:, 1], c = "green", marker='.',  s=marker_size, label='medium')
    
if x0[2, 1] > 300:
    ax6.scatter(x0[:, 0], x0[:, 1], c = "red", marker='.',  s=marker_size, label='high')
elif x0[2, 1] < 150:
    ax6.scatter(x0[:, 0], x0[:, 1], c = "blue", marker='.',  s=marker_size, label='low')
else:
    ax6.scatter(x0[:, 0], x0[:, 1], c = "green", marker='.',  s=marker_size, label='medium') 
ax6.set_xlim(0, 4.5)
ax6.set_ylim(0 ,800)
yticks = np.linspace(0 ,800,9)
xticks = np.linspace(0, 4.5, 10)

ax6.set_xticks(xticks)
ax6.set_yticks(yticks)
plt.ylabel('$\mathrm{Q_{st}}$',  fontdict=title_font)
plt.xlabel('Density',  fontdict=title_font)


from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=3)
model.fit(datas)
label_pred = model.predict(datas) 
x0 = datas[label_pred == 0]
x1 = datas[label_pred == 1]
x2 = datas[label_pred == 2]
ax6 = fig.add_subplot(2,2,4)
if x2[2, 1] > 200:
    ax6.scatter(x2[:, 0], x2[:, 1], c = "red", marker='.',  s=marker_size, label='high')
elif x2[2, 1] < 50:
    ax6.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='.',  s=marker_size, label='low')
else:
    ax6.scatter(x2[:, 0], x2[:, 1], c = "green", marker='.',  s=marker_size, label='medium')

if x1[2, 1] > 200:
    ax6.scatter(x1[:, 0], x1[:, 1], c = "red", marker='.',  s=marker_size, label='high')
elif x1[2, 1] < 50:
    ax6.scatter(x1[:, 0], x1[:, 1], c = "blue", marker='.',  s=marker_size, label='low')
else:
    ax6.scatter(x1[:, 0], x1[:, 1], c = "green", marker='.',  s=marker_size, label='medium')
    
if x0[2, 1] > 200:
    ax6.scatter(x0[:, 0], x0[:, 1], c = "red", marker='.',  s=marker_size, label='high')
elif x0[2, 1] < 50:
    ax6.scatter(x0[:, 0], x0[:, 1], c = "blue", marker='.',  s=marker_size, label='low')
else:
    ax6.scatter(x0[:, 0], x0[:, 1], c = "green", marker='.',  s=marker_size, label='medium') 
ax6.set_xlim(0, 4.5)
ax6.set_ylim(0 ,800)
yticks = np.linspace(0 ,800,9)
xticks = np.linspace(0, 4.5, 10)

ax6.set_xticks(xticks)
ax6.set_yticks(yticks)
plt.xlabel('Density',  fontdict=title_font)


plt.show()
#'''