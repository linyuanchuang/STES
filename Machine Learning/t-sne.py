import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Arial', size='20')
cmap = plt.get_cmap('viridis')


data_A = pd.read_csv(r'./mofs_cofs.csv',sep=',',encoding='gbk')
data_B = pd.read_csv(r'./core_mofs.csv',sep=',',encoding='gbk')
data_C = pd.read_csv(r'./core_cofs.csv',sep=',',encoding='gbk')

features_A = data_A.loc[:,['LCD','VF','density','PLD','Va','ASA']]
features_B = data_B.loc[:,['LCD','VF','Density','PLD','Va','ASA']]
features_C = data_C.loc[:,['LCD','VF','Density','PLD','Va','ASA']]
LCD_A = np.array(data_A.loc[:,['VF']])
LCD_B = np.array(data_B.loc[:,['VF']])
LCD_C = np.array(data_C.loc[:,['VF']])
LCD = []

for i in range (len(LCD_A)):
    LCD.append(LCD_A[i][0])
for i in range (len(LCD_B)):
    LCD.append(LCD_B[i][0])
for i in range (len(LCD_C)):
    LCD.append(LCD_C[i][0])
    
    
labels_A = pd.Series(["A"] * len(features_A))
labels_B = pd.Series(["B"] * len(features_B))
labels_C = pd.Series(["C"] * len(features_C))

features = pd.concat([features_A, features_B, features_C], axis=0)
labels = pd.concat([labels_A, labels_B, labels_C], axis=0)
tsne = TSNE(n_components=2, perplexity=10, learning_rate=200)
features_embedded = tsne.fit_transform(features)

fig = plt.figure(figsize=(18, 12), dpi=80)




ax1 = fig.add_subplot(2,2,1)
scatter1 = ax1.scatter(features_embedded[:, 0], features_embedded[:, 1],c=LCD, cmap=cmap)
#添加colorbar的坐标轴，设置相对位置和大小
cax = fig.add_axes([0.05, 0.55, 0.02, 0.35])
cbar1 = plt.colorbar(scatter1, cax=cax)

# 调整colorbar标签位置和方向
cbar1.ax.yaxis.set_ticks_position('left') 
cbar1.ax.yaxis.set_label_position('left') 
cbar1.set_label('VF')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.axis('off')

ax2 = fig.add_subplot(2,2,2)
index_B = labels_B.index[labels_B == "B"]

scatter2_1 = ax2.scatter(features_embedded[:, 0], features_embedded[:, 1],c='grey')
scatter2_2 = ax2.scatter(features_embedded[index_B, 0], features_embedded[index_B, 1], c="#36AE37")
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.axis('off')

ax3 = fig.add_subplot(2,2,3)
index_C = labels_C.index[labels_C == "C"]
scatter3_1 = ax3.scatter(features_embedded[:, 0], features_embedded[:, 1],c='grey')
scatter3_2 = ax3.scatter(features_embedded[index_C, 0], features_embedded[index_C, 1], c="#164993")
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.axis('off')

ax4 = fig.add_subplot(2,2,4)
index_A = labels_A.index[labels_A == "A"]
scatter4_1 = ax4.scatter(features_embedded[:, 0], features_embedded[:, 1],c='grey')
scatter4_2 = ax4.scatter(features_embedded[index_A, 0], features_embedded[index_A, 1], c="#E71F19")
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.axis('off')

# 在四张子图的中央添加整体的图例
handles = [scatter1, scatter2_2, scatter3_2, scatter4_2]
labels = ['All databases', 'CoRE MOFs-2019', 'CoRE COFs-v6.0', 'The databases in this work']
fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=4)

plt.savefig('tsne1.pdf', format='pdf', dpi=600)
plt.savefig('tsne1.png', format='png', dpi=600)




fig = plt.figure(figsize=(18, 12), dpi=80)

ax4 = fig.add_subplot(1,1,1)
index_A = labels_A.index[labels_A == "A"]
scatter4_1 = ax4.scatter(features_embedded[:, 0], features_embedded[:, 1],c='grey',label='CoRE MOFs-2019 and CoRE COFs-v6.0' )
scatter4_2 = ax4.scatter(features_embedded[index_A, 0], features_embedded[index_A, 1], c="#E71F19",label='The databases in this work')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.axis('off')
plt.legend()
#
plt.savefig('tsne2.png', format='png', dpi=600)
plt.savefig('tsne2.pdf', format='pdf', dpi=600)
plt.show() # 显示图像