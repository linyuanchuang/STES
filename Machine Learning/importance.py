import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import palettable




plt.rc('font', size=24,family= 'Arial')

data = pd.read_csv(r'mofs_cofs.csv',sep=',',encoding='gbk')

colormap = plt.get_cmap("jet")  # 色带
title_font = {'family': 'Arial',
        'weight': 'bold',
        'size': 20}    

##########################ΔW############################
df_y = data['ΔW']
df_X= data.drop(columns=['Qst','No','ΔW','-<ΔadsH>','MaxER','MinER','ANC','APC','Free_energy_ave'])
feat_labels = df_X.columns[0:]

from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

# define log transform function
def log_transform(x):
    return np.log(x)

# create FunctionTransformer object to apply log transform to KH column
log_transformer = FunctionTransformer(log_transform, validate=False)

# create MinMaxScaler object
scaler = MinMaxScaler()

# apply log transform to KH column and then normalize all columns
df_X['KH'] = log_transformer.transform(df_X[['KH']])
df_X.iloc[:, :] = scaler.fit_transform(df_X.iloc[:, :])

# reassign all columns (including KH column) to df_X
df_X = pd.DataFrame(df_X, columns=feat_labels)


df_y=df_y.values.tolist()
df_X=df_X.values.tolist()


df_X1 = []
df_y1 = []
df_X2 = []
df_y2 = []
for i in range(len(df_X)):
    if df_X[i][8]==1:
        df_X1.append(df_X[i])
        df_y1.append(df_y[i])
    else:
        df_X2.append(df_X[i])
        df_y2.append(df_y[i])

rest_data=[]
rest_data2=[]
for l in df_X1:
    #删除一行中第c列的值
    rest_l=l[:8]
    rest_l.extend(l[9:])
    #将删除后的结果加入结果数组
    rest_data.append(rest_l)
for l in df_X2:
    #删除一行中第c列的值
    rest_l=l[:8]
    rest_l.extend(l[9:])
    #将删除后的结果加入结果数组
    rest_data2.append(rest_l)
df_X1 = rest_data
df_X2 = rest_data2

# 切分数据集
train_x1, test_x1, train_y1, test_y1 = train_test_split(df_X1, df_y1, train_size=0.8, random_state=24)
train_x2, test_x2, train_y2, test_y2 = train_test_split(df_X2, df_y2, train_size=0.8, random_state=24)

train_x = train_x1 + train_x2
train_y = train_y1 + train_y2
test_x = test_x1 + test_x2
test_y = test_y1 + test_y2

def muti_score(model):
    warnings.filterwarnings('ignore')
    r2 = cross_val_score(model, train_x, train_y, scoring='r2', cv=10)
    print("r2:",r2.mean())

model = RandomForestRegressor(n_estimators=46,min_samples_split=4,min_samples_leaf=2,max_features=0.925, max_depth=19)
model = model.fit(train_x, train_y)
r2 = model.score(test_x, test_y)
r2_train = model.score(train_x, train_y)
print('dW:\ntrain:', r2_train)
print('test: ', r2)
muti_score(model)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1] #[::-1]表示将各指标按权重大小进行排序输出
for f in range(np.array(train_x).shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

s = (8,3)
col = np.zeros(s)
index_y = feat_labels.tolist()
index_y.remove('type')
for i in range(len(index_y)):
    index_1 = feat_labels[indices].tolist().index(index_y[i])
    col[i][0] = importances[indices][index_1]



##########################adsH############################
df_y = data['-<ΔadsH>']
df_X= data.drop(columns=['Qst','No','ΔW','-<ΔadsH>','MaxER','MinER','ANC','APC','Free_energy_ave'])
feat_labels = df_X.columns[0:]


# apply log transform to KH column and then normalize all columns
df_X['KH'] = log_transformer.transform(df_X[['KH']])
df_X.iloc[:, :] = scaler.fit_transform(df_X.iloc[:, :])

# reassign all columns (including KH column) to df_X
df_X = pd.DataFrame(df_X, columns=feat_labels)


df_y=df_y.values.tolist()
df_X=df_X.values.tolist()



df_X1 = []
df_y1 = []
df_X2 = []
df_y2 = []
for i in range(len(df_X)):
    if df_X[i][8]==1:
        df_X1.append(df_X[i])
        df_y1.append(df_y[i])
    else:
        df_X2.append(df_X[i])
        df_y2.append(df_y[i])

rest_data=[]
rest_data2=[]
for l in df_X1:
    #删除一行中第c列的值
    rest_l=l[:8]
    rest_l.extend(l[9:])
    #将删除后的结果加入结果数组
    rest_data.append(rest_l)
for l in df_X2:
    #删除一行中第c列的值
    rest_l=l[:8]
    rest_l.extend(l[9:])
    #将删除后的结果加入结果数组
    rest_data2.append(rest_l)
df_X1 = rest_data
df_X2 = rest_data2
# 切分数据集
train_x1, test_x1, train_y1, test_y1 = train_test_split(df_X1, df_y1, train_size=0.8, random_state=24)
train_x2, test_x2, train_y2, test_y2 = train_test_split(df_X2, df_y2, train_size=0.8, random_state=24)

train_x = train_x1 + train_x2
train_y = train_y1 + train_y2
test_x = test_x1 + test_x2
test_y = test_y1 + test_y2

model = RandomForestRegressor(n_estimators=463,min_samples_split=4,min_samples_leaf=1,max_features=0.392, max_depth=47)
model.fit(train_x, train_y)
r2 = model.score(test_x, test_y)
r2_train = model.score(train_x, train_y)
print('\nadsH:\ntrain:', r2_train)
print('test: ', r2)
muti_score(model)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1] #[::-1]表示将各指标按权重大小进行排序输出
for f in range(np.array(train_x).shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

for i in range(len(index_y)):
    index_1 = feat_labels[indices].tolist().index(index_y[i])
    col[i][1] = importances[indices][index_1]





##########################Qst############################
df_y = data['Qst']
df_X= data.drop(columns=['Qst','No','ΔW','-<ΔadsH>','MaxER','MinER','ANC','APC','Free_energy_ave'])
feat_labels = df_X.columns[0:]


# apply log transform to KH column and then normalize all columns
df_X['KH'] = log_transformer.transform(df_X[['KH']])
df_X.iloc[:, :] = scaler.fit_transform(df_X.iloc[:, :])

# reassign all columns (including KH column) to df_X
df_X = pd.DataFrame(df_X, columns=feat_labels)


df_y=df_y.values.tolist()
df_X=df_X.values.tolist()



df_X1 = []
df_y1 = []
df_X2 = []
df_y2 = []
for i in range(len(df_X)):
    if df_X[i][8]==1:
        df_X1.append(df_X[i])
        df_y1.append(df_y[i])
    else:
        df_X2.append(df_X[i])
        df_y2.append(df_y[i])

rest_data=[]
rest_data2=[]
for l in df_X1:
    #删除一行中第c列的值
    rest_l=l[:8]
    rest_l.extend(l[9:])
    #将删除后的结果加入结果数组
    rest_data.append(rest_l)
for l in df_X2:
    #删除一行中第c列的值
    rest_l=l[:8]
    rest_l.extend(l[9:])
    #将删除后的结果加入结果数组
    rest_data2.append(rest_l)
df_X1 = rest_data
df_X2 = rest_data2

# 切分数据集
train_x1, test_x1, train_y1, test_y1 = train_test_split(df_X1, df_y1, train_size=0.8, random_state=24)
train_x2, test_x2, train_y2, test_y2 = train_test_split(df_X2, df_y2, train_size=0.8, random_state=24)

train_x = train_x1 + train_x2
train_y = train_y1 + train_y2
test_x = test_x1 + test_x2
test_y = test_y1 + test_y2


model = RandomForestRegressor(n_estimators=150,min_samples_split=4,min_samples_leaf=2,max_features=0.940, max_depth=36)
model.fit(train_x, train_y)
r2 = model.score(test_x, test_y)
r2_train = model.score(train_x, train_y)
print('\nQed:\ntrain:', r2_train)
print('test: ', r2)
muti_score(model)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1] #[::-1]表示将各指标按权重大小进行排序输出
for f in range(np.array(train_x).shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


for i in range(len(index_y)):
    index_1 = feat_labels[indices].tolist().index(index_y[i])
    col[i][2] = importances[indices][index_1]



df = pd.DataFrame(col, 
                  index=['LCD','VF',chr(961),'K$\mathrm{_{H}}$','V$\mathrm{_{a}}$','C$\mathrm{_{ratio}}$','A'+chr(949),'A'+chr(963)],
                  columns=['$\Delta$W','<$\Delta$$\mathrm{_{ads}}$H>','Q$\mathrm{_{ed}}$'])

fig = plt.figure(figsize=(11, 8), dpi=80)
ax = fig.add_subplot()
#定义横纵坐标的刻度


yLabel = feat_labels[indices].tolist()
xLabel = ['$\Delta$W','-<$\Delta$$\mathrm{_{ads}}$H>','$\mathrm{Q_{ed}}$']

ax.set_yticks(range(len(yLabel)))
ax.set_yticklabels(yLabel)
ax.set_xticks(range(len(xLabel)))
ax.set_xticklabels(xLabel)


sns.heatmap(data=df,cmap = 'RdBu_r' )


# 添加颜色条
cbar = ax.collections[-1].colorbar
cbar.ax.tick_params(labelsize=20)
cbar.ax.yaxis.set_label_position('right') 
cbar.set_label('Relative importance')

fig.savefig('importance.pdf', format='pdf', dpi=600)
fig.savefig('importance.png', format='png', dpi=600)
plt.show()
   


