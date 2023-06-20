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
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import shap

import warnings
warnings.filterwarnings("ignore")


plt.rc('font', size=13,family= 'Arial')

data = pd.read_excel("predict.xlsx")
df_y = data[['Qed']]
#df_X= data.drop(columns=['Qst','No','ΔW','-<ΔadsH>','MaxER','MinER','ANC','APC','Free_energy_ave'])
df_X = data.loc[:,['LCD','VF','density','KH','Va','C_ratio','Aε','Aσ']]
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


train_x = df_X[:-40]
train_y = df_y[:-40]

test_x = df_X[-40:]
test_y = df_y[-40:]




#####################SVR算法####################
model =SVR(kernel='rbf', gamma=1.0, C= 890, epsilon=0.54)
model.fit(train_x, train_y)
r2_1 = model.score(test_x, test_y)
#print('\nSVR：', r2)
actual = [i[0] for i in test_y]
predictions = model.predict(test_x)



#输出csv内容
ranks = np.argsort(np.argsort(predictions)) 

actual_ranks = np.argsort(np.argsort(actual)) 

Qed = list(zip(actual, actual_ranks,predictions,ranks))
name = ['Actual','actual_ranks','SVR_pre','SVR_ranks']

datas1=pd.DataFrame(columns=name, data=Qed)


#####################MLP算法####################
model = MLPRegressor(hidden_layer_sizes=(300,300,300), ## 隐藏层的神经元个数
                    activation='relu', 
                    solver='adam', 
                    alpha=0.0001,   ## L2惩罚参数
                    max_iter=200, 
                    random_state=123,
                   )

model = model.fit(train_x, train_y)
r2_2 = model.score(test_x, test_y)
#print('\nMLP:', r2)

predictions = model.predict(test_x)

#输出csv内容
ranks = np.argsort(np.argsort(predictions)) 
Qed = list(zip(predictions,ranks))
name = ['MLP_pre','MLP_ranks']


datas2=pd.DataFrame(columns=name, data=Qed)





#####################DT算法####################
model =  DecisionTreeRegressor(min_samples_split=9, max_features=0.925,max_depth=90, min_samples_leaf=3)
model.fit(train_x, train_y)
r2_3 = model.score(test_x, test_y)
#print('\nDT:', r2)

predictions = model.predict(test_x)
#输出csv内容
ranks = np.argsort(np.argsort(predictions)) 
Qed = list(zip(predictions,ranks))
name = ['DT_pre','DT_ranks']
datas3=pd.DataFrame(columns=name, data=Qed)



#####################HGBR算法####################
model =  HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=36 ,max_depth=24,min_samples_leaf=37,max_bins=106)
model.fit(train_x, train_y)
r2_4 = model.score(test_x, test_y)
#print('\nHGBR:', r2)

predictions = model.predict(test_x)
#输出csv内容
ranks = np.argsort(np.argsort(predictions)) 
Qed = list(zip(predictions,ranks))
name = ['HGBR_pre','HGBR_ranks']
datas4=pd.DataFrame(columns=name, data=Qed)





#####################GBR算法####################
model = GradientBoostingRegressor(max_features=0.959,n_estimators=108, max_depth=17 ,min_samples_leaf=27,min_samples_split=6)
model.fit(train_x, train_y)
r2_5 = model.score(test_x, test_y)
#print('\nGBR:', r2)

predictions = model.predict(test_x)
#输出csv内容
ranks = np.argsort(np.argsort(predictions)) 
Qed = list(zip(predictions,ranks))
name = ['GBR_pre','GBR_ranks']
datas5=pd.DataFrame(columns=name, data=Qed)





#####################RF算法####################
model = RandomForestRegressor(n_estimators=150,min_samples_split=4,min_samples_leaf=2,max_features=0.940, max_depth=36)
model.fit(train_x, train_y)
r2_6 = model.score(test_x, test_y)
#print('\nRF:', r2)

predictions = model.predict(test_x)
#输出csv内容
ranks = np.argsort(np.argsort(predictions)) 
Qed = list(zip(predictions,ranks))
name = ['RF_pre','RF_ranks']
datas6=pd.DataFrame(columns=name, data=Qed)







# 读取CSV文件
df = pd.read_csv('predict_Qed.csv')
# 选择"actual"和"DT_pre"两列数据
actual_col = df['Actual']
RF_pre_col = df['RF_pre']
GBR_pre_col = df['GBR_pre']
SVR_pre_col = df['SVR_pre']
MLP_pre_col = df['MLP_pre']
DT_pre_col = df['DT_pre']
HGBR_pre_col = df['HGBR_pre']





print('\nSVR:'+ str(format(metrics.r2_score(actual_col,SVR_pre_col),"0.6f")) )
print('\nMLP:'+ str(format(metrics.r2_score(actual_col,MLP_pre_col),"0.6f")) )
print('\nDT:'+ str(format(metrics.r2_score(actual_col,DT_pre_col),"0.6f")) )
print('\nHGBR:'+ str(format(metrics.r2_score(actual_col,HGBR_pre_col),"0.6f")))
print('\nGBR:'+ str(format(metrics.r2_score(actual_col,GBR_pre_col),"0.6f"))) 
print('\nRF:'+ str(format(metrics.r2_score(actual_col,RF_pre_col),"0.6f")))





actual_col=actual_col.values.tolist()
SVR_pre_col=SVR_pre_col.values.tolist()
MLP_pre_col=MLP_pre_col.values.tolist()
DT_pre_col=DT_pre_col.values.tolist()
GBR_pre_col=GBR_pre_col.values.tolist()
RF_pre_col=RF_pre_col.values.tolist()
HGBR_pre_col=HGBR_pre_col.values.tolist()


R2_1  = metrics.r2_score(actual_col, SVR_pre_col)
R2_2  = metrics.r2_score(actual_col, MLP_pre_col)
R2_3  = metrics.r2_score(actual_col, DT_pre_col)
R2_4  = metrics.r2_score(actual_col, HGBR_pre_col)
R2_5  = metrics.r2_score(actual_col, GBR_pre_col)
R2_6  = metrics.r2_score(actual_col, RF_pre_col)

if r2_1>=R2_1 or r2_2>=R2_2 or r2_3>=R2_3  or r2_4>=R2_4  or r2_5>=R2_5  or r2_6>=R2_6:
    datas = pd.concat([datas1,datas2,datas3,datas4,datas5,datas6],axis = 1)
    datas.to_csv("predict_Qed.csv") 




