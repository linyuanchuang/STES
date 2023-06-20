import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import shap


plt.rc('font', size=20,family= 'Arial')

df = pd.read_csv('mofs_cofs_result_Qed.csv')
# 选择"actual"和"DT_pre"两列数据
actual_col = df['Actual']
SVR_pre_col = df['SVR_pre']
RF_pre_col = df['RF_pre']
DT_pre_col = df['DT_pre']
MLP_pre_col = df['MLP_pre']
GBR_pre_col = df['GBR_pre']
HGBR_pre_col = df['HGBR_pre']

actual_ranks = df['actual_ranks']
SVR_ranks = df['SVR_ranks']
RF_ranks = df['RF_ranks']
DT_ranks = df['DT_ranks']
MLP_ranks = df['MLP_ranks']
GBR_ranks = df['GBR_ranks']
HGBR_ranks = df['HGBR_ranks']






colormap = plt.get_cmap("jet")  # 色带
radius = 15  # 半径
radius_2 = 25 
marker_size = 100
title_font = {'family': 'Arial',
        'size': 20}  


def shap_model(model,feature_x,dataframe_x):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_x)  # 传入特征矩阵X，计算SHAP值
    shap.summary_plot(shap_values, dataframe_x)  
        
        
def density_calc(x, y, radius):
    '''
    散点密度计算（以便给散点图中的散点密度进行颜色渲染）
    :param x:
    :param y:
    :param radius:
    :return:  数据密度
    '''
    res = np.empty(len(x), dtype=np.float32)
    for i in range(len(x)):
        res[i] = np.sum((x > (x[i] - radius)) & (x < (x[i] + radius))
                        & (y > (y[i] - radius)) & (y < (y[i] + radius)))
    return res

#5阶交叉验证
def muti_score(model):
    warnings.filterwarnings('ignore')
    r2 = cross_val_score(model, train_x, train_y, scoring='r2', cv=10)
    print("validation:",r2.mean())
'''
model_name=["rbf_svr","mlr","gbm","rf","hgbr"]
for name in model_name:
    model=eval(name)
    print(name)
    muti_score(model)
'''

#####################SVR算法####################


#################################实际值绘图####################################
fig1 = plt.figure(figsize=(20, 12), dpi=80)
ax = fig1.add_subplot(2,3,1)
plt.axis([0, 800, 0, 800])

model_col = SVR_pre_col
ranks = SVR_ranks
R2 = "(a) R$^{2}$ = "+ str(format(metrics.r2_score(actual_col,model_col),"0.2f"))  
n=len(actual_ranks)
SROCC = 1-(6*sum((ranks-actual_ranks)**2))/(n*(n*n-1))

sc1 = ax.scatter(actual_col[0:1274],model_col[0:1274],label='training MOF',edgecolors='skyblue')
sc1.set_facecolor("none")
sc2 = ax.scatter(actual_col[1274:1488],model_col[1274:1488],label='training COF',edgecolors='orange')
sc2.set_facecolor("none")
sc3 = ax.scatter(actual_col[1488:1807],model_col[1488:1807],label='test MOF',c='skyblue',edgecolors='dimgray')
sc4 = ax.scatter(actual_col[1807:1861],model_col[1807:1861],label='test COF',c='orange',edgecolors='dimgray')


ax.plot((0, 800), (0, 800), transform=ax.transAxes, ls='--', c='k')
#ax.set_xlabel('Actual' ,fontsize =24)
ax.legend(loc='lower right')
ax.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$ (J/g)' ,fontsize=22)
ax.set_title('SVR' ,fontsize =24)
ax.text(15, 741.5, R2,fontsize= 20)
ax.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([])

######################################value_density_count####################################
fig2 = plt.figure(figsize=(20, 12), dpi=80)
ax3 = fig2.add_subplot(2,3,1)

Z1 = density_calc(actual_col, model_col, radius)
sc9 = ax3.scatter(actual_col, model_col, c=Z1, cmap=colormap, marker=".", s=marker_size,
            norm=colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax3.plot((0, 800), (0, 800), transform=ax3.transAxes, ls='--', c='k')

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc9,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)

ax3.text(15, 741.5, R2,fontsize= 20)
#ax1.set_xlabel(r'$\mathrm{COP_{C}}$',fontsize=22)
ax3.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
ax3.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax3.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax3.set_xticks(xticks)
ax3.set_yticks(yticks)
ax3.set_title('SVR' ,fontsize =24)
ax3.set_xticklabels([])



###################################value_counts##################################
fig3 = plt.figure(figsize=(20, 12), dpi=80)
ax12 = fig3.add_subplot(2,3,1)

data_imshow1 = [ ([0]*50) for i in range(0,50) ]
for i in range(len(model_col)):
    if actual_col[i]>0 and model_col[i]>0:
        xx = int(actual_col[i]/16)   
        yy = int(model_col[i]/16)
        data_imshow1[yy][xx] = data_imshow1[yy][xx] + 1    
#print(data_imshow1)
sc12 = ax12.imshow(data_imshow1,interpolation='none',origin='lower',cmap=colormap,extent=[0,800,0,800],aspect='auto',norm=matplotlib.colors.LogNorm(vmin=Z1.min(),vmax=0.5 * Z1.max()))
ax12.plot((0, 800), (0, 800), transform=ax12.transAxes, ls='--', c='k')

#axs.legend(loc = 'lower right',fontsize=22)
divider = make_axes_locatable(ax12)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc12,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)
#ax1.set_xlabel(r'$\mathrm{COP_{C}}$',fontsize=22)
ax12.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
ax12.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax12.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax12.set_xticks(xticks)
ax12.set_yticks(yticks)
ax12.set_title('SVR' ,fontsize =24)
ax12.set_xticklabels([])
ax12.text(15, 741.5, R2,fontsize= 20)


########################################SROCC绘图############################
fig4 = plt.figure(figsize=(20, 12), dpi=80)
ax = fig4.add_subplot(2,3,1)
plt.axis([0, 2000, 0, 2000])

sc1 = ax.scatter(actual_ranks[0:1274],ranks[0:1274],label='training MOF',edgecolors='skyblue')
sc1.set_facecolor("none")
sc2 = ax.scatter(actual_ranks[1274:1488],ranks[1274:1488],label='training COF',edgecolors='orange')
sc2.set_facecolor("none")
sc3 = ax.scatter(actual_ranks[1488:1807],ranks[1488:1807],label='test MOF',c='skyblue',edgecolors='dimgray')
sc4 = ax.scatter(actual_ranks[1807:1861],ranks[1807:1861],label='test COF',c='orange',edgecolors='dimgray')
ax.plot((0, 2000), (0, 2000), transform=ax.transAxes, ls='--', c='k')
#ax.set_xlabel('Actual' ,fontsize =24)

ax.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.legend(loc='lower right')
ax.set_ylabel('Predicted ranks of '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax.set_title('SVR' ,fontsize =24)
ax.set_xticklabels([])
SRCC = "(a) SRCC = "+ str(format(SROCC,"0.2f"))        
ax.text(37.5, 1853.5, SRCC,fontsize= 20)

######################################SROCC_density_count####################################
fig5 = plt.figure(figsize=(20, 12), dpi=80)
ax3 = fig5.add_subplot(2,3,1)
Z1 = density_calc(actual_ranks, ranks, radius_2)
sc9 = ax3.scatter(actual_ranks, ranks, c=Z1, cmap=colormap, marker=".", s=marker_size, norm=colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax3.plot((0, 2000), (0, 2000), transform=ax3.transAxes, ls='--', c='k')

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc9,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)

#ax1.set_xlabel(r'$\mathrm{COP_{C}}$',fontsize=22)
ax3.set_ylabel('Predicted ranks of '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax3.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax3.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax3.set_xticks(xticks)
ax3.set_yticks(yticks)
ax3.set_title('SVR' ,fontsize =24)
ax3.set_xticklabels([])
ax3.text(37.5, 1853.5, SRCC,fontsize= 20)

###################################SROCC_counts##################################
fig6 = plt.figure(figsize=(20, 12), dpi=80)

ax12 = fig6.add_subplot(2,3,1)
data_imshow1 = [ ([0]*50) for i in range(0,50) ]
for i in range(len(actual_ranks)):
   xx = int(actual_ranks[i]/40)   
   yy = int(ranks[i]/40)
   data_imshow1[yy][xx] = data_imshow1[yy][xx] + 1    
#print(data_imshow1)
sc12 = ax12.imshow(data_imshow1,interpolation='none',origin='lower',cmap=colormap,extent=[0,2000,0,2000],aspect='auto',norm=matplotlib.colors.LogNorm(vmin=Z1.min(),vmax=0.5 * Z1.max()))
ax12.plot((0, 2000), (0, 2000), transform=ax12.transAxes, ls='--', c='k')

#axs.legend(loc = 'lower right',fontsize=22)
divider = make_axes_locatable(ax12)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc12,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)
#ax1.set_xlabel(r'$\mathrm{COP_{C}}$',fontsize=22)
ax12.set_ylabel('Predicted ranks of '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax12.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax12.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax12.set_xticks(xticks)
ax12.set_yticks(yticks)
ax12.set_title('SVR' ,fontsize =24)
ax12.set_xticklabels([])
ax12.text(37.5, 1853.5, SRCC,fontsize= 20)





#####################MLP算法####################

ax = fig1.add_subplot(2,3,2)
plt.axis([0, 800, 0, 800])

model_col = MLP_pre_col
ranks = MLP_ranks
SROCC = 1-(6*sum((ranks-actual_ranks)**2))/(n*(n*n-1))
R2 = "(b) R$^{2}$ = "+ str(format(metrics.r2_score(actual_col,model_col),"0.2f"))  
sc1 = ax.scatter(actual_col[0:1274],model_col[0:1274],label='training MOF',edgecolors='skyblue')
sc1.set_facecolor("none")
sc2 = ax.scatter(actual_col[1274:1488],model_col[1274:1488],label='training COF',edgecolors='orange')
sc2.set_facecolor("none")
sc3 = ax.scatter(actual_col[1488:1807],model_col[1488:1807],label='test MOF',c='skyblue',edgecolors='dimgray')
sc4 = ax.scatter(actual_col[1807:1861],model_col[1807:1861],label='test COF',c='orange',edgecolors='dimgray')

ax.plot((0, 800), (0, 800), transform=ax.transAxes, ls='--', c='k')
ax.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_title('MLP' ,fontsize =24)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.text(15, 741.5, R2,fontsize= 20)



ax3 = fig2.add_subplot(2,3,2)
Z1 = density_calc(actual_col, model_col, radius)
sc9 = ax3.scatter(actual_col, model_col, c=Z1, cmap=colormap, marker=".", s=marker_size,
            norm=colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax3.plot((0, 800), (0, 800), transform=ax3.transAxes, ls='--', c='k')

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc9,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)

#ax1.set_xlabel(r'$\mathrm{COP_{C}}$',fontsize=22)
#ax3.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax3.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax3.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax3.set_xticks(xticks)
ax3.set_yticks(yticks)
ax3.set_title('MLP' ,fontsize =24)

ax3.set_xticklabels([])
ax3.set_yticklabels([]) 
ax3.text(15, 741.5, R2,fontsize= 20)

ax2 = fig3.add_subplot(2,3,2)
data_imshow1 = [ ([0]*50) for i in range(0,50) ]
for i in range(len(model_col)):
    if actual_col[i]>0 and model_col[i]>0:
        xx = int(actual_col[i]/16)   
        yy = int(model_col[i]/16)
        data_imshow1[yy][xx] = data_imshow1[yy][xx] + 1    
#print(data_imshow1)
sc7 = ax2.imshow(data_imshow1,interpolation='none',origin='lower',cmap=colormap,extent=[0,800,0,800],aspect='auto',norm=matplotlib.colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax2.plot((0, 800), (0, 800), transform=ax2.transAxes, ls='--', c='k')

#axs.legend(loc = 'lower right',fontsize=22)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc7,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)
#ax1.set_xlabel(r'$\mathrm{COP_{C}}$',fontsize=22)
#ax2.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax2.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax2.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
ax2.set_title('MLP' ,fontsize =24)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.text(15, 741.5, R2,fontsize= 20)


########################################SROCC绘图############################
ax = fig4.add_subplot(2,3,2)
plt.axis([0, 2000, 0, 2000])


sc1 = ax.scatter(actual_ranks[0:1274],ranks[0:1274],label='training MOF',edgecolors='skyblue')
sc1.set_facecolor("none")
sc2 = ax.scatter(actual_ranks[1274:1488],ranks[1274:1488],label='training COF',edgecolors='orange')
sc2.set_facecolor("none")
sc3 = ax.scatter(actual_ranks[1488:1807],ranks[1488:1807],label='test MOF',c='skyblue',edgecolors='dimgray')
sc4 = ax.scatter(actual_ranks[1807:1861],ranks[1807:1861],label='test COF',c='orange',edgecolors='dimgray')
ax.plot((0, 2000), (0, 2000), transform=ax.transAxes, ls='--', c='k')
#ax.set_xlabel('Actual' ,fontsize=24)
#ax.legend(loc='lower right')
#ax.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax.set_title('MLP' ,fontsize =24)
ax.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([])
ax.set_yticklabels([])
SRCC = "(b) SRCC = "+ str(format(SROCC,"0.2f"))    
ax.text(37.5, 1853.5, SRCC,fontsize= 20)

######################################SROCC_density_count####################################
ax3 = fig5.add_subplot(2,3,2)
Z1 = density_calc(actual_ranks, ranks, radius_2)
sc9 = ax3.scatter(actual_ranks, ranks, c=Z1, cmap=colormap, marker=".", s=marker_size,
            norm=colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax3.plot((0, 2000), (0, 2000), transform=ax3.transAxes, ls='--', c='k')

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc9,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)
#ax1.set_xlabel(r'$\mathrm{COP_{C}}$',fontsize=22)
#ax3.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax3.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax3.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax3.set_xticks(xticks)
ax3.set_yticks(yticks)
ax3.set_title('MLP' ,fontsize =24)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.text(37.5, 1853.5, SRCC,fontsize= 20)

###################################SROCC_counts##################################
ax2 = fig6.add_subplot(2,3,2)

data_imshow1 = [ ([0]*50) for i in range(0,50) ]
for i in range(len(actual_ranks)):
   xx = int(actual_ranks[i]/40)   
   yy = int(ranks[i]/40)
   data_imshow1[yy][xx] = data_imshow1[yy][xx] + 1    
#print(data_imshow1)
sc7 = ax2.imshow(data_imshow1,interpolation='none',origin='lower',cmap=colormap,extent=[0,2000,0,2000],aspect='auto',norm=matplotlib.colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax2.plot((0, 2000), (0, 2000), transform=ax2.transAxes, ls='--', c='k')

#axs.legend(loc = 'lower right',fontsize=22)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc7,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)
#ax1.set_xlabel(r'$\mathrm{COP_{C}}$',fontsize=22)
#ax2.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax2.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax2.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
ax2.set_title('MLP' ,fontsize =24)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.text(37.5, 1853.5, SRCC,fontsize= 20)
















#####################DT算法####################

ax = fig1.add_subplot(2,3,3)
plt.axis([0, 800, 0, 800])


model_col = DT_pre_col
ranks = DT_ranks

SROCC = 1-(6*sum((ranks-actual_ranks)**2))/(n*(n*n-1))
R2 = "(c) R$^{2}$ = "+ str(format(metrics.r2_score(actual_col,model_col),"0.2f"))  
sc1 = ax.scatter(actual_col[0:1274],model_col[0:1274],label='training MOF',edgecolors='skyblue')
sc1.set_facecolor("none")
sc2 = ax.scatter(actual_col[1274:1488],model_col[1274:1488],label='training COF',edgecolors='orange')
sc2.set_facecolor("none")
sc3 = ax.scatter(actual_col[1488:1807],model_col[1488:1807],label='test MOF',c='skyblue',edgecolors='dimgray')
sc4 = ax.scatter(actual_col[1807:1861],model_col[1807:1861],label='test COF',c='orange',edgecolors='dimgray')

ax.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.plot((0, 800), (0, 800), transform=ax.transAxes, ls='--', c='k')
ax.set_title('DT' ,fontsize =24)
ax.text(15, 741.5, R2,fontsize= 20)


ax3 = fig2.add_subplot(2,3,3)
Z1 = density_calc(actual_col, model_col, radius)
sc9 = ax3.scatter(actual_col, model_col, c=Z1, cmap=colormap, marker=".", s=marker_size,
            norm=colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax3.plot((0, 800), (0, 800), transform=ax3.transAxes, ls='--', c='k')

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc9,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)

#ax1.set_xlabel(r'$\mathrm{COP_{C}}$',fontsize=22)
#ax3.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax3.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax3.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax3.set_xticks(xticks)
ax3.set_yticks(yticks)
ax3.set_title('DT' ,fontsize =24)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.text(15, 741.5, R2,fontsize= 20)

ax2 = fig3.add_subplot(2,3,3)
data_imshow1 = [ ([0]*50) for i in range(0,50) ]
for i in range(len(model_col)):
    if actual_col[i]>0 and model_col[i]>0:
        xx = int(actual_col[i]/16)   
        yy = int(model_col[i]/16)
        data_imshow1[yy][xx] = data_imshow1[yy][xx] + 1    
#print(data_imshow1)
sc7 = ax2.imshow(data_imshow1,interpolation='none',origin='lower',cmap=colormap,extent=[0,800,0,800],aspect='auto',norm=matplotlib.colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax2.plot((0, 800), (0, 800), transform=ax2.transAxes, ls='--', c='k')

#axs.legend(loc = 'lower right',fontsize=22)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc7,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)
#ax1.set_xlabel(r'$\mathrm{COP_{C}}$',fontsize=22)
#ax2.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax2.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax2.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
ax2.set_title('DT' ,fontsize =24)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.text(15, 741.5, R2,fontsize= 20)



########################################SROCC绘图############################
ax = fig4.add_subplot(2,3,3)
plt.axis([0, 2000, 0, 2000])

sc1 = ax.scatter(actual_ranks[0:1274],ranks[0:1274],label='training MOF',edgecolors='skyblue')
sc1.set_facecolor("none")
sc2 = ax.scatter(actual_ranks[1274:1488],ranks[1274:1488],label='training COF',edgecolors='orange')
sc2.set_facecolor("none")
sc3 = ax.scatter(actual_ranks[1488:1807],ranks[1488:1807],label='test MOF',c='skyblue',edgecolors='dimgray')
sc4 = ax.scatter(actual_ranks[1807:1861],ranks[1807:1861],label='test COF',c='orange',edgecolors='dimgray')
ax.plot((0, 2000), (0, 2000), transform=ax.transAxes, ls='--', c='k')
#ax.set_xlabel('Actual' ,fontsize=24)
#ax.legend(loc='lower right')
#ax.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax.set_title('DT' ,fontsize =24)
ax.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([])
ax.set_yticklabels([])
SRCC = "(c) SRCC = "+ str(format(SROCC,"0.2f"))    
ax.text(37.5, 1853.5, SRCC,fontsize= 20)

######################################SROCC_density_count####################################
ax3 = fig5.add_subplot(2,3,3)
Z1 = density_calc(actual_ranks, ranks, radius_2)
sc9 = ax3.scatter(actual_ranks, ranks, c=Z1, cmap=colormap, marker=".", s=marker_size,
            norm=colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax3.plot((0, 2000), (0, 2000), transform=ax3.transAxes, ls='--', c='k')

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc9,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)

#ax1.set_xlabel(r'$\mathrm{COP_{C}}$',fontsize=22)
#ax3.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax3.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax3.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax3.set_xticks(xticks)
ax3.set_yticks(yticks)
ax3.set_title('DT' ,fontsize =24)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.text(37.5, 1853.5, SRCC,fontsize= 20)

###################################SROCC_counts##################################
ax2 = fig6.add_subplot(2,3,3)

data_imshow1 = [ ([0]*50) for i in range(0,50) ]
for i in range(len(actual_ranks)):
   xx = int(actual_ranks[i]/40)   
   yy = int(ranks[i]/40)
   data_imshow1[yy][xx] = data_imshow1[yy][xx] + 1    
#print(data_imshow1)
sc7 = ax2.imshow(data_imshow1,interpolation='none',origin='lower',cmap=colormap,extent=[0,2000,0,2000],aspect='auto',norm=matplotlib.colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax2.plot((0, 2000), (0, 2000), transform=ax2.transAxes, ls='--', c='k')

#axs.legend(loc = 'lower right',fontsize=22)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc7,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)
#ax1.set_xlabel(r'$\mathrm{COP_{C}}$',fontsize=22)
#ax2.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax2.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax2.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
ax2.set_title('DT' ,fontsize =24)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.text(37.5, 1853.5, SRCC,fontsize= 20)











#####################HGBR算法####################

ax = fig1.add_subplot(2,3,4)
plt.axis([0, 800, 0, 800])

model_col = HGBR_pre_col
ranks = HGBR_ranks

SROCC = 1-(6*sum((ranks-actual_ranks)**2))/(n*(n*n-1))
R2 = "(d) R$^{2}$ = "+ str(format(metrics.r2_score(actual_col,model_col),"0.2f"))  
sc1 = ax.scatter(actual_col[0:1274],model_col[0:1274],label='training MOF',edgecolors='skyblue')
sc1.set_facecolor("none")
sc2 = ax.scatter(actual_col[1274:1488],model_col[1274:1488],label='training COF',edgecolors='orange')
sc2.set_facecolor("none")
sc3 = ax.scatter(actual_col[1488:1807],model_col[1488:1807],label='test MOF',c='skyblue',edgecolors='dimgray')
sc4 = ax.scatter(actual_col[1807:1861],model_col[1807:1861],label='test COF',c='orange',edgecolors='dimgray')


ax.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.text(15, 741.5, R2,fontsize= 20)
ax.plot((0, 800), (0, 800), transform=ax.transAxes, ls='--', c='k')
#ax.set_ylabel('Predict ' ,fontsize=22)
ax.set_title('HGBR' ,fontsize =24)
ax.set_xlabel(r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
ax.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)


ax3 = fig2.add_subplot(2,3,4)
Z1 = density_calc(actual_col, model_col, radius)
sc9 = ax3.scatter(actual_col, model_col, c=Z1, cmap=colormap, marker=".", s=marker_size,
            norm=colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax3.plot((0, 800), (0, 800), transform=ax3.transAxes, ls='--', c='k')

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc9,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)
ax3.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
ax3.set_xlabel(r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
ax3.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax3.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax3.set_xticks(xticks)
ax3.set_yticks(yticks)
ax3.set_title('HGBR' ,fontsize =24)
ax3.text(15, 741.5, R2,fontsize= 20)


ax2 = fig3.add_subplot(2,3,4)
data_imshow1 = [ ([0]*50) for i in range(0,50) ]
for i in range(len(model_col)):
    if actual_col[i]>0 and model_col[i]>0:
        xx = int(actual_col[i]/16)   
        yy = int(model_col[i]/16)
        data_imshow1[yy][xx] = data_imshow1[yy][xx] + 1    
#print(data_imshow1)
sc7 = ax2.imshow(data_imshow1,interpolation='none',origin='lower',cmap=colormap,extent=[0,800,0,800],aspect='auto',norm=matplotlib.colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax2.plot((0, 800), (0, 800), transform=ax2.transAxes, ls='--', c='k')

#axs.legend(loc = 'lower right',fontsize=22)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc7,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)
ax2.set_xlabel(r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
ax2.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
ax2.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax2.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
ax2.set_title('HGBR' ,fontsize =24)
ax2.text(15, 741.5, R2,fontsize= 20)


########################################SROCC绘图############################
ax = fig4.add_subplot(2,3,4)
plt.axis([0, 2000, 0, 2000])

sc1 = ax.scatter(actual_ranks[0:1274],ranks[0:1274],label='training MOF',edgecolors='skyblue')
sc1.set_facecolor("none")
sc2 = ax.scatter(actual_ranks[1274:1488],ranks[1274:1488],label='training COF',edgecolors='orange')
sc2.set_facecolor("none")
sc3 = ax.scatter(actual_ranks[1488:1807],ranks[1488:1807],label='test MOF',c='skyblue',edgecolors='dimgray')
sc4 = ax.scatter(actual_ranks[1807:1861],ranks[1807:1861],label='test COF',c='orange',edgecolors='dimgray')
ax.plot((0, 2000), (0, 2000), transform=ax.transAxes, ls='--', c='k')
#ax.set_xlabel('Actual' ,fontsize=24)
#ax.legend(loc='lower right')
ax.set_xlabel(r'Ranks of $\mathrm{Q_{ed}}$',fontsize=22)
ax.set_ylabel('Predicted ranks of '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax.set_title('HGBR' ,fontsize =24)
ax.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
SRCC = "(d) SRCC = "+ str(format(SROCC,"0.2f"))    
ax.text(37.5, 1853.5, SRCC,fontsize= 20)

######################################SROCC_density_count####################################
ax3 = fig5.add_subplot(2,3,4)
Z1 = density_calc(actual_ranks, ranks, radius_2)
sc9 = ax3.scatter(actual_ranks, ranks, c=Z1, cmap=colormap, marker=".", s=marker_size,
            norm=colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax3.plot((0, 2000), (0, 2000), transform=ax3.transAxes, ls='--', c='k')

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc9,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)

ax3.set_xlabel(r'Ranks of $\mathrm{Q_{ed}}$',fontsize=22)
ax3.set_ylabel('Predicted ranks of '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax3.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax3.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax3.set_xticks(xticks)
ax3.set_yticks(yticks)
ax3.set_title('HGBR' ,fontsize =24)
ax3.text(37.5, 1853.5, SRCC,fontsize= 20)



###################################SROCC_counts##################################
ax2 = fig6.add_subplot(2,3,4)

data_imshow1 = [ ([0]*50) for i in range(0,50) ]
for i in range(len(actual_ranks)):
   xx = int(actual_ranks[i]/40)   
   yy = int(ranks[i]/40)
   data_imshow1[yy][xx] = data_imshow1[yy][xx] + 1    
#print(data_imshow1)
sc7 = ax2.imshow(data_imshow1,interpolation='none',origin='lower',cmap=colormap,extent=[0,2000,0,2000],aspect='auto',norm=matplotlib.colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax2.plot((0, 2000), (0, 2000), transform=ax2.transAxes, ls='--', c='k')

#axs.legend(loc = 'lower right',fontsize=22)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc7,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)
ax2.set_xlabel(r'Ranks of $\mathrm{Q_{ed}}$',fontsize=22)
ax2.set_ylabel('Predicted ranks of '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax2.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax2.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
ax2.set_title('HGBR' ,fontsize =24)
ax2.text(37.5, 1853.5, SRCC,fontsize= 20)






#####################GBR算法####################

ax = fig1.add_subplot(2,3,5)
plt.axis([0, 800, 0, 800])


model_col = GBR_pre_col
ranks = GBR_ranks
SROCC = 1-(6*sum((ranks-actual_ranks)**2))/(n*(n*n-1))
R2 = "(e) R$^{2}$ = "+ str(format(metrics.r2_score(actual_col,model_col),"0.2f"))  
sc1 = ax.scatter(actual_col[0:1274],model_col[0:1274],label='training MOF',edgecolors='skyblue')
sc1.set_facecolor("none")
sc2 = ax.scatter(actual_col[1274:1488],model_col[1274:1488],label='training COF',edgecolors='orange')
sc2.set_facecolor("none")
sc3 = ax.scatter(actual_col[1488:1807],model_col[1488:1807],label='test MOF',c='skyblue',edgecolors='dimgray')
sc4 = ax.scatter(actual_col[1807:1861],model_col[1807:1861],label='test COF',c='orange',edgecolors='dimgray')


ax.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.text(15, 741.5, R2,fontsize= 20)
ax.plot((0, 800), (0, 800), transform=ax.transAxes, ls='--', c='k')
ax.set_xlabel(r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
ax.set_title('GBR' ,fontsize =24)
ax.set_yticklabels([])



ax3 = fig2.add_subplot(2,3,5)
Z1 = density_calc(actual_col, model_col, radius)
sc9 = ax3.scatter(actual_col, model_col, c=Z1, cmap=colormap, marker=".", s=marker_size,
            norm=colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax3.plot((0, 800), (0, 800), transform=ax3.transAxes, ls='--', c='k')

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc9,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)

ax3.set_xlabel(r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
ax3.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax3.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax3.set_xticks(xticks)
ax3.set_yticks(yticks)
ax3.set_title('GBR' ,fontsize =24)
ax3.text(15, 741.5, R2,fontsize= 20)
ax3.set_yticklabels([])


ax2 = fig3.add_subplot(2,3,5)
data_imshow1 = [ ([0]*50) for i in range(0,50) ]
for i in range(len(model_col)):
    if actual_col[i]>0 and model_col[i]>0:
        xx = int(actual_col[i]/16)   
        yy = int(model_col[i]/16)
        data_imshow1[yy][xx] = data_imshow1[yy][xx] + 1    
#print(data_imshow1)
sc7 = ax2.imshow(data_imshow1,interpolation='none',origin='lower',cmap=colormap,extent=[0,800,0,800],aspect='auto',norm=matplotlib.colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax2.plot((0, 800), (0, 800), transform=ax2.transAxes, ls='--', c='k')

#axs.legend(loc = 'lower right',fontsize=22)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc7,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)
ax2.set_xlabel(r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
ax2.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax2.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
ax2.set_title('GBR' ,fontsize =24)
ax2.text(15, 741.5, R2,fontsize= 20)
ax2.set_yticklabels([])

########################################SROCC绘图############################
ax = fig4.add_subplot(2,3,5)
plt.axis([0, 2000, 0, 2000])

sc1 = ax.scatter(actual_ranks[0:1274],ranks[0:1274],label='training MOF',edgecolors='skyblue')
sc1.set_facecolor("none")
sc2 = ax.scatter(actual_ranks[1274:1488],ranks[1274:1488],label='training COF',edgecolors='orange')
sc2.set_facecolor("none")
sc3 = ax.scatter(actual_ranks[1488:1807],ranks[1488:1807],label='test MOF',c='skyblue',edgecolors='dimgray')
sc4 = ax.scatter(actual_ranks[1807:1861],ranks[1807:1861],label='test COF',c='orange',edgecolors='dimgray')
ax.plot((0, 2000), (0, 2000), transform=ax.transAxes, ls='--', c='k')
#ax.set_xlabel('Actual' ,fontsize =24)
#ax.legend(loc='lower right')
ax.set_xlabel(r'Ranks of $\mathrm{Q_{ed}}$',fontsize=22)
ax.set_title('GBR' ,fontsize =24)
ax.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
SRCC = "(e) SRCC = "+ str(format(SROCC,"0.2f"))    
ax.text(37.5, 1853.5, SRCC,fontsize= 20)
ax.set_yticklabels([])

######################################SROCC_density_count####################################
ax3 = fig5.add_subplot(2,3,5)
Z1 = density_calc(actual_ranks, ranks, radius_2)
sc9 = ax3.scatter(actual_ranks, ranks, c=Z1, cmap=colormap, marker=".", s=marker_size,
            norm=colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax3.plot((0, 2000), (0, 2000), transform=ax3.transAxes, ls='--', c='k')

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc9,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)

ax3.set_xlabel(r'Ranks of $\mathrm{Q_{ed}}$',fontsize=22)
ax3.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax3.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax3.set_xticks(xticks)
ax3.set_yticks(yticks)
ax3.set_title('GBR' ,fontsize =24)
ax3.text(37.5, 1853.5, SRCC,fontsize= 20)
ax3.set_yticklabels([])

###################################SROCC_counts##################################
ax2 = fig6.add_subplot(2,3,5)

data_imshow1 = [ ([0]*50) for i in range(0,50) ]
for i in range(len(actual_ranks)):
   xx = int(actual_ranks[i]/40)   
   yy = int(ranks[i]/40)
   data_imshow1[yy][xx] = data_imshow1[yy][xx] + 1    
#print(data_imshow1)
sc7 = ax2.imshow(data_imshow1,interpolation='none',origin='lower',cmap=colormap,extent=[0,2000,0,2000],aspect='auto',norm=matplotlib.colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax2.plot((0, 2000), (0, 2000), transform=ax2.transAxes, ls='--', c='k')

#axs.legend(loc = 'lower right',fontsize=22)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc7,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)
ax2.set_xlabel(r'Ranks of $\mathrm{Q_{ed}}$',fontsize=22)
ax2.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax2.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
ax2.set_title('GBR' ,fontsize =24)
ax2.text(37.5, 1853.5, SRCC,fontsize= 20)
ax2.set_yticklabels([])
















#####################RF算法####################

ax = fig1.add_subplot(2,3,6)
plt.axis([0, 800, 0, 800])


model_col = RF_pre_col
ranks = RF_ranks
SROCC = 1-(6*sum((ranks-actual_ranks)**2))/(n*(n*n-1))
R2 = "(f) R$^{2}$ = "+ str(format(metrics.r2_score(actual_col,model_col),"0.2f"))  
sc1 = ax.scatter(actual_col[0:1274],model_col[0:1274],label='training MOF',edgecolors='skyblue')
sc1.set_facecolor("none")
sc2 = ax.scatter(actual_col[1274:1488],model_col[1274:1488],label='training COF',edgecolors='orange')
sc2.set_facecolor("none")
sc3 = ax.scatter(actual_col[1488:1807],model_col[1488:1807],label='test MOF',c='skyblue',edgecolors='dimgray')
sc4 = ax.scatter(actual_col[1807:1861],model_col[1807:1861],label='test COF',c='orange',edgecolors='dimgray')



ax.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.plot((0, 800), (0, 800), transform=ax.transAxes, ls='--', c='k')
ax.set_xlabel(r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
#ax.set_ylabel('Predict ' ,fontsize=22)
ax.set_title('RF' ,fontsize =24)
ax.text(15, 741.5, R2,fontsize= 20)
ax.set_yticklabels([])


ax3 = fig2.add_subplot(2,3,6)
Z1 = density_calc(actual_col, model_col, radius)
sc9 = ax3.scatter(actual_col, model_col, c=Z1, cmap=colormap, marker=".", s=marker_size,
            norm=colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax3.plot((0, 800), (0, 800), transform=ax3.transAxes, ls='--', c='k')

divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc9,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)

ax3.set_xlabel(r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
ax3.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax3.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax3.set_xticks(xticks)
ax3.set_yticks(yticks)
ax3.set_title('RF' ,fontsize =24)
ax3.set_yticklabels([])
ax3.text(15, 741.5, R2,fontsize= 20)


ax2 = fig3.add_subplot(2,3,6)
data_imshow1 = [ ([0]*50) for i in range(0,50) ]
for i in range(len(model_col)):
    if actual_col[i]>0 and model_col[i]>0:
        xx = int(actual_col[i]/16)   
        yy = int(model_col[i]/16)
        data_imshow1[yy][xx] = data_imshow1[yy][xx] + 1    
#print(data_imshow1)
sc7 = ax2.imshow(data_imshow1,interpolation='none',origin='lower',cmap=colormap,extent=[0,800,0,800],aspect='auto',norm=matplotlib.colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax2.plot((0, 800), (0, 800), transform=ax2.transAxes, ls='--', c='k')

#axs.legend(loc = 'lower right',fontsize=22)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc7,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)
ax2.set_xlabel(r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
ax2.set_ylim(0,800)
yticks = np.linspace(0,800,9)
ax2.set_xlim(0,800)
xticks = np.linspace(0,800,9)
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
ax2.set_title('RF' ,fontsize =24)
ax2.set_yticklabels([])
ax2.text(15, 741.5, R2,fontsize= 20)


########################################SROCC绘图############################
ax = fig4.add_subplot(2,3,6)
plt.axis([0, 2000, 0, 2000])

sc1 = ax.scatter(actual_ranks[0:1274],ranks[0:1274],label='training MOF',edgecolors='skyblue')
sc1.set_facecolor("none")
sc2 = ax.scatter(actual_ranks[1274:1488],ranks[1274:1488],label='training COF',edgecolors='orange')
sc2.set_facecolor("none")
sc3 = ax.scatter(actual_ranks[1488:1807],ranks[1488:1807],label='test MOF',c='skyblue',edgecolors='dimgray')
sc4 = ax.scatter(actual_ranks[1807:1861],ranks[1807:1861],label='test COF',c='orange',edgecolors='dimgray')
ax.plot((0, 2000), (0, 2000), transform=ax.transAxes, ls='--', c='k')
ax.set_xlabel(r'Ranks of $\mathrm{Q_{ed}}$',fontsize=22)
ax.set_title('RF' ,fontsize =24)
ax.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_yticklabels([])
SRCC = "(f) SRCC = "+ str(format(SROCC,"0.2f"))    
ax.text(37.5, 1853.5, SRCC,fontsize= 20)



######################################SROCC_density_count####################################
ax8 = fig5.add_subplot(2,3,6)
Z1 = density_calc(actual_ranks, ranks, radius_2)
sc8 = ax8.scatter(actual_ranks, ranks, c=Z1, cmap=colormap, marker=".", s=marker_size,
            norm=colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax8.plot((0, 2000), (0, 2000), transform=ax8.transAxes, ls='--', c='k')

divider = make_axes_locatable(ax8)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc8,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)

ax8.set_xlabel(r'Ranks of $\mathrm{Q_{ed}}$',fontsize=22)
ax8.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax8.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax8.set_xticks(xticks)
ax8.set_yticks(yticks)
ax8.set_title('RF' ,fontsize =24)
ax8.set_yticklabels([])
ax8.text(37.5, 1853.5, SRCC,fontsize= 20)


###################################SROCC_counts##################################
ax2 = fig6.add_subplot(2,3,6)

data_imshow1 = [ ([0]*50) for i in range(0,50) ]
for i in range(len(actual_ranks)):
   xx = int(actual_ranks[i]/40)   
   yy = int(ranks[i]/40)
   data_imshow1[yy][xx] = data_imshow1[yy][xx] + 1    
#print(data_imshow1)
sc7 = ax2.imshow(data_imshow1,interpolation='none',origin='lower',cmap=colormap,extent=[0,2000,0,2000],aspect='auto',norm=matplotlib.colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
ax2.plot((0, 2000), (0, 2000), transform=ax2.transAxes, ls='--', c='k')

#axs.legend(loc = 'lower right',fontsize=22)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.08)
cbar1 = plt.colorbar(sc7,cax=cax)
cbar1.ax.set_title('Counts', y=1,fontsize=20,fontdict=title_font)
ax2.set_xlabel(r'Ranks of $\mathrm{Q_{ed}}$',fontsize=22)
ax2.set_ylim(0,2000)
yticks = np.linspace(0,2000,5)
ax2.set_xlim(0,2000)
xticks = np.linspace(0,2000,5)
ax2.set_xticks(xticks)
ax2.set_yticks(yticks)
ax2.set_title('RF' ,fontsize =24)
ax2.set_yticklabels([])
ax2.text(37.5, 1853.5, SRCC,fontsize= 20)




print('\nSVR: \ntrain: '+ str(format(metrics.r2_score(actual_col[0:1488],SVR_pre_col[0:1488]),"0.6f")) +'\ntest: '+ str(format(metrics.r2_score(actual_col[1488:1861],SVR_pre_col[1488:1861]),"0.6f"))+'\nall: '+ str(format(metrics.r2_score(actual_col,SVR_pre_col),"0.6f")))
print('\nMLP: \ntrain: '+ str(format(metrics.r2_score(actual_col[0:1488],MLP_pre_col[0:1488]),"0.6f")) +'\ntest: '+ str(format(metrics.r2_score(actual_col[1488:1861],MLP_pre_col[1488:1861]),"0.6f"))+'\nall: '+ str(format(metrics.r2_score(actual_col,MLP_pre_col),"0.6f")))
print('\nDT: \ntrain: '+ str(format(metrics.r2_score(actual_col[0:1488],DT_pre_col[0:1488]),"0.6f")) +'\ntest: '+ str(format(metrics.r2_score(actual_col[1488:1861],DT_pre_col[1488:1861]),"0.6f"))+'\nall: '+ str(format(metrics.r2_score(actual_col,DT_pre_col),"0.6f")))
print('\nHGBR: \ntrain: '+ str(format(metrics.r2_score(actual_col[0:1488],HGBR_pre_col[0:1488]),"0.6f")) +'\ntest: '+ str(format(metrics.r2_score(actual_col[1488:1861],HGBR_pre_col[1488:1861]),"0.6f"))+'\nall: '+ str(format(metrics.r2_score(actual_col,HGBR_pre_col),"0.6f")))
print('\nGBR: \ntrain: '+ str(format(metrics.r2_score(actual_col[0:1488],GBR_pre_col[0:1488]),"0.6f")) +'\ntest: '+ str(format(metrics.r2_score(actual_col[1488:1861],GBR_pre_col[1488:1861]),"0.6f"))+'\nall: '+ str(format(metrics.r2_score(actual_col,GBR_pre_col),"0.6f")))
print('\nRF: \ntrain: '+ str(format(metrics.r2_score(actual_col[0:1488],RF_pre_col[0:1488]),"0.6f")) +'\ntest: '+ str(format(metrics.r2_score(actual_col[1488:1861],RF_pre_col[1488:1861]),"0.6f"))+'\nall: '+ str(format(metrics.r2_score(actual_col,RF_pre_col),"0.6f")))


plt.show()
fig1.savefig('Figure1.png', format='png', dpi=600)
fig2.savefig('Figure2.png', format='png', dpi=600)
fig3.savefig('Figure3.png', format='png', dpi=600)
fig4.savefig('Figure4.png', format='png', dpi=600)
fig5.savefig('Figure5.png', format='png', dpi=600)
fig6.savefig('Figure6.png', format='png', dpi=600)

fig1.savefig('Figure1.pdf', format='pdf', dpi=600)
fig2.savefig('Figure2.pdf', format='pdf', dpi=600)
fig3.savefig('Figure3.pdf', format='pdf', dpi=600)
fig4.savefig('Figure4.pdf', format='pdf', dpi=600)
fig5.savefig('Figure5.pdf', format='pdf', dpi=600)
fig6.savefig('Figure6.pdf', format='pdf', dpi=600)

#'''


#"""