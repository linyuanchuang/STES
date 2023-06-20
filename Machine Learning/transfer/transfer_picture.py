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

df = pd.read_csv('predict_Qed.csv')
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

     
#####################SVR算法####################


#################################实际值绘图####################################
fig1 = plt.figure(figsize=(20, 12), dpi=80)
ax = fig1.add_subplot(2,3,1)
plt.axis([0, 400, 0, 400])

model_col = SVR_pre_col
ranks = SVR_ranks
n=len(actual_ranks)
SROCC = 1-(6*sum((ranks-actual_ranks)**2))/(n*(n*n-1))

R2 = "(a) R$^{2}$ = "+ str(format(metrics.r2_score(actual_col,model_col),"0.2f"))  

sc1 = ax.scatter(actual_col[0:20],model_col[0:20],label='MOF',edgecolors='skyblue')
sc2 = ax.scatter(actual_col[20:40],model_col[20:40],label='COF',edgecolors='orange')


ax.plot((0, 400), (0, 400), transform=ax.transAxes, ls='--', c='k')
#ax.set_xlabel('Actual' ,fontsize =24)
ax.legend(loc='lower right')
ax.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$ (J/g)' ,fontsize=22)
ax.set_title('SVR' ,fontsize =24)
ax.text(7.5, 370.75, R2,fontsize= 20)
ax.set_ylim(0,400)
yticks = np.linspace(0,400,9)
ax.set_xlim(0,400)
xticks = np.linspace(0,400,9)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([])


fig4 = plt.figure(figsize=(20, 12), dpi=80)
ax1 = fig4.add_subplot(2,3,1)
plt.axis([0, 40, 0, 40])

sc1 = ax1.scatter(actual_ranks[0:20],ranks[0:20],label='MOF',edgecolors='skyblue')
sc2 = ax1.scatter(actual_ranks[20:40],ranks[20:40],label='COF',edgecolors='orange')
ax1.plot((0, 40), (0, 40), transform=ax.transAxes, ls='--', c='k')
#ax1.set_xlabel('Actual' ,fontsize =20)

ax1.set_ylim(0,40)
yticks = np.linspace(0,40,9)
ax1.set_xlim(0,40)
xticks = np.linspace(0,40,9)
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
ax1.legend(loc='lower right')
ax1.set_ylabel('Predicted ranks of '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax1.set_title('SVR' ,fontsize =24)
ax1.set_xticklabels([])
SRCC = "(a) SRCC = "+ str(format(SROCC,"0.2f"))        
ax1.text(0.75, 37.075, SRCC,fontsize =20)






#####################MLP算法####################

ax = fig1.add_subplot(2,3,2)
plt.axis([0, 400, 0, 400])

model_col = MLP_pre_col
ranks = MLP_ranks
SROCC = 1-(6*sum((ranks-actual_ranks)**2))/(n*(n*n-1))

R2 = "(b) R$^{2}$ = "+ str(format(metrics.r2_score(actual_col,model_col),"0.2f"))  
sc1 = ax.scatter(actual_col[0:20],model_col[0:20],label='MOF',edgecolors='skyblue')
sc2 = ax.scatter(actual_col[20:40],model_col[20:40],label='COF',edgecolors='orange')



ax.plot((0, 400), (0, 400), transform=ax.transAxes, ls='--', c='k')
ax.set_ylim(0,400)
yticks = np.linspace(0,400,9)
ax.set_xlim(0,400)
xticks = np.linspace(0,400,9)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_title('MLP' ,fontsize =24)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.text(7.5, 370.75, R2,fontsize= 20)


ax1 = fig4.add_subplot(2,3,2)
plt.axis([0, 40, 0, 40])

sc1 = ax1.scatter(actual_ranks[0:20],ranks[0:20],label='MOF',edgecolors='skyblue')
sc2 = ax1.scatter(actual_ranks[20:40],ranks[20:40],label='COF',edgecolors='orange')

ax1.plot((0, 40), (0, 40), transform=ax.transAxes, ls='--', c='k')
#ax1.set_xlabel('Actual' ,fontsize =20)
#ax1.legend(loc='lower right')
#ax1.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$',fontsize=18)
ax1.set_title('MLP' ,fontsize =24)
ax1.set_ylim(0,40)
yticks = np.linspace(0,40,9)
ax1.set_xlim(0,40)
xticks = np.linspace(0,40,9)
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
SRCC = "(b) SRCC = "+ str(format(SROCC,"0.2f"))    
ax1.text(0.75, 37.075, SRCC,fontsize =20)



#####################DT算法####################

ax = fig1.add_subplot(2,3,3)
plt.axis([0, 400, 0, 400])


model_col = DT_pre_col
ranks = DT_ranks
SROCC = 1-(6*sum((ranks-actual_ranks)**2))/(n*(n*n-1))

R2 = "(c) R$^{2}$ = "+ str(format(metrics.r2_score(actual_col,model_col),"0.2f"))  
sc1 = ax.scatter(actual_col[0:20],model_col[0:20],label='MOF',edgecolors='skyblue')
sc2 = ax.scatter(actual_col[20:40],model_col[20:40],label='COF',edgecolors='orange')



ax.set_ylim(0,400)
yticks = np.linspace(0,400,9)
ax.set_xlim(0,400)
xticks = np.linspace(0,400,9)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.plot((0, 400), (0, 400), transform=ax.transAxes, ls='--', c='k')
ax.set_title('DT' ,fontsize =24)
ax.text(7.5, 370.75, R2,fontsize= 20)



ax1 = fig4.add_subplot(2,3,3)
plt.axis([0, 40, 0, 40])

sc1 = ax1.scatter(actual_ranks[0:20],ranks[0:20],label='MOF',edgecolors='skyblue')

sc2 = ax1.scatter(actual_ranks[20:40],ranks[20:40],label='COF',edgecolors='orange')
ax1.plot((0, 40), (0, 40), transform=ax.transAxes, ls='--', c='k')
#ax1.set_xlabel('Actual' ,fontsize =20)
#ax1.legend(loc='lower right')
#ax1.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$',fontsize=18)
ax1.set_title('DT' ,fontsize =24)
ax1.set_ylim(0,40)
yticks = np.linspace(0,40,9)
ax1.set_xlim(0,40)
xticks = np.linspace(0,40,9)
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
SRCC = "(c) SRCC = "+ str(format(SROCC,"0.2f"))    
ax1.text(0.75, 37.075, SRCC,fontsize =20)




#####################HGBR算法####################

ax = fig1.add_subplot(2,3,4)
plt.axis([0, 400, 0, 400])

model_col = HGBR_pre_col
ranks = HGBR_ranks
SROCC = 1-(6*sum((ranks-actual_ranks)**2))/(n*(n*n-1))

R2 = "(d) R$^{2}$ = "+ str(format(metrics.r2_score(actual_col,model_col),"0.2f"))  
sc1 = ax.scatter(actual_col[0:20],model_col[0:20],label='MOF',edgecolors='skyblue')

sc2 = ax.scatter(actual_col[20:40],model_col[20:40],label='COF',edgecolors='orange')




ax.set_ylim(0,400)
yticks = np.linspace(0,400,9)
ax.set_xlim(0,400)
xticks = np.linspace(0,400,9)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.text(7.5, 370.75, R2,fontsize= 20)
ax.plot((0, 400), (0, 400), transform=ax.transAxes, ls='--', c='k')
#ax.set_ylabel('Predict ' ,fontsize=22)
ax.set_title('HGBR' ,fontsize =24)
ax.set_xlabel(r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
ax.set_ylabel('Predicted '+r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)



ax1 = fig4.add_subplot(2,3,4)
plt.axis([0, 40, 0, 40])

sc1 = ax1.scatter(actual_ranks[0:20],ranks[0:20],label='MOF',edgecolors='skyblue')

sc2 = ax1.scatter(actual_ranks[20:40],ranks[20:40],label='COF',edgecolors='orange')
ax1.plot((0, 40), (0, 40), transform=ax.transAxes, ls='--', c='k')
#ax1.set_xlabel('Actual' ,fontsize =20)
#ax1.legend(loc='lower right')
ax1.set_xlabel(r'Ranks of $\mathrm{Q_{ed}}$',fontsize=22)
ax1.set_ylabel('Predicted ranks of '+r'$\mathrm{Q_{ed}}$',fontsize=22)
ax1.set_title('HGBR' ,fontsize =24)
ax1.set_ylim(0,40)
yticks = np.linspace(0,40,9)
ax1.set_xlim(0,40)
xticks = np.linspace(0,40,9)
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
SRCC = "(d) SRCC = "+ str(format(SROCC,"0.2f"))    
ax1.text(0.75, 37.075, SRCC,fontsize =20)



#####################GBR算法####################

ax = fig1.add_subplot(2,3,5)
plt.axis([0, 400, 0, 400])


model_col = GBR_pre_col
ranks = GBR_ranks
SROCC = 1-(6*sum((ranks-actual_ranks)**2))/(n*(n*n-1))

R2 = "(e) R$^{2}$ = "+ str(format(metrics.r2_score(actual_col,model_col),"0.2f"))  
sc1 = ax.scatter(actual_col[0:20],model_col[0:20],label='MOF',edgecolors='skyblue')

sc2 = ax.scatter(actual_col[20:40],model_col[20:40],label='COF',edgecolors='orange')




ax.set_ylim(0,400)
yticks = np.linspace(0,400,9)
ax.set_xlim(0,400)
xticks = np.linspace(0,400,9)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.text(7.5, 370.75, R2,fontsize= 20)
ax.plot((0, 400), (0, 400), transform=ax.transAxes, ls='--', c='k')
ax.set_xlabel(r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
ax.set_title('GBR' ,fontsize =24)
ax.set_yticklabels([])







ax1 = fig4.add_subplot(2,3,5)
plt.axis([0, 40, 0, 40])

sc1 = ax1.scatter(actual_ranks[0:20],ranks[0:20],label='MOF',edgecolors='skyblue')

sc2 = ax1.scatter(actual_ranks[20:40],ranks[20:40],label='COF',edgecolors='orange')
ax1.plot((0, 40), (0, 40), transform=ax.transAxes, ls='--', c='k')
ax1.set_xlabel(r'Ranks of $\mathrm{Q_{ed}}$',fontsize=22)
ax1.set_title('GBR' ,fontsize =24)
ax1.set_ylim(0,40)
yticks = np.linspace(0,40,9)
ax1.set_xlim(0,40)
xticks = np.linspace(0,40,9)
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
SRCC = "(e) SRCC = "+ str(format(SROCC,"0.2f"))    
ax1.text(0.75, 37.075, SRCC,fontsize =20)
ax1.set_yticklabels([])




#####################RF算法####################

ax = fig1.add_subplot(2,3,6)
plt.axis([0, 400, 0, 400])


model_col = RF_pre_col
ranks = RF_ranks
SROCC = 1-(6*sum((ranks-actual_ranks)**2))/(n*(n*n-1))

R2 = "(f) R$^{2}$ = "+ str(format(metrics.r2_score(actual_col,model_col),"0.2f"))  
sc1 = ax.scatter(actual_col[0:20],model_col[0:20],label='MOF',edgecolors='skyblue')

sc2 = ax.scatter(actual_col[20:40],model_col[20:40],label='COF',edgecolors='orange')


ax.set_ylim(0,400)
yticks = np.linspace(0,400,9)
ax.set_xlim(0,400)
xticks = np.linspace(0,400,9)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.plot((0, 400), (0, 400), transform=ax.transAxes, ls='--', c='k')
ax.set_xlabel(r'$\mathrm{Q_{ed}}$ (J/g)',fontsize=22)
#ax.set_ylabel('Predict ' ,fontsize=22)
ax.set_title('RF' ,fontsize =24)
ax.text(7.5, 370.75, R2,fontsize= 20)
ax.set_yticklabels([])



ax1 = fig4.add_subplot(2,3,6)
plt.axis([0, 40, 0, 40])

sc1 = ax1.scatter(actual_ranks[0:20],ranks[0:20],label='MOF',edgecolors='skyblue')

sc2 = ax1.scatter(actual_ranks[20:40],ranks[20:40],label='COF',edgecolors='orange')
ax1.plot((0, 40), (0, 40), transform=ax.transAxes, ls='--', c='k')
ax1.set_xlabel(r'Ranks of $\mathrm{Q_{ed}}$',fontsize=22)
ax1.set_title('RF' ,fontsize =24)
ax1.set_ylim(0,40)
yticks = np.linspace(0,40,9)
ax1.set_xlim(0,40)
xticks = np.linspace(0,40,9)
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
ax1.set_yticklabels([])
SRCC = "(f) SRCC = "+ str(format(SROCC,"0.2f"))    
ax1.text(0.75, 37.075, SRCC,fontsize =20)








plt.show()
fig1.savefig('transfer.png', format='png', dpi=600)
fig1.savefig('transfer.pdf', format='pdf', dpi=600)
fig4.savefig('transfer_SRCC.png', format='png', dpi=600)
fig4.savefig('transfer_SRCC.pdf', format='pdf', dpi=600)

#'''


#"""