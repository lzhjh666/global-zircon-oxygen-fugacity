"""Regenerate paper-data plots from the published CSV files, without fitting models.

These are reproducible redraws of the current main-figure data, not the original
matplotlib source used to create the bitmaps embedded in the draft manuscript.
Run: python workflows/figures/plot_release_figures.py
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]

def read(path):
    return pd.read_csv(REPO / path).sort_index()

def write(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / name, dpi=240, bbox_inches='tight')
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=REPO / 'figures/redrawn_from_tables')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK SC','Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})

    train=read('results/model_development/模型指标汇总.csv')
    koh=read('results/kohistan_validation/五岩体真实值与预测值.csv')
    mlp=train[(train['元素方案']=='筛选后元素') & (train['模型']=='MLP_64x32')]
    cv=float(mlp['平均折MSE'].iloc[0]);hold=float(read('results/model_development/锁定地点独立测试指标.csv').query("模型 == 'MLP_64x32'")['MSE'].iloc[0])
    kp=koh.pivot(index='岩体',columns='模型',values='预测ΔFMQ');true=koh.drop_duplicates('岩体').set_index('岩体')['真实ΔFMQ']
    pred=(kp['MLP_64x32']+kp['ExtraTrees_正则化'])/2
    mse=float(np.mean((pred-true)**2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.6));labels=['Grouped CV\nMLP','Five-region\nholdout MLP','Kohistan\nensemble']
    axes[0].bar(labels,[cv,hold,mse],color=['#476e91','#6594ae','#d6815e'])
    for i,v in enumerate([cv,hold,mse]):axes[0].text(i,v+.015,f'{v:.3f}',ha='center')
    axes[0].set(ylabel=r'MSE ($\Delta$FMQ$^2$)',title='Validation at three transfer scales',ylim=(0,max(cv,hold,mse)*1.18))
    biases=(pred-true).sort_values();axes[1].barh(biases.index,biases.values,color=np.where(biases.values<0,'#d6815e','#476e91'));axes[1].axvline(0,color='0.2',lw=.8)
    axes[1].set(xlabel=r'Predicted minus reference ($\Delta$FMQ)',title='Kohistan pluton biases')
    write(fig,a.output,'01_validation.png')

    t=read('results/sensitivity_results/02_时间窗_100Ma.csv').sort_values('年龄中心_Ma')
    l=read('results/sensitivity_results/05_LOSO汇总.csv').sort_values('年龄中心_Ma')
    x=t['年龄中心_Ma'].to_numpy(dtype=float)/1000
    fig,ax=plt.subplots(figsize=(11,5.2))
    ax.plot(x,t['来源等权中位数'].to_numpy(float),color='#173d61',lw=2.5,label='Source equal median')
    ax.plot(x,t['双模型颗粒中位数'].to_numpy(float),color='#dd9474',label='Grain median')
    ax.fill_between(l['年龄中心_Ma'].to_numpy(float)/1000,l['LOSO最小值'].to_numpy(float),l['LOSO最大值'].to_numpy(float),color='#173d61',alpha=.15,label='Leave one source out range')
    ax.set(xlabel='Age (Ga; older to right)',ylabel=r'Predicted $\Delta$FMQ',title='Global zircon redox trajectory, 0–4.0 Ga',xlim=(0,4));ax.legend(frameon=False,ncol=2)
    write(fig,a.output,'02_global_trajectory.png')

    fig,ax=plt.subplots(3,1,figsize=(11,8),sharex=True)
    ax[0].bar(x,t['颗粒数'],width=.075,color='#517b9c');ax[0].set_ylabel('Grains')
    ax[1].bar(x,t['独立来源数'],width=.075,color='#559986');ax[1].set_ylabel('Sources')
    ax[2].plot(x,t['最大来源占比'],color='#be8058');ax[2].set(ylabel='Largest source share',xlabel='Age (Ga; older to right)',xlim=(0,4),ylim=(0,1))
    fig.suptitle('Age-bin sample support');write(fig,a.output,'03_data_support.png')

    b=read('results/bootstrap/05_四项宽阶段及窗口敏感性_来源Bootstrap.csv');b=b[b['方案']=='主定义'].copy()
    fig,ax=plt.subplots(figsize=(10,4.7));y=np.arange(len(b));v=b['观察差值_ΔFMQ'].to_numpy(float)
    lo=b['CI95下限'].to_numpy(float);hi=b['CI95上限'].to_numpy(float)
    ax.errorbar(v,y,xerr=np.array([v-lo,hi-v]),fmt='o',color='#173d61',capsize=4)
    ax.axvline(0,color='0.4',lw=.8);ax.set_yticks(y,['Early rise','Late Archean high','Mid-Proterozoic low','Young recovery'])
    ax.invert_yaxis();ax.set(xlabel=r'Contrast $\Delta$FMQ (95% source bootstrap CI)',title='Four pre-defined broad-stage contrasts')
    write(fig,a.output,'04_bootstrap_contrasts.png')

    dom=read('results/age_domain/01_每100Ma适用域与数据支撑.csv').sort_values('年龄中心_Ma')
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True);xd=dom['年龄中心_Ma'].to_numpy(float)/1000
    axes[0].plot(xd,dom['归一化域距离_中位数'],color='#276e88');axes[0].plot(xd,dom['归一化域距离_P90'],color='#d6815e',alpha=.8)
    axes[0].axhline(1,color='0.4',ls='--',lw=.8);axes[0].set(ylabel='Normalized domain distance',title='Model domain by age')
    axes[1].plot(xd,dom['原适用域通过率'],color='#276e88');axes[1].set(xlabel='Age (Ga; older to right)',ylabel='Fraction inside domain',ylim=(0,1.05),xlim=(0,4))
    write(fig,a.output,'05_age_domain.png')

    stress=read('results/kohistan_noise/04_四情景方向保留概率.csv')
    subset=stress[(stress['误差情景']=='来源分层误差') & (stress['聚合方法']=='来源等权')]
    names={'古老地球氧化上升':'Early rise','晚太古代高值':'Late Archean high',
           '中元古代低值_相对两侧':'Mid-Proterozoic low','显生宙恢复':'Young recovery'}
    fig,ax=plt.subplots(figsize=(9,4.5));labels=subset['结论'].map(names)
    ax.barh(np.arange(len(subset)),subset['方向保留概率'].to_numpy(float),color='#517b9c');ax.set_yticks(np.arange(len(subset)),labels)
    ax.set(xlim=(0,1.05),xlabel='Direction retention fraction',title='Kohistan error stress test');write(fig,a.output,'06_kohistan_stress.png')
    print('Wrote six redraws to',a.output)

if __name__=='__main__': main()
