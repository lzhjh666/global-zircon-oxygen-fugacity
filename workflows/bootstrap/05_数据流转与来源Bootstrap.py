#!/usr/bin/env python3
"""冻结数据的逐ID对账、筛选漏斗、来源Bootstrap和窗口敏感性。

不训练模型，不改变冻结表。直接点击运行即可。
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
INPUT=ROOT/'inputs'; OUTPUT=ROOT/'论文五项修订结果'
OLD=INPUT/'旧版10844候选.csv'
NEW=INPUT/'最终10879冻结.csv'
DOMAIN=INPUT/'原汇编筛选审计.json'
GEOROC=INPUT/'GEOROC扫描统计.json'

SCENARIOS={
 '主定义':{
  '早期氧化上升':((3000,3300),(3500,4001)),
  '晚太古代高值':((2500,3000),(1800,2500)),
  '中元古代相对低值':((800,1800),((400,800),(1800,2500))),
  '年轻阶段恢复':((0,400),(800,1800)),
 },
 '较窄窗口':{
  '早期氧化上升':((3000,3300),(3700,4001)),
  '晚太古代高值':((2600,2900),(1900,2400)),
  '中元古代相对低值':((1000,1600),((500,800),(1900,2400))),
  '年轻阶段恢复':((0,400),(1000,1600)),
 },
 '相邻宽窗口':{
  '早期氧化上升':((3000,3500),(3500,4001)),
  '晚太古代高值':((2500,3000),(2000,2500)),
  '中元古代相对低值':((800,1800),((400,800),(1800,2300))),
  '年轻阶段恢复':((0,500),(800,1800)),
 },
}
DIR={'早期氧化上升':1,'晚太古代高值':1,'中元古代相对低值':-1,'年轻阶段恢复':1}

def args():
 p=argparse.ArgumentParser(description='本地完成论文数据审计与四项来源Bootstrap（不训练）')
 p.add_argument('--iterations',type=int,default=10000)
 p.add_argument('--seed',type=int,default=20261118)
 p.add_argument('--output',type=Path,default=OUTPUT)
 return p.parse_args()

def read():
 old=pd.read_csv(OLD,low_memory=False)
 new=pd.read_csv(NEW,low_memory=False)
 if len(old)!=10844 or len(new)!=10879:raise ValueError('旧版或终版记录数不符；请使用随包文件')
 for d in (old,new):
  if d.GLOBAL_RECORD_ID.isna().any() or d.GLOBAL_RECORD_ID.duplicated().any():
   raise ValueError('GLOBAL_RECORD_ID存在空白或重复')
 if new[['AGE_MA_CLEAN','SOURCE_ID','DELTA_FMQ_ENSEMBLE']].isna().any().any():
  raise ValueError('终版关键字段有空值')
 return old,new

def file_hash(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def audit(old,new,out):
 old_ids=set(old.GLOBAL_RECORD_ID.astype(str));new_ids=set(new.GLOBAL_RECORD_ID.astype(str))
 retained=old_ids&new_ids;removed=old_ids-new_ids;added=new_ids-old_ids
 if (len(retained),len(removed),len(added))!=(10252,592,627):
  raise RuntimeError(f'逐ID对账与冻结漏斗不一致：共同{len(retained)}、删除{len(removed)}、新增{len(added)}')
 removed_df=old.loc[old.GLOBAL_RECORD_ID.astype(str).isin(removed)].copy()
 removed_df['旧版移出原因_按冻结标记']=np.select(
  [~removed_df.DOMAIN_PASS.fillna(False).astype(bool),removed_df.COMPOSITION_DUPLICATE.fillna(False).astype(bool)],
  ['元素组成适用域外','潜在重复'],default='需核查')
 removed_df.to_csv(out/'01_旧版移出592条逐ID.csv',index=False,encoding='utf-8-sig')
 new.loc[new.GLOBAL_RECORD_ID.astype(str).isin(added)].to_csv(out/'02_新增627条逐ID.csv',index=False,encoding='utf-8-sig')
 if removed_df['旧版移出原因_按冻结标记'].value_counts().to_dict()!={'元素组成适用域外':588,'潜在重复':4}:
  raise RuntimeError('移出原因数量异常，请检查冻结口径')
 with DOMAIN.open(encoding='utf-8') as f:w=json.load(f)
 with GEOROC.open(encoding='utf-8') as f:g=json.load(f)
 funnel=pd.DataFrame([
  ['Wang原始0–4000 Ma母集',w['global_input_rows'],'原始补充表'],
  ['Wang旧版12元素及QC候选',len(old),'年龄、14元素至少12项、P/La/Th-U筛选'],
  ['剔除域外',-588,'元素组成适用域判定'],
  ['剔除潜在重复',-4,'重复审计'],
  ['Wang保留',len(retained),'旧新版共同ID'],
  ['GEOROC原始',g['raw_rows'],'独立全库扫描'],
  ['GEOROC锆石记录',g['zircon_rows'],'矿物筛选'],
  ['GEOROC年龄0–4000 Ma',g['age_0_4000'],'年龄筛选'],
  ['GEOROC碎屑/沉积关键词',g['detrital_keyword'],'身份候选'],
  ['GEOROC至少12元素',g['at_least_12_features'],'模型特征门槛'],
  ['GEOROC通过P/La/Th-U QC',g['zircon_qc_pass'],'质量筛选'],
  ['GEOROC位于训练域内',g['in_domain'],'距离阈值'],
  ['GEOROC最终新增',len(added),'岩性、文献与严格成分重复审计'],
  ['最终冻结总数',len(new),'10252+627'],
 ],columns=['阶段','记录数或变化数','规则'])
 funnel.to_csv(out/'03_双来源筛选漏斗.csv',index=False,encoding='utf-8-sig')
 # 原Wang的逐项QC可用；GEOROC仅持有合并后的通过标记，单项标记为空。
 rows=[]
 for origin,group in new.groupby('DATA_ORIGIN',dropna=False):
  row={'DATA_ORIGIN':origin,'数量':len(group)}
  for col in ['QC_P_GT750','QC_LA_GT1','QC_TH_U_LT0_1','ZIRCON_QC_PASS','P','La','Th/U','TH_U']:
   row[col+'_非空数']=int(group[col].notna().sum()) if col in group else 0
  if 'ZIRCON_QC_PASS' in group:row['总体QC通过数']=int(group.ZIRCON_QC_PASS.fillna(False).astype(bool).sum())
  rows.append(row)
 pd.DataFrame(rows).to_csv(out/'04_QC标记完整性审计.csv',index=False,encoding='utf-8-sig')
 return funnel,removed_df

def intervals():
 ranges=[]
 for defs in SCENARIOS.values():
  for target,reference in defs.values():
   ranges.append(target)
   if isinstance(reference[0],tuple):ranges.extend(reference)
   else:ranges.append(reference)
 return list(dict.fromkeys(ranges))

def precompute(new):
 sources=np.array(sorted(new.SOURCE_ID.astype(str).unique()))
 age=new.AGE_MA_CLEAN.to_numpy(float)
 code=pd.Categorical(new.SOURCE_ID.astype(str),categories=sources).codes
 value=new.DELTA_FMQ_ENSEMBLE.to_numpy(float)
 matrices={};support={}
 for interval in intervals():
  lo,hi=interval;mask=(age>=lo)&(age<hi)
  v=np.full(len(sources),np.nan)
  for i in np.unique(code[mask]):v[i]=np.median(value[mask&(code==i)])
  matrices[interval]=v;support[interval]=(int(mask.sum()),int(np.isfinite(v).sum()))
 return sources,matrices,support

def median(v,idx):
 x=v[idx];x=x[np.isfinite(x)]
 return float(np.median(x)) if len(x) else np.nan

def contrast(definition,matrices,idx):
 target,reference=definition
 a=median(matrices[target],idx)
 if isinstance(reference[0],tuple):
  b=np.mean([median(matrices[r],idx) for r in reference])
 else:b=median(matrices[reference],idx)
 return a-b

def bootstrap(new,out,iterations,seed):
 sources,matrices,support=precompute(new)
 rng=np.random.default_rng(seed);baseline_idx=np.arange(len(sources))
 rows=[];iterations_rows=[]
 for scenario,defs in SCENARIOS.items():
  for name,definition in defs.items():
   observed=contrast(definition,matrices,baseline_idx)
   reps=np.empty(iterations)
   for i in range(iterations):
    # 同一来源在多个区间出现时，整轮共用抽中的来源ID，保留跨时代相关结构。
    if name==list(defs)[0]:
     if i==0:draws=np.empty((iterations,len(sources)),dtype=np.int16)
     draws[i]=rng.integers(0,len(sources),size=len(sources))
    reps[i]=contrast(definition,matrices,draws[i])
   valid=np.isfinite(reps);q=np.quantile(reps[valid],[.025,.5,.975])
   direction=float((DIR[name]*reps[valid]>0).mean())
   used=[definition[0]]+(list(definition[1]) if isinstance(definition[1][0],tuple) else [definition[1]])
   row={'方案':scenario,'对比':name,'目标年龄_Ma':str(definition[0]),'参照年龄_Ma':str(definition[1]),
        '目标颗粒数':support[definition[0]][0],'目标来源数':support[definition[0]][1],
        '参照来源数':';'.join(str(support[r][1]) for r in used[1:]),
        '观察差值_ΔFMQ':observed,'CI95下限':q[0],'Bootstrap中位':q[1],'CI95上限':q[2],
        '方向保留率':direction,'CI95排除零':bool(q[0]>0 or q[2]<0),'有效模拟次数':int(valid.sum())}
   rows.append(row)
   if scenario=='主定义':
    iterations_rows.append(pd.DataFrame({'对比':name,'迭代':np.arange(1,iterations+1),'来源Bootstrap差值':reps}))
 summary=pd.DataFrame(rows)
 summary.to_csv(out/'05_四项宽阶段及窗口敏感性_来源Bootstrap.csv',index=False,encoding='utf-8-sig')
 pd.concat(iterations_rows,ignore_index=True).to_csv(out/'06_主定义逐次Bootstrap.csv',index=False,encoding='utf-8-sig')
 return summary

def figures(funnel,summary,out):
 fig_dir=out/'figures';fig_dir.mkdir(exist_ok=True)
 plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','Noto Sans CJK SC','DejaVu Sans'],'axes.unicode_minus':False})
 main=summary[summary.方案.eq('主定义')].copy();x=np.arange(len(main))
 fig,ax=plt.subplots(figsize=(9,5))
 ax.vlines(x,main['CI95下限'],main['CI95上限'],color='#28587d',lw=2)
 ax.scatter(x,main['观察差值_ΔFMQ'],color='#28587d',zorder=3)
 ax.axhline(0,color='#555',lw=1);ax.set_xticks(x,main.对比,rotation=15);ax.set(ylabel='来源等权差值 ΔFMQ',title='四项宽阶段差值与来源Bootstrap 95%区间')
 fig.tight_layout();fig.savefig(fig_dir/'01_四项来源Bootstrap区间.png',dpi=230);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5))
 for i,name in enumerate(DIR):
  g=summary[summary.对比.eq(name)].set_index('方案').loc[list(SCENARIOS)]
  xx=np.arange(len(g))+i*4
  ax.vlines(xx,g['CI95下限'],g['CI95上限'],lw=1.8)
  ax.scatter(xx,g['观察差值_ΔFMQ'],s=24,label=name,zorder=3)
 ax.axhline(0,color='#555',lw=1);ax.set_xticks(np.concatenate([np.arange(3)+i*4 for i in range(4)]),[x for _ in range(4) for x in ['主','窄','邻']]);ax.set(ylabel='来源等权差值 ΔFMQ',title='窗口定义敏感性');ax.legend(ncol=2,fontsize=8)
 fig.tight_layout();fig.savefig(fig_dir/'02_三种窗口定义敏感性.png',dpi=230);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,5))
 b=funnel[funnel.阶段.isin(['Wang旧版12元素及QC候选','Wang保留','GEOROC最终新增','最终冻结总数'])]
 ax.barh(b.阶段,b['记录数或变化数'],color=['#4c78a8','#2b8579','#f2a65a','#284a6d'])
 ax.set(xlabel='记录数',title='旧版数据对账与最终汇编');fig.tight_layout();fig.savefig(fig_dir/'03_数据流转.png',dpi=230);plt.close(fig)

def report(funnel,removed,summary,old,new,out):
 main=summary[summary.方案.eq('主定义')]
 def as_text(df):
  return '```text\n'+df.to_string(index=False)+'\n```'
 lines=['# 最终数据与四项趋势统计核验','',f'旧版 {len(old):,} → 剔除适用域外588及潜在重复4 → 原汇编保留10,252 → 新增GEOROC 627 → 最终 {len(new):,}。','',
        'P>750 ppm、La>1 ppm和Th/U<0.1在筛选时为剔除条件；缺测值不单独触发剔除。旧汇编逐项标记保留在冻结表中；新增GEOROC的逐项QC列未写入统一表，但总体ZIRCON_QC_PASS为True。单项阈值的原始实现见全库扫描脚本。','',
        '来源Bootstrap以55个SOURCE_ID整体有放回重采样，同一来源在各年龄段使用相同抽样次数；区间为2.5%—97.5%百分位。它测量来源构成的不确定度，与Kohistan模型误差情景不同。','',
        '## 主定义四项结果','',as_text(main),'',
        '## 全部窗口情景','',as_text(summary),'',
        '注意：宽阶段对比、旧版五项窄窗对比与Kohistan误差方向率不属于同一个估计量；论文应分别报告。']
 (out/'07_结果解释与稿件替换清单.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
 a=args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 old,new=read()
 funnel,removed=audit(old,new,out)
 summary=bootstrap(new,out,a.iterations,a.seed)
 figures(funnel,summary,out)
 report(funnel,removed,summary,old,new,out)
 (out/'08_输入哈希与运行参数.json').write_text(json.dumps({'old_sha256':file_hash(OLD),'new_sha256':file_hash(NEW),'iterations':a.iterations,'seed':a.seed,'training_performed':False},ensure_ascii=False,indent=2),encoding='utf-8')
 print('完成：',out)
 print(summary[summary.方案.eq('主定义')][['对比','观察差值_ΔFMQ','CI95下限','CI95上限','方向保留率']].to_string(index=False))

if __name__=='__main__':main()
