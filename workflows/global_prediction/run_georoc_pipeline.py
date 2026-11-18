#!/usr/bin/env python3
"""全球碎屑锆石：论文数据/GEOROC 清洗、冻结双模型预测与绘图。

本脚本只加载随包提供的冻结模型进行推理，不包含 fit、训练或调参步骤。
"""
from __future__ import annotations

import argparse, csv, json, re, sys, warnings, zipfile, shutil, smtplib, hashlib
from pathlib import Path
from email.message import EmailMessage
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

FEATURES = ["Ce", "Nd", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "U", "Hf", "Ti"]
META = ["CITATION","SAMPLE NAME","TECTONIC SETTING","LOCATION","LOCATION COMMENT","LATITUDE (MIN.)","LATITUDE (MAX.)","LONGITUDE (MIN.)","LONGITUDE (MAX.)","LAND/SEA (SAMPLING)","ROCK NAME","ROCK TEXTURE","ALTERATION","MINERAL","SPOT","CRYSTAL","RIM/CORE (MINERAL GRAINS)","PRIMARY/SECONDARY"]
AGES = ["AGE(MA)","AGE(KA)","AGE_PB207_PB206(MA)","AGE_PB207_PB206(KA)","AGE_PB207_U235(MA)","AGE_PB207_U235(KA)","AGE_PB206_U238(MA)","AGE_PB206_U238(KA)","AGE_PB208_TH232(MA)","AGE_PB208_TH232(KA)"]
DETRITAL = re.compile(r"detrit|sediment|sandstone|quartzite|conglomerate|greywacke|graywacke|metasediment|shale|mudstone|siltstone|beach|river|stream|alluvial|placer", re.I)

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--data-mode", choices=["paper","georoc"], default="paper", help="默认使用论文公开全球汇编；需要时可切换回 GEOROC")
    p.add_argument("--paper-data", default="data/B38079_SuppTab2_全球碎屑锆石微量元素.xlsx")
    p.add_argument("--input", default="GEOROC全球锆石原始数据_V10_615124条.zip")
    p.add_argument("--models", default="models")
    p.add_argument("--output", default="GEOROC_双模型预测结果")
    p.add_argument("--chunksize", type=int, default=30000)
    p.add_argument("--min-features", type=int, default=12, help="14个模型元素至少实测数；12表示最多缺2个")
    p.add_argument("--strict-features", type=int, default=14)
    p.add_argument("--bin-ma", type=int, default=100)
    return p.parse_args()

def setup_plot():
    plt.rcParams.update({"font.sans-serif":["Noto Sans CJK SC","Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False,"figure.dpi":130,"savefig.dpi":220})

def locate_csv(p:Path, temp:Path):
    if p.suffix.lower() != ".zip": return p
    with zipfile.ZipFile(p) as z:
        ns=[n for n in z.namelist() if n.lower().endswith('.csv')]
        if len(ns)!=1: raise RuntimeError(f"ZIP 中 CSV 数量不是 1：{ns}")
        q=temp/Path(ns[0]).name
        if not q.exists():
            with z.open(ns[0]) as src, q.open('wb') as dst:
                while b:=src.read(8*1024*1024): dst.write(b)
        return q

def numeric(s):
    s=s.astype('string').str.strip().str.replace(',','',regex=False)
    below=s.str.match(r"^<\s*[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?$",na=False)
    v=pd.to_numeric(s.str.replace(r"^[<>~=≤≥]\s*",'',regex=True),errors='coerce')
    v.loc[below]=v.loc[below]/2
    return v.mask(~np.isfinite(v)|(v<=0)).astype(float),below.astype('int8')

def age_numeric(s):
    s=s.astype('string').str.strip().str.replace(',','',regex=False)
    v=pd.to_numeric(s.str.replace(r"^[<>~=≤≥]\s*",'',regex=True),errors='coerce')
    return v.mask(~np.isfinite(v)|(v<0)).astype(float)

def element(chunk,e):
    out=pd.Series(np.nan,index=chunk.index,dtype=float); dl=pd.Series(0,index=chunk.index,dtype='int8')
    for col,factor in [(f'{e.upper()}(PPM)',1),(f'{e.upper()}(WT%)',10000),(f'{e.upper()}(PPB)',.001)]:
        if col not in chunk: continue
        v,b=numeric(chunk[col]); take=out.isna()&v.notna(); out.loc[take]=v.loc[take]*factor; dl.loc[take]=b.loc[take]
    return out,dl

def ages(chunk):
    vals={}
    for c in AGES:
        if c in chunk:
            vals[c]=age_numeric(chunk[c])/(1000 if c.endswith('(KA)') else 1)
    age=pd.Series(np.nan,index=chunk.index,dtype=float); src=pd.Series('',index=chunk.index,dtype='string')
    def fill(c,mask=None,label=None):
        if c not in vals:return
        t=age.isna()&vals[c].notna()
        if mask is not None:t&=mask
        age.loc[t]=vals[c].loc[t];src.loc[t]=label or c
    fill('AGE(MA)');fill('AGE(KA)')
    def combined(ma,ka):
        a=vals.get(ma);b=vals.get(ka)
        return b if a is None else a if b is None else a.combine_first(b)
    u=combined('AGE_PB206_U238(MA)','AGE_PB206_U238(KA)')
    pb=combined('AGE_PB207_PB206(MA)','AGE_PB207_PB206(KA)')
    if u is not None:
        t=age.isna()&u.notna()&(u<=1000);age.loc[t]=u.loc[t];src.loc[t]='AGE_PB206_U238(best<1Ga)'
    if pb is not None:
        t=age.isna()&pb.notna();age.loc[t]=pb.loc[t];src.loc[t]='AGE_PB207_PB206(best>1Ga)'
    if u is not None:
        t=age.isna()&u.notna();age.loc[t]=u.loc[t];src.loc[t]='AGE_PB206_U238(fallback)'
    for c in ['AGE_PB207_U235(MA)','AGE_PB207_U235(KA)','AGE_PB208_TH232(MA)','AGE_PB208_TH232(KA)']:fill(c)
    return age,src

def append(df,p,first):
    df.to_csv(p,mode='w' if first else 'a',header=first,index=False,encoding='utf-8-sig',quoting=csv.QUOTE_MINIMAL)

def clean(inp,out,chunksize,minf,strictf):
    temp=out/'_临时解压';temp.mkdir(exist_ok=True); src=locate_csv(inp,temp)
    # GEOROC V10 CSV 是混合单字节文本，既含 Windows-1252 字符，也含
    # cp1252 中未定义的字节（如 0x81）。latin-1 对 0–255 每个字节都有
    # 一对一映射，可保证整份文件读到末尾而不因文本备注字符中断。
    encoding='latin1'
    header=pd.read_csv(src,nrows=0,encoding=encoding).columns.tolist()
    use=[c for c in META+AGES if c in header]
    for e in FEATURES: use += [c for c in (f'{e.upper()}(PPM)',f'{e.upper()}(WT%)',f'{e.upper()}(PPB)') if c in header]
    for e in ['P','La','Th']: use += [c for c in (f'{e.upper()}(PPM)',f'{e.upper()}(WT%)',f'{e.upper()}(PPB)') if c in header]
    use=list(dict.fromkeys(use)); paths={k:out/v for k,v in {'broad':'01_全球锆石宽松候选.csv','detrital':'02_碎屑锆石候选.csv','strict':'03_碎屑锆石高完整度候选.csv'}.items()}
    for p in paths.values(): p.unlink(missing_ok=True)
    first={k:True for k in paths}; stat={'原始记录':0,'锆石记录':0,'0至4000Ma':0,'宽松候选':0,'碎屑候选':0,'高完整度碎屑候选':0,'低于检出限值':0}; present={e:0 for e in FEATURES}
    for no,c in enumerate(pd.read_csv(src,usecols=use,chunksize=chunksize,dtype=str,encoding=encoding,low_memory=False,on_bad_lines='warn'),1):
        stat['原始记录']+=len(c); mineral=c.get('MINERAL',pd.Series('ZIRCON',index=c.index)).fillna('').str.upper(); c=c.loc[mineral.str.contains('ZIRCON',na=False)].copy();stat['锆石记录']+=len(c)
        if c.empty:continue
        d=pd.DataFrame(index=c.index)
        for x in META:
            if x in c:d[x]=c[x]
        d['AGE_MA_CLEAN'],d['AGE_SOURCE']=ages(c); keep=d.AGE_MA_CLEAN.between(0,4000);c=c.loc[keep];d=d.loc[keep];stat['0至4000Ma']+=len(d)
        for e in FEATURES:
            d[e],dl=element(c,e);d[e+'_BELOW_DL']=dl;present[e]+=int(d[e].notna().sum());stat['低于检出限值']+=int(dl.sum())
        d['N_FEATURES_MEASURED']=d[FEATURES].notna().sum(axis=1);d['MISSING_FEATURES']=d[FEATURES].isna().apply(lambda r:';'.join(r.index[r]),axis=1)
        d['P_PPM']=element(c,'P')[0];d['LA_PPM_QC']=element(c,'La')[0];d['TH_PPM_QC']=element(c,'Th')[0]
        d['TH_U_RATIO_QC']=d['TH_PPM_QC']/d['U']
        d['QC_P_GT750']=d['P_PPM'].gt(750).fillna(False)
        d['QC_LA_GT1']=d['LA_PPM_QC'].gt(1).fillna(False)
        d['QC_TH_U_LT0_1']=d['TH_U_RATIO_QC'].lt(.1).fillna(False)
        d['PAPER_QC_PASS']=~(d['QC_P_GT750']|d['QC_LA_GT1']|d['QC_TH_U_LT0_1'])
        txt=c[[x for x in ['ROCK NAME','ROCK TEXTURE','LOCATION','LOCATION COMMENT','SAMPLE NAME','CITATION'] if x in c]].fillna('').astype(str).agg(' | '.join,axis=1)
        d['DETRITAL_KEYWORD_MATCH']=txt.str.contains(DETRITAL,na=False)
        for side in ['LATITUDE','LONGITUDE']:
            coord=[]
            for suffix in (' (MIN.)',' (MAX.)'):
                col=side+suffix
                coord.append(pd.to_numeric(c[col],errors='coerce') if col in c else pd.Series(np.nan,index=c.index))
            d[side+'_CENTER']=pd.concat(coord,axis=1).mean(axis=1)
        sets={'broad':d[d.N_FEATURES_MEASURED>=minf]};sets['detrital']=sets['broad'][sets['broad'].DETRITAL_KEYWORD_MATCH & sets['broad'].PAPER_QC_PASS];sets['strict']=sets['detrital'][sets['detrital'].N_FEATURES_MEASURED>=strictf]
        for k,x in sets.items():
            stat[{'broad':'宽松候选','detrital':'碎屑候选','strict':'高完整度碎屑候选'}[k]]+=len(x)
            if len(x):append(x,paths[k],first[k]);first[k]=False
        print(f'清洗块 {no}：原始累计 {stat["原始记录"]:,}，碎屑候选 {stat["碎屑候选"]:,}',flush=True)
    comp=pd.DataFrame({'元素':FEATURES,'有效数':[present[e] for e in FEATURES]});comp['完整率']=comp['有效数']/max(stat['0至4000Ma'],1);comp.to_csv(out/'04_模型元素完整率.csv',index=False,encoding='utf-8-sig')
    (out/'05_清洗统计.json').write_text(json.dumps(stat,ensure_ascii=False,indent=2),encoding='utf-8')
    return paths,comp,stat

def clean_paper(inp:Path,out:Path,minf:int,strictf:int):
    """读取公开全球碎屑锆石汇编；缺失1–2个模型元素由冻结模型内置插补器处理。"""
    if not inp.exists(): raise FileNotFoundError(f'找不到论文数据：{inp}')
    raw=pd.read_excel(inp,header=2)
    rename={'Selected_age':'AGE_MA_CLEAN','Region':'LOCATION','Sub_region':'LOCATION COMMENT','Sample':'SAMPLE NAME','Reference':'CITATION'}
    raw=raw.rename(columns=rename)
    required=['AGE_MA_CLEAN']+FEATURES
    missing=[c for c in required if c not in raw.columns]
    if missing: raise RuntimeError(f'论文数据缺少必要字段：{missing}')
    d=pd.DataFrame(index=raw.index)
    for c in ['ID','AGE_MA_CLEAN','Age_2s_error','SAMPLE NAME','LOCATION','LOCATION COMMENT','CITATION','Analysis']+FEATURES+['P','La','Th','Th/U']:
        if c in raw:d[c]=raw[c]
    for c in ['AGE_MA_CLEAN']+FEATURES+['P','La','Th','Th/U']:
        if c in d:d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d[d.AGE_MA_CLEAN.between(0,4000)].copy()
    d[FEATURES]=d[FEATURES].where(d[FEATURES]>0)
    d['AGE_SOURCE']='论文汇编 Selected_age'
    d['DATA_SOURCE']='Wang et al. (2025), DOI:10.1130/B38079.1, Supplement Table S2'
    d['N_FEATURES_MEASURED']=d[FEATURES].notna().sum(axis=1)
    d['MISSING_FEATURES']=d[FEATURES].isna().apply(lambda r:';'.join(r.index[r]),axis=1)
    d['P_PPM']=d.get('P',np.nan);d['LA_PPM_QC']=d.get('La',np.nan);d['TH_PPM_QC']=d.get('Th',np.nan)
    ratio=d.get('Th/U',pd.Series(np.nan,index=d.index))
    ratio=ratio.combine_first(d['TH_PPM_QC']/d['U'])
    d['TH_U_RATIO_QC']=ratio
    d['QC_P_GT750']=d['P_PPM'].gt(750).fillna(False)
    d['QC_LA_GT1']=d['LA_PPM_QC'].gt(1).fillna(False)
    d['QC_TH_U_LT0_1']=d['TH_U_RATIO_QC'].lt(.1).fillna(False)
    d['PAPER_QC_PASS']=~(d.QC_P_GT750|d.QC_LA_GT1|d.QC_TH_U_LT0_1)
    d['DETRITAL_KEYWORD_MATCH']=True
    d['LATITUDE_CENTER']=np.nan;d['LONGITUDE_CENTER']=np.nan
    broad=d[d.N_FEATURES_MEASURED>=minf].copy()
    detrital=broad[broad.PAPER_QC_PASS].copy()
    strict=detrital[detrital.N_FEATURES_MEASURED>=strictf].copy()
    paths={'broad':out/'01_全球锆石宽松候选.csv','detrital':out/'02_碎屑锆石候选.csv','strict':out/'03_碎屑锆石高完整度候选.csv'}
    for k,x in [('broad',broad),('detrital',detrital),('strict',strict)]:x.to_csv(paths[k],index=False,encoding='utf-8-sig')
    comp=pd.DataFrame({'元素':FEATURES,'有效数':[int(d[e].notna().sum()) for e in FEATURES]})
    comp['完整率']=comp['有效数']/max(len(d),1);comp.to_csv(out/'04_模型元素完整率.csv',index=False,encoding='utf-8-sig')
    stat={'原始记录':int(len(raw)),'锆石记录':int(len(raw)),'0至4000Ma':int(len(d)),'宽松候选':int(len(broad)),'碎屑候选':int(len(detrital)),'高完整度碎屑候选':int(len(strict)),'低于检出限值':0}
    (out/'05_清洗统计.json').write_text(json.dumps(stat,ensure_ascii=False,indent=2),encoding='utf-8')
    return paths,comp,stat

def load_models(folder):
    if sklearn.__version__ != '1.3.2':
        raise RuntimeError(f'冻结模型需要 scikit-learn==1.3.2，当前为 {sklearn.__version__}。请先运行：python -m pip install -r requirements.txt')
    fs=list(folder.glob('*.joblib')); mlp=next((f for f in fs if 'MLP' in f.name.upper()),None); extra=next((f for f in fs if 'EXTRA' in f.name.upper()),None)
    if not mlp or not extra:raise FileNotFoundError('models 文件夹中必须同时存在 MLP 和 ExtraTrees 的 joblib 文件')
    mlp_bundle=joblib.load(mlp)['bundle'];extra_bundle=joblib.load(extra)['bundle']
    for name,bundle in [('MLP',mlp_bundle),('ExtraTrees',extra_bundle)]:
        model_features=[re.split(r'[（(]',str(x),maxsplit=1)[0] for x in bundle.get('features',[])]
        if model_features != FEATURES:
            raise RuntimeError(f'{name} 模型的特征顺序与代码不一致：{bundle.get("features")}')
    return mlp_bundle,extra_bundle

def predict_file(path,out,models,chunksize):
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError('没有筛选出可预测的碎屑锆石记录，请检查筛选条件或原始数据。')
    mlp,extra=models; target=out/'06_碎屑锆石双模型预测.csv';target.unlink(missing_ok=True);first=True
    for no,d in enumerate(pd.read_csv(path,chunksize=chunksize,low_memory=False),1):
        raw=d[FEATURES].apply(pd.to_numeric,errors='coerce').where(lambda x:x>0); X=np.log10(raw)
        Xm=mlp['imputer'].transform(X); pred_m=np.mean([m.predict(mlp['scaler'].transform(Xm)) for m in mlp['models']],axis=0)
        Xe=extra['imputer'].transform(X); pred_e=extra['model'].predict(Xe)
        d['DELTA_FMQ_MLP']=pred_m;d['DELTA_FMQ_EXTRATREES']=pred_e;d['DELTA_FMQ_ENSEMBLE']=(pred_m+pred_e)/2;d['MODEL_ABS_DIFF']=np.abs(pred_m-pred_e)
        append(d,target,first);first=False;print(f'预测块 {no}：{len(d):,} 条',flush=True)
    return target

def binned(pred,binma):
    d=pd.read_csv(pred,usecols=['AGE_MA_CLEAN','DELTA_FMQ_MLP','DELTA_FMQ_EXTRATREES','DELTA_FMQ_ENSEMBLE','MODEL_ABS_DIFF'],low_memory=False)
    edges=np.arange(0,4000+binma,binma);ages_for_bin=d.AGE_MA_CLEAN.clip(upper=np.nextafter(4000.0,-np.inf));d['AGE_BIN']=pd.cut(ages_for_bin,edges,include_lowest=True,right=False);g=d.groupby('AGE_BIN',observed=False)
    rows=[]
    for interval,x in g:
        row={'年龄中心_Ma':float(interval.left+binma/2),'样本数':len(x)}
        for c,n in [('DELTA_FMQ_MLP','MLP'),('DELTA_FMQ_EXTRATREES','ExtraTrees'),('DELTA_FMQ_ENSEMBLE','双模型均值')]:
            v=x[c].dropna();row[n+'_中位数']=v.median();row[n+'_均值']=v.mean();row[n+'_Q25']=v.quantile(.25);row[n+'_Q75']=v.quantile(.75)
        row['模型差异中位数']=x.MODEL_ABS_DIFF.median();rows.append(row)
    t=pd.DataFrame(rows);t['双模型均值_三窗平滑']=t['双模型均值_中位数'].rolling(3,center=True,min_periods=1).median();t['变化率_每100Ma']=np.gradient(t['双模型均值_三窗平滑'],t['年龄中心_Ma'])*100
    return t,d

def savefig(fig,p):fig.tight_layout();fig.savefig(p,bbox_inches='tight');plt.close(fig)

def plots(pred,comp,stat,out,binma):
    setup_plot(); figs=out/'图件';figs.mkdir(exist_ok=True);t,d=binned(pred,binma);t.to_csv(out/'07_每100Ma氧逸度演化与变化率.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots();ax.hist(d.AGE_MA_CLEAN,bins=80,color='#3978a8');ax.set(xlabel='年龄（Ma）',ylabel='锆石数量',title='0–4000 Ma 年龄分布');savefig(fig,figs/'01_年龄分布直方图.png')
    fig,ax=plt.subplots();ax.plot(t['年龄中心_Ma'],t['样本数'],color='#d55e00');ax.set(xlabel='年龄（Ma）',ylabel='样本数',title=f'每 {binma} Ma 时间窗样本量');savefig(fig,figs/'02_每100Ma样本量.png')
    fig,ax=plt.subplots(figsize=(9,4.5));ax.bar(comp['元素'],comp['完整率']*100,color='#4c9f70');ax.set(ylabel='完整率（%）',title='14 个模型元素完整率');savefig(fig,figs/'03_14元素完整率.png')
    cnt=pd.read_csv(pred,usecols=['N_FEATURES_MEASURED']).N_FEATURES_MEASURED.value_counts().sort_index();fig,ax=plt.subplots();ax.bar(cnt.index,cnt.values,color='#7b6fd0');ax.set(xlabel='每颗锆石实测模型元素数',ylabel='样本数',title='每颗锆石实测元素数量');savefig(fig,figs/'04_实测元素数量分布.png')
    loc=pd.read_csv(pred,usecols=['LONGITUDE_CENTER','LATITUDE_CENTER']).dropna();fig,ax=plt.subplots(figsize=(10,4.8));ax.scatter(loc.LONGITUDE_CENTER,loc.LATITUDE_CENTER,s=3,alpha=.3);ax.set(xlim=(-180,180),ylim=(-90,90),xlabel='经度',ylabel='纬度',title='全球碎屑锆石候选空间分布');ax.grid(alpha=.2);savefig(fig,figs/'05_全球经纬度分布.png')
    labels=['宽松候选','碎屑候选','高完整度碎屑候选'];fig,ax=plt.subplots();ax.bar(labels,[stat[x] for x in labels],color=['#999999','#3978a8','#d55e00']);ax.tick_params(axis='x',rotation=15);ax.set(ylabel='数量',title='候选样本数量对比');savefig(fig,figs/'06_候选数量对比.png')
    fig,ax=plt.subplots(figsize=(10,5));x=t['年龄中心_Ma'];ax.plot(x,t['MLP_中位数'],label='MLP',alpha=.75);ax.plot(x,t['ExtraTrees_中位数'],label='ExtraTrees',alpha=.75);ax.plot(x,t['双模型均值_三窗平滑'],label='双模型均值（3窗平滑）',color='black',lw=2);ax.fill_between(x,t['双模型均值_Q25'],t['双模型均值_Q75'],alpha=.18,color='gray',label='双模型四分位范围');ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='年龄（Ma，向右更古老）',ylabel='预测 ΔFMQ',title='0–4000 Ma 全球碎屑锆石氧逸度演化');ax.legend();savefig(fig,figs/'07_双模型氧逸度演化.png')
    fig,ax=plt.subplots(figsize=(10,4.5));ax.plot(x,t['变化率_每100Ma'],color='#c43c39');ax.axhline(0,color='black',lw=.8);ax.fill_between(x,0,t['变化率_每100Ma'],where=t['变化率_每100Ma']>=0,color='#d55e00',alpha=.25);ax.fill_between(x,0,t['变化率_每100Ma'],where=t['变化率_每100Ma']<0,color='#3978a8',alpha=.25);ax.set(xlabel='年龄（Ma，向右更古老）',ylabel='ΔFMQ / 100 Ma',title='氧逸度变化率（沿年龄增大方向）');savefig(fig,figs/'08_氧逸度变化率.png')
    fig,ax=plt.subplots();ax.plot(x,t['模型差异中位数'],color='#8c564b');ax.set(xlabel='年龄（Ma）',ylabel='|MLP−ExtraTrees|',title='两模型预测分歧');savefig(fig,figs/'09_模型差异.png')
    summary=pd.DataFrame([stat]);summary.to_csv(out/'08_自动绘图统计表.csv',index=False,encoding='utf-8-sig')

def pack_and_send(out:Path, config_path:Path):
    archive=Path(shutil.make_archive(str(out), 'zip', root_dir=out.parent, base_dir=out.name))
    print(f'结果已打包：{archive}')
    if not config_path.exists(): print('未找到 QQ 邮箱配置，跳过发送。');return
    cfg=json.loads(config_path.read_text(encoding='utf-8-sig'))
    if not cfg.get('enabled',False): print('QQ 邮箱发送未启用，跳过发送。');return
    sender=str(cfg.get('sender_qq_email','')).strip();recipient=str(cfg.get('recipient_email') or sender).strip();code=str(cfg.get('smtp_authorization_code','')).strip()
    if not sender or not recipient or not code: print('未填写 QQ 邮箱 SMTP 授权码，结果已打包但未发送。');return
    part_bytes=int(cfg.get('part_size_mb',20))*1024*1024;files=[]
    if archive.stat().st_size<=part_bytes: files=[archive]
    else:
        with archive.open('rb') as src:
            i=1
            while block:=src.read(part_bytes):
                p=archive.with_suffix(archive.suffix+f'.part{i:03d}');p.write_bytes(block);files.append(p);i+=1
        helper=out.parent/'合并QQ邮件分卷.py'
        helper.write_text("from pathlib import Path\nparts=sorted(Path('.').glob('*.zip.part*'))\nif not parts:raise SystemExit('请把全部分卷放在同一文件夹')\nout=Path(parts[0].name.split('.part')[0])\nwith out.open('wb') as w:\n    for p in parts:w.write(p.read_bytes())\nprint('合并完成：',out)\n",encoding='utf-8-sig')
    for i,p in enumerate(files,1):
        msg=EmailMessage();msg['From']=sender;msg['To']=recipient;msg['Subject']=f'GEOROC 双模型预测结果（{i}/{len(files)}）'
        digest=hashlib.sha256(p.read_bytes()).hexdigest();msg.set_content(f'自动发送的 GEOROC 运行结果。\n附件：{p.name}\nSHA256：{digest}')
        msg.add_attachment(p.read_bytes(),maintype='application',subtype='octet-stream',filename=p.name)
        if i==1 and len(files)>1:
            helper=out.parent/'合并QQ邮件分卷.py';msg.add_attachment(helper.read_bytes(),maintype='text',subtype='x-python',filename=helper.name)
        with smtplib.SMTP_SSL('smtp.qq.com',465,timeout=120) as s:s.login(sender,code);s.send_message(msg)
        print(f'QQ 邮箱发送完成：{i}/{len(files)}')

def main():
    a=args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    print('阶段 1/4：数据清洗（不会训练模型）')
    if a.data_mode=='paper': paths,comp,stat=clean_paper(Path(a.paper_data).resolve(),out,a.min_features,a.strict_features)
    else: paths,comp,stat=clean(Path(a.input).resolve(),out,a.chunksize,a.min_features,a.strict_features)
    print('阶段 2/4：加载冻结模型并预测');models=load_models(Path(a.models).resolve());pred=predict_file(paths['detrital'],out,models,a.chunksize)
    print('阶段 3/4：统计与自动绘图');plots(pred,comp,stat,out,a.bin_ma)
    (out/'运行完成.txt').write_text('清洗、双模型预测、氧逸度演化及变化率绘图均已完成。\n',encoding='utf-8-sig')
    print('阶段 4/4：打包结果并发送 QQ 邮箱');pack_and_send(out,Path('QQ邮箱配置.json').resolve());print('\n全部完成：',out)

if __name__=='__main__':
    try:main()
    except Exception as e:print(f'\n运行失败：{e}',file=sys.stderr);raise
