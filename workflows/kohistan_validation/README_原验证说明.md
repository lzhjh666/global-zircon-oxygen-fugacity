# Kohistan 2026全新独立外部验证包

## 为什么选它

- 数据来自2026年中央Kohistan Arc研究，晚于现有模型训练资料。
- 7个岩体共提供139颗有效锆石元素数据；其中5个岩体具有角闪石独立氧逸度约束，共113颗可评分锆石。
- Kohistan、Matiltan、Gabral、Deshai、Diwangar和Shahibagh均不在当前训练地点清单中。
- 氧逸度由角闪石Ridolfi (2021)方法约束，不是用锆石Ce-U-Ti公式生成，因此可以作为真正的外部检验。
- 模型所需17元素全部存在：Ce、Nd、Sm、Eu、Gd、Tb、Dy、Ho、Er、Tm、Yb、Lu、Th、U、Hf、Ti、Y。

## 如何运行

1. 打开“模型文件放这里”，按其中说明复制至少一个锁定模型。
2. 双击 `开始验证_Kohistan_约1至3分钟.bat`。
3. 把生成的 `Kohistan独立外部验证结果_日期时间` 文件夹压缩后发回。

不需要重新训练模型，也不要把这批数据加入训练集后再称为外部验证。

## 评分规则

- 113颗锆石属于5个岩体，不把它们视为113个独立地点。
- 每个岩体先取逐颗预测中位数，再对5个岩体等权计算R²、MSE和RMSE。
- MSE与导师要求的0.4进行比较。
- 论文Table 2的组平均logfO2只报告到整数，标签存在舍入误差，解释时应同时查看逐岩体残差。

## 数据来源与换算

- Ahmad, T., Drüppel, K., Jan, M.Q., Zeh, A., Hezel, D.C. (2026). *Tectono-magmatic evolution of the central Kohistan Arc, NW Pakistan: Granitoid emplacement revealed by mineral chemistry*. Journal of Asian Earth Sciences, 303, 107049.
- DOI：https://doi.org/10.1016/j.jseaes.2026.107049
- 锆石元素：Supplementary Data 2，Table S1.4。
- 独立氧逸度：正文Table 2，Ridolfi (2021)角闪石方法。
- ΔFMQ换算：Frost (1991)高温FMQ缓冲式，采用论文各岩体平均角闪石温度和压力。

## 重要限制

角闪石和锆石可能记录岩浆演化的不同结晶阶段，论文氧逸度组均值也经过整数舍入。因此这是目前非常合适的独立外部检验，但仍应按“岩体级外部验证”解释，不能宣称为每颗锆石的独立实测真值。
