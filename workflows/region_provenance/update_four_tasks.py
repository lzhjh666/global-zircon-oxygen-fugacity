from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from io import BytesIO
from docx import Document
from docx.shared import Inches
import pandas as pd

BASE=Path(__file__).parent
OUT=BASE/"completed"
Z=ZipFile(next(BASE.glob("*.zip")))

def setp(d,i,s): d.paragraphs[i].text=s

for lang,name in [("cn","全球锆石氧逸度演化_中文修订稿.docx"),
                  ("en","Global_Zircon_Redox_Revised_Manuscript_EN.docx")]:
    d=Document(BytesIO(Z.read(name)))
    if lang=="cn":
        setp(d,7,"Zou等[1]建立多元素锆石机器学习氧逸度计。Rui Wang等[9]以锆石氧化还原指标研究全球碎屑锆石与超大陆周期，Xiangsong Wang等[6]进一步报道全球岩浆氧化状态曲线。本文的科学问题是：在地理隔离验证、冻结双模型、逐颗溯源、来源等权区间和预定年龄窗口下，全球时间信号的证据强度如何变化？这种设计将早期氧化上升识别为跨窗口一致的信号，同时使中元古代低值回到方向性观察的位置；不以不同论文的模型误差直接排座次。")
        setp(d,10,"本文连接三层证据：按地质地点隔离的模型检验、冻结模型在冈底斯独立碎屑记录中的区域应用、来源等权的全球长期重建。与已有全球锆石氧化态研究[6,9]相比，我们检验全球曲线在来源构成、地区组成和时窗定义改变后的解释力度。")
        setp(d,16,"数据筛选与质量控制。Wang母集20,347条，旧版候选10,844条；组成域移出588条、重复移出4条，保留10,252条；GEOROC新增627条，最终10,879条。P>750 ppm、La>1 ppm或Th/U<0.1为排除条件，单项未测不触发。新增627条由原始元素值重建的三项逐颗QC标记全部为False，与总QC通过标记一致；在补充表逐ID公开。GEOROC新增条目有沉积岩性证据，但碎屑颗粒的具体岩浆物源仍需结合原始文献判定；旧汇编10,252条缺统一的逐颗碎屑/原位身份字段。因此本文称全球锆石汇编，不把全部记录宣称为逐颗已确认碎屑锆石。")
        setp(d,23,"来源、地区组成与适用域。最终10,879条在100 Ma窗口中的颗粒数、来源数及最大来源占比见图3；逐颗保守大陆分类及年龄窗构成见补充图S2和表S2。3000–3300 Ma目标窗188颗、21个来源，3500–4000 Ma参照窗225颗、10个来源；前者主要分布在非洲84颗、大洋洲51颗、亚洲41颗，后者以大洋洲166颗为主。分别排除任一已分类大陆后，早期差值仍为正（+0.422至+0.681 ΔFMQ；表S2），说明观察到的上升不是某一个大陆单独造成，但两个时期的地理组成并不相同。完整14元素、较深域内子集及逐来源删除的检验继续提供样本和特征层面的支撑。")
        setp(d,43,"与现有研究的关系。Zou等[1]建立了多元素机器学习氧逸度计；Rui Wang等[9]与Xiangsong Wang等[6]已开展全球锆石氧化态重建。本文的推进是将地点隔离验证、完全留出地区、冻结后独立冈底斯应用、来源等权Bootstrap、地区组成和窗口敏感性连成同一可复核的推断链。早期上升在三套时窗和逐大陆删除下保持正向；中元古代低值则是观察到的较弱方向。")
        setp(d,46,"本汇编代表可形成和保存锆石的大陆岩浆系统。旧汇编缺统一的逐颗碎屑身份字段，新增627条有沉积岩性线索而未逐颗反演岩浆源；由此采用“全球锆石记录”的表述。本文以来源等权、地区组成表和逐大陆删除检验量化采样结构的影响，并在补充表保留逐颗身份与QC可核查程度。")
        setp(d,47,"冈底斯区域检验。Tang等[2]东冈底斯四条支流现代河沙的1,639颗锆石中，1,505颗符合本文元素与QC口径，1,490颗处于模型元素适用域内。在两个预定年龄段均有数据的三条河流中，10–25 Ma相对40–65 Ma的来源内中位预测值均升高，依次为+0.514、+0.500和+0.268 ΔFMQ（补充图S3）。这一方向与东冈底斯磁铁矿—钛铁矿和角闪石的独立阶段记录一致[10]。参考矿物测值与锆石并非逐颗配对，故该案例检验时间方向而非逐颗绝对误差。")
        setp(d,52,"按地点隔离验证、冻结双模型、逐颗溯源与来源等权重建，10,879条锆石构成可复核的长期氧逸度记录。最清晰的信号是早期大陆岩浆氧化上升：其来源Bootstrap区间在三套时窗中均排除零，逐大陆剔除后仍为正。冈底斯四河流碎屑锆石的区域应用独立重现了年轻阶段氧化增强方向。晚太古代主窗口高值具有区间支持；中元古代相对低值与年轻阶段恢复则为方向性观察。")
        ref9=d.add_paragraph("[9] Wang, R., Wu, S.-C., Weinberg, R. F., Collins, W. J. & Cawood, P. A. Zircons reveal the history of fluctuations in oxidation state of crustal magmatism and supercontinent cycle. Science Bulletin 69, 97–102 (2024). doi:10.1016/j.scib.2023.10.034.")
        ref10=d.add_paragraph("[10] Wang, R. et al. Increasing magmatic oxidation state from Paleocene to Miocene in the eastern Gangdese Belt, Tibet. Economic Geology 109, 1943–1965 (2014). doi:10.2113/econgeo.109.7.1943.")
        d.paragraphs[64]._p.addprevious(ref9._p)
        d.paragraphs[65]._p.addprevious(ref10._p)
        d.add_paragraph("补充图S2  各100 Ma窗口的保守大陆分类与粒数。未分配类别单独显示；逐大陆删除及来源数见表S2。")
        d.add_picture(str(OUT/"FigureS_100Ma大陆构成.png"),width=Inches(5.8))
        d.add_paragraph("补充图S3  东冈底斯四条河流中三个具有两时期记录的来源，点旁标注各阶段域内颗粒数。")
        d.add_picture(str(Path("gangdese_results/FigureS_域内冈底斯配对与样本数.png")),width=Inches(5.8))
        out=OUT/"全球锆石氧逸度演化_中文_四项完善稿.docx"
    else:
        setp(d,7,"Zou et al. [1] developed a multi-element machine-learning zircon oxybarometer. Rui Wang et al. [9] reconstructed redox variation from a global detrital zircon record, and Xiangsong Wang et al. [6] reported another global magmatic redox trajectory. We ask how geographical isolation in validation, frozen prediction, grain-level provenance, source bootstrap and predefined age windows change the evidential strength of long-term signals. These steps distinguish a window-robust early rise from a directional mid-Proterozoic low.")
        setp(d,10,"We connect locality-isolated calibration assessment, a frozen-model application to independent Gangdese detrital zircons, and source-weighted global reconstruction. The advance over previous global zircon redox records [6,9] is an explicit assessment of source composition, geography, uncertainty and age-window dependence for each temporal claim.")
        setp(d,16,"Data selection and provenance. The Wang compilation includes 20,347 initial analyses; 10,844 were candidates, 588 were removed as out of domain and four as potential duplicates, leaving 10,252. A GEOROC screen adds 627, giving 10,879. P>750 ppm, La>1 ppm and Th/U<0.1 are exclusion criteria when measured; an unmeasured individual quantity alone does not trigger exclusion. Rebuilt per-grain QC flags for the 627 GEOROC additions are all False and agree with the overall QC-pass field (Supplementary Table S2). The additions have sedimentary lithology context, but their magmatic source is not individually reconstructed; the older compilation lacks a uniform grain-level detrital versus in-situ identifier. We therefore call the application a global zircon compilation.")
        setp(d,23,"Source composition, geography and domain. Grains, independent sources and the dominant-source share per 100-Myr bin appear in Figure 3; conservative continental assignments are in Supplementary Figure S2 and Table S2. The 3000–3300 Ma target comprises 188 grains and 21 sources; the 3500–4000 Ma baseline has 225 grains and ten sources. Target grains include 84 from Africa, 51 from Oceania and 41 from Asia; the baseline includes 166 from Oceania. Omitting each classified continent in turn leaves a positive early rise (+0.422 to +0.681 ΔFMQ). Thus no single continent creates this contrast, although the regional composition changes between periods. Feature-complete and deep-in-domain subsets provide complementary checks.")
        setp(d,43,"Relation to previous work. Zou et al. [1] established multi-element machine-learning oxybarometry; Rui Wang et al. [9] and Xiangsong Wang et al. [6] had already published global zircon redox histories. Here, locality-isolated validation, a full regional holdout, a frozen-model Gangdese application, source-equal bootstrap, geographic composition and window sensitivity form a reproducible chain of inference. The early rise remains positive across all three windows and after omission of any classified continent; the mid-Proterozoic relative low remains a weaker directional observation.")
        setp(d,46,"The global compilation samples zircon-forming and preserved magmatic systems. The older component lacks a consistent grain-level detrital identity field, while sedimentary lithology contextualizes the 627 new analyses without uniquely determining their magmatic sources. We use the term global zircon record and provide grain-level identity and QC audit fields. Source weighting, continental counts and continent-omission tests quantify aspects of geographic sampling.")
        setp(d,47,"Gangdese regional application. Of 1,639 detrital zircons collected from four tributaries draining the eastern Gangdese batholith by Tang et al. [2], 1,505 meet our feature and QC rules and 1,490 pass the frozen feature-domain threshold. In all three rivers represented in both predefined intervals, median predicted ΔFMQ increases from 40–65 to 10–25 Ma by +0.514, +0.500 and +0.268, respectively (Supplementary Fig. S3). This direction agrees with independent magnetite–ilmenite and amphibole records of greater Miocene oxidation [10]. Mineral estimates are not paired reference labels for these individual zircons, so the regional comparison evaluates temporal direction.")
        setp(d,52,"Locality-isolated validation, frozen predictors, traceable analyses and source-equal aggregation provide a reproducible 10,879-grain global zircon redox history. The clearest signal is early continental magmatic oxidation: source-bootstrap intervals exclude zero in three windows and the contrast remains positive after omitting any classified continent. A separate Gangdese detrital record reproduces the direction of regional young-stage oxidation. The primary late Archean high has interval support, whereas the mid-Proterozoic low and younger recovery remain directional observations.")
        ref9=d.add_paragraph("[9] Wang, R., Wu, S.-C., Weinberg, R. F., Collins, W. J. & Cawood, P. A. Zircons reveal the history of fluctuations in oxidation state of crustal magmatism and supercontinent cycle. Science Bulletin 69, 97–102 (2024). doi:10.1016/j.scib.2023.10.034.")
        ref10=d.add_paragraph("[10] Wang, R. et al. Increasing magmatic oxidation state from Paleocene to Miocene in the eastern Gangdese Belt, Tibet. Economic Geology 109, 1943–1965 (2014). doi:10.2113/econgeo.109.7.1943.")
        d.paragraphs[64]._p.addprevious(ref9._p)
        d.paragraphs[65]._p.addprevious(ref10._p)
        d.add_paragraph("Supplementary Figure S2. Conservative continental assignments and grain counts by 100-Myr bin. Unassigned grains are displayed separately.")
        d.add_picture(str(OUT/"FigureS_100Ma大陆构成.png"),width=Inches(5.8))
        d.add_paragraph("Supplementary Figure S3. Paired within-river Gangdese contrasts, with the number of in-domain grains in each stage.")
        d.add_picture(str(Path("gangdese_results/FigureS_域内冈底斯配对与样本数.png")),width=Inches(5.8))
        out=OUT/"Global_Zircon_Redox_Four_Tasks_Revised.docx"
    d.save(out)
    print(out)
