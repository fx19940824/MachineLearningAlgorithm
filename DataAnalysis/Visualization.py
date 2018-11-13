from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def PlotBox(df):
    df = pd.DataFrame(np.random.rand(10, 2), columns=['A', 'B'])
    
    plt.figure(figsize=(10, 4))

    f = df.boxplot(
        sym='o',  # 异常点形状，参考marker
        vert=True,  # 是否垂直
        whis=1.5,  # IQR，默认1.5，也可以设置区间比如[5,95]，代表强制上下边缘为数据95%和5%位置
        patch_artist=True,  # 上下四分位框内是否填充，True为填充
        meanline=False,
        showmeans=True,  # 是否有均值线及其形状
        showbox=True,  # 是否显示箱线
        showcaps=True,  # 是否显示边缘线
        showfliers=True,  # 是否显示异常值
        notch=False,  # 中间箱体是否缺口
        return_type='dict'  # 返回类型为字典
    )
    plt.title('boxplot')

    for box in f['boxes']:
        box.set(color='b', linewidth=1)  # 箱体边框颜色
        box.set(facecolor='b', alpha=0.5)  # 箱体内部填充颜色
    for whisker in f['whiskers']:
        whisker.set(color='k', linewidth=0.5, linestyle='-')
    for cap in f['caps']:
        cap.set(color='gray', linewidth=2)
    for median in f['medians']:
        median.set(color='DarkBlue', linewidth=2)
    for flier in f['fliers']:
        flier.set(marker='o', color='y', alpha=0.5)
    # boxes, 箱线
    # medians, 中位值的横线,
    # whiskers, 从box到error bar之间的竖线.
    # fliers, 异常值
    # caps, error bar横线
    # means, 均值的横线,

def PlotHist(s):
    # 直方图
    s.hist(bins = 20,
        histtype = 'bar',
        align = 'mid',
        orientation = 'vertical',
        alpha=0.5,
        normed =True)
    # 密度图
    s.plot(kind='kde',style='k--')