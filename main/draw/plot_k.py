import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ==========================================
# 1. 绘制 DSA 模块的真实 k 值分布图 (Violin Plot + Swarm Plot)
# ==========================================
def plot_dsa_k_distribution():
    # 填入您刚刚提取出的真实数据
    data_sparse = [208, 228, 229, 235, 233, 232, 229, 231, 229, 229, 228, 231, 237, 233, 219, 220]
    data_dense = [241, 236, 230, 233, 234, 234, 232, 226, 236, 232, 231, 229, 235, 236, 239]
    data_complex = [233, 225, 229, 230, 238, 251, 251, 236, 233, 236, 240, 234, 240, 237, 243]
    data_lowlight = [238, 240, 238, 237, 230, 251, 249, 241, 233, 242, 234, 236, 236, 234, 241]

    # 合并数据
    data = np.concatenate([data_sparse, data_dense, data_complex, data_lowlight])
    
    # 生成对应的标签
    labels = (['Sparse'] * len(data_sparse) + 
              ['Dense'] * len(data_dense) + 
              ['Complex'] * len(data_complex) + 
              ['Low-light'] * len(data_lowlight))
    
    plt.figure(figsize=(9, 6))
    # 设置极具学术感的风格
    sns.set_theme(style="ticks", palette="pastel")
    
    # 画箱线图 (Boxplot) 更加直观地展示均值和上下四分位数
    ax = sns.boxplot(x=labels, y=data, width=0.5, showfliers=False,
                     boxprops=dict(alpha=0.8, edgecolor='black', linewidth=1.5),
                     medianprops=dict(color='red', linewidth=2))
    
    # 叠加散点图 (Stripplot)，把您提出来的每一个真实数据点都画上去，增强真实感
    sns.stripplot(x=labels, y=data, size=6, color="black", alpha=0.6, jitter=True)
    
    # 设置图表标题和坐标轴
    # plt.title("Distribution of Adaptive $k$ across Scene Types", fontsize=15, fontweight='bold', pad=15)
    plt.ylabel("Retained Active Tokens ($k$)", fontsize=13, fontweight='bold')
    plt.xlabel("Scene Category", fontsize=13, fontweight='bold')
    
    # 美化网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine(trim=True, left=True)
    
    plt.tight_layout()
    plt.savefig("dsa_real_k_distribution.pdf", dpi=300, bbox_inches='tight') # 保存高清 PDF
    plt.show()
    print("成功保存真实数据分布图: dsa_real_k_distribution.pdf")

if __name__ == "__main__":
    plot_dsa_k_distribution()
