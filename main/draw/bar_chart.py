import os
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 路径设置（请根据你的实际路径修改）
ANNOTATIONS_DIR = '/home/ubuntu/datasets/visdrone/xml/train'
SAVE_PATH = '/home/ubuntu/proejcts/RTDETR-main/dataset/VisdroneImage/class_distribution_styled.png'

# 优雅专业配色（适合论文风格）
color_map = {
    'pedestrian': '#4E79A7',
    'people': '#F28E2B',
    'bicycle': '#E15759',
    'car': '#76B7B2',
    'van': '#59A14F',
    'truck': '#EDC948',
    'tricycle': '#B07AA1',
    'awning-tricycle': '#FF9DA7',
    'bus': '#9C755F',
    'motor': '#BAB0AC'
}

# 统计类别数量
class_counts = Counter()
for file in os.listdir(ANNOTATIONS_DIR):
    if not file.endswith('.xml'):
        continue
    tree = ET.parse(os.path.join(ANNOTATIONS_DIR, file))
    root = tree.getroot()
    for obj in root.findall('object'):
        cls = obj.find('name').text
        class_counts[cls] += 1

# 排序类别（按数量从多到少）
sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
labels = [item[0] for item in sorted_items]
counts = [item[1] for item in sorted_items]
colors = [color_map.get(label, '#cccccc') for label in labels]  # 默认灰色

# 全局美化设置
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# 绘制图像
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, counts, color=colors, edgecolor='black', linewidth=0.6)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01, f'{height}',
             ha='center', va='bottom', fontsize=10)

# 样式设置
plt.title('VisDrone2019 Object Category Distribution', pad=15)
plt.xlabel('Category')
plt.ylabel('Number')
plt.xticks(rotation=30, ha='right')

# 图例
legend_handles = [mpatches.Patch(color=color_map[k], label=k) for k in labels]
plt.legend(handles=legend_handles, loc='upper right', fontsize=10)

# 图表边框优化
ax = plt.gca()
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 自动调整布局
plt.tight_layout()

# 保存图像
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 柱状图已保存到: {SAVE_PATH}")
