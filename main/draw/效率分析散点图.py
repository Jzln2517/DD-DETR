import matplotlib.pyplot as plt
import pandas as pd

# 数据准备
data = {
    'Method': [
        'YOLOv8-S', 'YOLOv10-S', 'YOLOv12-S', 'YOLOv13-S',
        'YOLOX-Tiny', 'FBRT-YOLO-S', 'FBRT-YOLO-L', 'RetinaNet-R50',
        'Faster R-CNN-R50', 'Cascade R-CNN-R50', 'ATSS-R50', 'RT-DETR-R18(Baseline)', 'Ours'
    ],
    'GFLOPs': [28.5, 64.5, 21.2, 20.1, 7.5, 22.9, 58.7, 210.0, 208.0, 236.0, 110.0, 57.0, 49.2],
    'AP': [17.3, 17.9, 17.6, 16.7, 14.8, 18.3, 19.6, 16.4, 19.4, 19.7, 20.4, 18.5, 21.4]
}
df = pd.DataFrame(data)

# 创建散点图
plt.figure(figsize=(12, 6))  # 增大画布尺寸以减少重叠
plt.scatter(df['GFLOPs'], df['AP'], color='blue', alpha=0.7, s=80, label='Other Models')  # 设置点的大小和透明度

# 特殊标记“Ours”点：红色星形，更大尺寸
ours_data = df[df['Method'] == 'Ours']
plt.scatter(ours_data['GFLOPs'], ours_data['AP'], color='red', s=300, marker='*', label='Our Model (Ours)')

# 添加方法名标签，使用偏移避免重叠
for i, row in df.iterrows():
    offset_x = 0
    offset_y = 0
    # 根据点位置调整偏移，减少重叠
    if row['Method'] == 'Ours':
        offset_x, offset_y = 8, 8  # 为“Ours”设置较大偏移
    elif row['GFLOPs'] > 200 and row['AP'] > 19:  # 高GFLOPs点向右偏移
        offset_x, offset_y = -9, 7
    elif row['GFLOPs'] > 200 and row['AP'] < 17:  # 高GFLOPs点向右偏移
        offset_x, offset_y = -9, 7
    elif row['GFLOPs'] > 150:  # 高GFLOPs点向右偏移
        offset_x, offset_y = 5, 0
    elif row['AP'] < 17:  # 低AP点向下偏移
        offset_x, offset_y = 0, -15
    else:
        offset_x, offset_y = 5, 5  # 默认偏移
    
    plt.annotate(
        row['Method'],
        (row['GFLOPs'], row['AP']),
        textcoords="offset points",
        xytext=(offset_x, offset_y),
        ha='center',
        fontsize=9,
        alpha=0.9
    )

# 设置坐标轴和标题
plt.xlabel('GFLOPs', fontsize=14, fontweight='bold')
plt.ylabel('AP', fontsize=14, fontweight='bold')
# plt.title('Accuracy-Efficiency Trade-off on VisDrone2019 Dataset', fontsize=16, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')  # 添加图例

# 调整布局
plt.tight_layout()

# 保存为EPS格式
plt.savefig("accuracy_efficiency_tradeoff.eps", format="eps")

# 显示图形
plt.show()
