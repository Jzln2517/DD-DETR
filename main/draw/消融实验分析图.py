import matplotlib.pyplot as plt
import numpy as np

# 方法名称（与消融表顺序一致）
methods = [
    "Baseline",
    "+LFI",
    "+DSA",
    "+DFEM",
    "+LFI + DSA",
    "+LFI + DFEM",
    "+DSA + DFEM",
    "Full Model (LFI+DSA+DFEM)"
]

# 指标数据（来自消融表格）
AP    = [18.5, 20.9, 20.4, 20.7, 21.0, 21.2, 20.6, 21.4]
AP50  = [32.8, 35.5, 34.8, 35.1, 35.6, 35.6, 34.8, 36.3]
AP_S  = [11.2, 12.4, 11.7, 12.1, 12.4, 12.3, 12.2, 12.9]
GFLOPs = [57.0, 46.9, 57.0, 59.3, 47.0, 49.2, 59.4, 49.2]
Params = [19.9, 13.8, 19.7, 22.7, 13.7, 16.7, 22.5, 16.5]

x = np.arange(len(methods))
bar_width = 0.2          # 三个柱子，宽度缩小
offset = bar_width       # 偏移量

fig, ax1 = plt.subplots(figsize=(12, 6))

# ------- 左轴：精度指标（柱状图） -------
bars_ap   = ax1.bar(x - offset, AP,   width=bar_width, label="AP",        color="#4C72B0")
bars_ap50 = ax1.bar(x,           AP50, width=bar_width, label="AP$_{50}$", color="#55A868")
bars_aps  = ax1.bar(x + offset,  AP_S, width=bar_width, label="AP$_S$",   color="#8B008B")

ax1.set_ylabel("AP / AP$_{50}$ / AP$_S$ (%)", fontsize=12, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=20, fontsize=10)
ax1.tick_params(axis="y", labelsize=11)
ax1.set_ylim(0, 42)   # 为最高值（AP50约36.3）留出空间

# ------- 右轴：效率指标（折线图） -------
ax2 = ax1.twinx()
line_gflops, = ax2.plot(x, GFLOPs, marker="o", color="#C44E52", linewidth=2,
                        markersize=7, label="GFLOPs")
line_params, = ax2.plot(x, Params, marker="D", color="#E69F00", linewidth=2,
                        markersize=7, linestyle="-.", label="Params (M)")

ax2.set_ylabel("GFLOPs / Params (M)", fontsize=12, fontweight="bold")
ax2.tick_params(axis="y", labelsize=11)
ax2.set_ylim(0, 65)

# ------- 图例合并 -------
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2,
           loc="upper left", fontsize=9, ncol=2)

# 网格（仅左轴）
ax1.grid(True, linestyle="--", alpha=0.6, axis="y")

plt.tight_layout()
plt.savefig("ablation_ap_gflops_params.eps", format="eps")
plt.show()