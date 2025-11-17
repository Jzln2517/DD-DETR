import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

def parse_xml_get_wh(xml_path):
    """
    从 XML 文件中提取所有目标框的宽度和高度
    """
    wh_list = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            w = xmax - xmin
            h = ymax - ymin
            if w > 0 and h > 0:
                wh_list.append((w, h))
    except Exception as e:
        print(f"[解析错误] {xml_path}：{e}")
    return wh_list

def collect_all_wh(xml_folder):
    """
    遍历所有 XML 文件，收集所有目标框的宽高
    """
    all_wh = []
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith('.xml')]
    for xml_file in tqdm(xml_files, desc="解析 XML 文件"):
        xml_path = os.path.join(xml_folder, xml_file)
        wh = parse_xml_get_wh(xml_path)
        all_wh.extend(wh)
    return all_wh

def plot_wh_heatmap(wh_list, save_path=None, show_plot=False):
    """
    绘制目标框宽高分布热力图
    """
    if not wh_list:
        print("没有目标框数据可用于绘图。")
        return

    widths = [w for w, h in wh_list]
    heights = [h for w, h in wh_list]

    plt.figure(figsize=(7, 8))
    plt.hexbin(widths, heights, gridsize=300, cmap='magma', norm=LogNorm())
    plt.colorbar(label='Count')
    plt.title("All anchors' distribution", fontsize=18)
    plt.xlabel("Width/(Pixel)", fontsize=14)
    plt.ylabel("High/(Pixel)", fontsize=14)
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    if show_plot:
        plt.show()

    plt.close()

if __name__ == "__main__":
    # === 配置部分 ===
    xml_folder = "/home/ubuntu/datasets/visdrone/xml/train"  # 修改为你实际的 XML 路径
    output_path = "/home/ubuntu/proejcts/RTDETR-main/dataset/VisdroneImage/anchor_distribution.png"
    show_plot = False  # 设置为 True 会在窗口中弹出图像

    # === 主程序流程 ===
    if not os.path.exists(xml_folder):
        print(f"[错误] XML 文件夹不存在: {xml_folder}")
    else:
        all_wh = collect_all_wh(xml_folder)
        if len(all_wh) == 0:
            print("[警告] 没有提取到任何宽高数据，请检查 XML 文件内容")
        else:
            plot_wh_heatmap(all_wh, save_path=output_path, show_plot=show_plot)
