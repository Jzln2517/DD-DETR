import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from collections import defaultdict
from tqdm import tqdm

def parse_xml_per_class(xml_path):
    """
    从 XML 文件中提取所有目标框的宽高和类别
    返回：dict[class_name] = [(w, h), ...]
    """
    wh_dict = defaultdict(list)
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            cls = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            w = xmax - xmin
            h = ymax - ymin
            if w > 0 and h > 0:
                wh_dict[cls].append((w, h))
    except Exception as e:
        print(f"[解析错误] {xml_path}: {e}")
    return wh_dict

def collect_class_wh(xml_folder):
    """
    遍历所有 XML 文件，收集所有类别的宽高列表
    返回：dict[class_name] = [(w, h), ...]
    """
    class_wh = defaultdict(list)
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith('.xml')]
    for xml_file in tqdm(xml_files, desc="解析 XML 文件"):
        xml_path = os.path.join(xml_folder, xml_file)
        file_data = parse_xml_per_class(xml_path)
        for cls, whs in file_data.items():
            class_wh[cls].extend(whs)
    return class_wh

def plot_class_wh(class_wh_dict, save_dir):
    """
    为每个类别绘制热力图并保存
    """
    os.makedirs(save_dir, exist_ok=True)

    for cls_name, wh_list in class_wh_dict.items():
        if not wh_list:
            continue

        widths = [w for w, h in wh_list]
        heights = [h for w, h in wh_list]

        plt.figure(figsize=(7, 8))
        plt.hexbin(widths, heights, gridsize=300, cmap='magma', norm=LogNorm())
        plt.colorbar(label='Count')
        plt.title(f"{cls_name} anchors's distribution", fontsize=18)
        plt.xlabel("Width/(Pixel)", fontsize=14)
        plt.ylabel("High/(Pixel)", fontsize=14)
        plt.grid(True)

        out_path = os.path.join(save_dir, f"{cls_name}_anchor_distribution.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存: {out_path}")

if __name__ == "__main__":
    # === 配置路径 ===
    xml_folder = "/home/ubuntu/datasets/visdrone/xml/train"  # 修改为你的 XML 路径
    save_dir = "/home/ubuntu/proejcts/RTDETR-main/dataset/VisdroneImage/class_distributions"

    if not os.path.exists(xml_folder):
        print(f"[错误] XML 文件夹不存在: {xml_folder}")
    else:
        class_wh = collect_class_wh(xml_folder)
        plot_class_wh(class_wh, save_dir)
