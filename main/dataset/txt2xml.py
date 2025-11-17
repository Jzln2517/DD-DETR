import os
import xml.etree.ElementTree as ET
from PIL import Image
 
# 类别编号和名称的映射
class_mapping = {       # 替换为自己的类别编号和名称的映射
    '0': 'pedestrian',
    '1': 'people',
    '2': 'bicycle',
    '3': 'car',
    '4': 'van',
    '5': 'truck',
    '6': 'tricycle',
    '7': 'awning-tricycle',
    '8': 'bus',
    '9': 'motor'

}
 
# 源文件夹和目标文件夹路径
source_folder = r'/home/ubuntu/datasets/visdrone/labels/train'      # 替换为存放txt标签的文件夹路径
target_folder = r'/home/ubuntu/datasets/visdrone/xml/train'      # 替换为保存xml文件的文件夹路径
image_folder = r'/home/ubuntu/datasets/visdrone/images/train'    # 替换为存放图片的文件夹路径
 
# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)
 
# 遍历源文件夹中的所有TXT文件
for txt_file in os.listdir(source_folder):
    if txt_file.endswith('.txt'):
        # 构建完整的文件路径
        txt_path = os.path.join(source_folder, txt_file)
        
        # 创建XML文件名
        xml_file_name = txt_file.replace('.txt', '.xml')
        xml_file_path = os.path.join(target_folder, xml_file_name)
        
        # 获取图像文件名（假设图像文件和TXT文件同名）
        image_file_name = txt_file.replace('.txt', '.jpg')  # 图像文件是JPG格式
        image_path = os.path.join(image_folder, image_file_name)
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue
        
        # 使用PIL库获取图像尺寸
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        
        # 解析TXT文件并创建XML结构
        with open(txt_path, 'r') as txt_file:
            tree = ET.ElementTree(ET.Element('annotation'))
            root = tree.getroot()
            
            # 添加folder, filename, path元素
            folder = ET.SubElement(root, 'folder')
            folder.text = os.path.basename(os.path.dirname(image_path))  # 获取图片所在的文件夹名称
            filename = ET.SubElement(root, 'filename')
            filename.text = image_file_name
            path = ET.SubElement(root, 'path')
            path.text = os.path.abspath(image_path)  # 获取图片的绝对路径
            
            # 添加source元素
            source = ET.SubElement(root, 'source')
            database = ET.SubElement(source, 'database')
            database.text = 'Unknown'
            
            # 添加size元素
            size = ET.SubElement(root, 'size')
            width = ET.SubElement(size, 'width')
            width.text = str(image_width)
            height = ET.SubElement(size, 'height')
            height.text = str(image_height)
            depth = ET.SubElement(size, 'depth')
            depth.text = '1'
            
            # 添加segmented元素
            segmented = ET.SubElement(root, 'segmented')
            segmented.text = '0'
            
            # 遍历TXT文件中的每一行
            for line in txt_file:
                values = line.strip().split()
                category_id = values[0]
                x_center = float(values[1]) * image_width
                y_center = float(values[2]) * image_height
                bbox_width = float(values[3]) * image_width
                bbox_height = float(values[4]) * image_height
                
                # 创建object元素
                obj = ET.SubElement(root, 'object')
                name = ET.SubElement(obj, 'name')
                name.text = class_mapping[category_id]
                pose = ET.SubElement(obj, 'pose')
                pose.text = 'Unspecified'
                truncated = ET.SubElement(obj, 'truncated')
                truncated.text = '0'
                difficult = ET.SubElement(obj, 'difficult')
                difficult.text = '0'
                
                # 创建bounding_box元素
                bndbox = ET.SubElement(obj, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(int(x_center - bbox_width / 2))
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(int(y_center - bbox_height / 2))
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(int(x_center + bbox_width / 2))
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(int(y_center + bbox_height / 2))
            
            # 保存XML文件
            tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)
 
# 打印完成信息
print('Conversion from TXT to XML completed.')