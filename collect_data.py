import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Đường dẫn tới thư mục chứa ảnh và annotation
image_folder = 'train'
annotation_folder = 'train'
output_folder = 'dataset'

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Lặp qua tất cả các file annotation .xml trong thư mục
for annotation_file in tqdm(os.listdir(annotation_folder)):
    if annotation_file.endswith('.xml'):
        annotation_path = os.path.join(annotation_folder, annotation_file)
        
        # Đọc file annotation
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # Lấy tên file ảnh từ annotation
        filename = root.find('filename').text
        image_path = os.path.join(image_folder, filename)
        
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image can not be read: {image_path}")
            continue
        
        # Lặp qua tất cả các đối tượng trong annotation
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            # Kiểm tra nếu class name là 0 hoặc 1
            if class_name in ['0', '1']:
                # Lấy tọa độ bounding box
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                # Cắt ảnh theo bounding box
                cropped_image = image[ymin:ymax, xmin:xmax]
                
                # Lưu ảnh đã cắt vào thư mục output
                output_image_path = os.path.join(output_folder, f"{filename}_class_{class_name}_{xmin}_{ymin}.jpg")
                cv2.imwrite(output_image_path, cropped_image)

print("Done.")
