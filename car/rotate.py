import cv2
import os

# Đường dẫn thư mục chứa ảnh nguồn và thư mục lưu ảnh đã xoay
input_folder = 'dataset/car'  # Thư mục chứa ảnh nguồn
output_folder = 'dataset'  # Thư mục lưu ảnh đã xoay

# Đảm bảo thư mục lưu ảnh đã xoay tồn tại, nếu không thì tạo mới
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Hàm xoay ảnh 180 độ và lưu ảnh mới
def rotate_image_180(input_path, output_path):
    # Đọc ảnh
    image = cv2.imread(input_path)

    # Xoay ảnh 180 độ
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    rotated_image = cv2.flip(rotated_image, 0) 

    # Lưu ảnh đã xoay
    cv2.imwrite(output_path, rotated_image)

# Duyệt qua các ảnh trong thư mục đầu vào và thực hiện xoay
for filename in os.listdir(input_folder):
    if filename.endswith((".png", ".jpg", ".jpeg")):  # Lọc các định dạng ảnh
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, 'rotated_' + filename)
        
        # Xoay ảnh và lưu kết quả
        rotate_image_180(input_image_path, output_image_path)

print("Hoàn thành xoay và lưu ảnh.")