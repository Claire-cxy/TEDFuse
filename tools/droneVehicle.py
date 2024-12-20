import os
from PIL import Image


# 定义处理图像的函数
def crop_image(image_path, output_folder):
    image = Image.open(image_path)
    width, height = image.size

    left = 100
    top = 100
    right = width - 100
    bottom = height - 100

    cropped_image = image.crop((left, top, right, bottom))

    # 获取图像文件名
    image_name = os.path.basename(image_path)
    # 保存裁剪后的图像
    cropped_image.save(os.path.join(output_folder, image_name))


# 定义要处理的文件夹路径
input_folder = '/home/cuixinyu/Fusion_new/test_img/visDrone/vi'  # 替换为你的输入文件夹路径
output_folder = '/home/cuixinyu/Fusion_new/test_img/droneVehicle/vi'  # 替换为你的输出文件夹路径

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(input_folder, filename)
        crop_image(image_path, output_folder)

print("所有图像已处理并保存至输出文件夹。")
