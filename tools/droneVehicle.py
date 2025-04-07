import os
from PIL import Image



def crop_image(image_path, output_folder):
    image = Image.open(image_path)
    width, height = image.size

    left = 100
    top = 100
    right = width - 100
    bottom = height - 100

    cropped_image = image.crop((left, top, right, bottom))


    image_name = os.path.basename(image_path)

    cropped_image.save(os.path.join(output_folder, image_name))


input_folder = '/home/cuixinyu/TEDFuse/test_img/Drone/vi'  
output_folder = '/home/cuixinyu/TEDFuse/test_img/droneVehicle/vi'  


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(input_folder, filename)
        crop_image(image_path, output_folder)

print("所有图像已处理并保存至输出文件夹。")
