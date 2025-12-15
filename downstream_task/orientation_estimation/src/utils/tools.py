"""一些用于判断的小工具
"""

from PIL import Image

# 1. 判断图片的分辨率是否是480x480
def is_480x480_image(save_path):
    try:
        # 打开图片文件
        with Image.open(save_path) as img:
            # 获取图片的宽度和高度
            width, height = img.size

            # 判断是否是480x480的图片
            if width == 480 and height == 480:
                return True
            else:
                return False
    except Exception as e:
        # 如果无法打开图片或出现错误，返回False
        print(f"Error: {e}")
        return False