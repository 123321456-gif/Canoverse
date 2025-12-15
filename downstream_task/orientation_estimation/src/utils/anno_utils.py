import math
import matplotlib.pyplot as plt

from PIL import Image
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
import json


####################################
# 根据图像上标注的2D点，转到3D点作为标签

####################################




# 执行一个类别之前，对是否处理过进行判断
def check_exist(annotation_dir,cate, instance_ids, visFlag, rendered_root):
    contiuneFlag = False
    # 如果annotation下存在这个类而且keys == instance_ids，跳过
    cate_annotation_path = os.path.join(annotation_dir, cate+'.json')
    if os.path.exists(cate_annotation_path):
        with open(cate_annotation_path, 'r') as f:
            cate_annotations = json.load(f)

        # cate_annotations.keys()个数 和 instance_ids 个数相同
        if len(cate_annotations.keys()) == len(instance_ids) and visFlag == False and cate_annotations[list(cate_annotations.keys())[0]] != None:
            print(f'{cate} annotation exists.')
            print('-------------------------')
            contiuneFlag = True
        #如果len(instance_ids) == 0，记录下来
        if len(instance_ids) == 0:
            print(f'{cate} has no instances.')
            print('-------------------------')
            log_path = os.path.join(os.path.dirname(rendered_root), 'log/annoation_NoneCate.txt')
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a') as f:
                f.write(f'error category:{cate} \n')
            contiuneFlag = True
    return contiuneFlag


# 保存结果
def save_results(click_infos, cate, annotation_dir, cate_annotations, start_time):
    # 判断annotation_dir是否存在，不存在则创建
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)
        
    # 保存标注结果
    cate_annotation_path = os.path.join(annotation_dir, cate+'.json')
    with open(cate_annotation_path, 'w') as f:
        json.dump(cate_annotations,f,indent=4)
    # click_info_path = os.path.join(annotation_dir, cate+'_click.json')
    # with open(click_info_path, 'w') as f:
    #     json.dump(click_infos,f,indent=4)
    end_time = time.time()
    print(f'{cate} annotation saved. annotation time: {end_time-start_time}s')
    print('-------------------------')

############点击相关函数

def index2key(index):
    if index==0:
        return 'temp'
    elif index==1:
        return 'orgObj'
    elif index==2:
        return 'alignObj'
    elif index==3:
        return 'alignGeoObj'
    elif index==4:
        return 'reAlignObj'
    else:
        return None

##################3

# 加载图片
def load_images(instance_ids:list, temp_dir:str, org_dir:str, align_dir:str, align_geo_dir:str, reAlign_dir:str)->dict:
    def load_images(instance_id):
        org_pic = os.path.join(org_dir, instance_id)
        align_pic = os.path.join(align_dir, instance_id)
        align_geo_pic = os.path.join(align_geo_dir, instance_id)
        reAlign_pic = os.path.join(reAlign_dir, instance_id)
        return {
            'org_pic': load_image(org_pic),
            'align_pic': load_image(align_pic),
            'align_geo_pic': load_image(align_geo_pic),
            'reAlign_pic': load_image(reAlign_pic)
        }
    temp_pic_path=os.path.join(temp_dir, 'ins_0000.png')
    # 预加载temp_pic
    temp_pic = load_image(temp_pic_path)

    res_images = {}
    # 使用线程池加载图片，线程池适合IO密集型任务，进程池适合CPU密集型任务（不受GIL限制）
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_images, instance_id): instance_id for instance_id in instance_ids}
        for future in tqdm(futures, desc="Loading and merging images"):
            instance_id = futures[future]
            try:
                images = future.result()
                res_images[instance_id] = [temp_pic, images['org_pic'], images['align_pic'], images['align_geo_pic'], images['reAlign_pic']]
            except Exception as exc:
                print(f'{instance_id} generated an exception: {exc}')
    
    return res_images

# 创建图像
def create_figure():
    plt.ion()
    fig, axes = plt.subplots(1, 5, figsize=(22, 9))
    for ax in axes.flatten():
        ax.axis('off')
    return fig, axes

# 读取图片
def load_image(path):
    img=Image.open(path)
    return img

# 根据用户分配类别
def get_user_categories(cates):
    visFlag = False
    # 请求用户输入用户名
    username = input("input your name: ")
    if username == 'jl':
        '''def count_non_empty_folders(path):
            count = 0
            filesNum = 0
            
            # 遍历指定路径下的所有文件夹
            for foldername in os.listdir(path):
                folder_path = os.path.join(path, foldername)
                
                # 确保是文件夹
                if os.path.isdir(folder_path):
                    # 获取文件夹内的所有文件
                    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                    
                    # 检查文件个数
                    if len(files) >= 2:  # 文件个数大于等于2
                        filesNum = filesNum + len(files)
                        count += 1
            
            return count, filesNum
        path = '/data4/jl/project/one-shot/results/objaverse_lvis/align'
        count, filesNum = count_non_empty_folders(path)
        print(f'number: {count, filesNum}')
        asdf
        # cates = cates[:int(len(cates)/2)]
        cates = ['shoulder_bag']  # , 'bicycle', 'mo'
        visFlag = True'''
        cates = cates[:int(len(cates)/2)]
    elif username == 'zpp':
        cates = cates[int(len(cates)/2):]
    elif username == 'all':
        pass

    return cates, visFlag

# 重排类别
def reCates(cates):
    # 个数从大到小， 导致gpu负载不均衡，重排
    sorted_cates = []
    left, right = 0, len(cates) - 1
    while left <= right:  # 交替排列类别
        sorted_cates.append(cates[left])  # 添加左边的元素
        if left != right:  # 防止重复添加中间元素
            sorted_cates.append(cates[right])  # 添加右边的元素
        left += 1
        right -= 1
    cates = sorted_cates
    return cates





color_map = {
    1: [255, 0, 0,255], # 红色
    2: [0, 255, 0,255], # 绿色
    3: [0, 0, 255,255], # 蓝色
    4: [255, 255, 0,255], # 黄色
    5: [255, 0, 255,255], # 紫色
    6: [0, 255, 255,255], # 青色
    7: [0, 0, 0 ,255], # 白色
}
# text_map = {
#     1: 'y+',
#     2: 'y-',
#     3: 'h+',
#     4: 'h-',
# }
text_map = {
    1: 'y180',
    2: 'y90',
    3: 'h+',
    4: 'h-',
}


def generate_circle_points(center_x, center_y, radius, image_width, image_height):

    """
    生成以点击点为中心的圆形区域内所有点的坐标，且不超出图像边界。

    :param center_x: 圆心的x坐标
    :param center_y: 圆心的y坐标
    :param radius: 圆的半径
    :param image_width: 图像的宽度
    :param image_height: 图像的高度
    :return: 圆内点的坐标列表
    """
    points = []
    # 计算圆的边界，确保不超出图像
    left = max(0, center_x - radius)
    right = min(image_width, center_x + radius + 1)
    top = max(0, center_y - radius)
    bottom = min(image_height, center_y + radius + 1)

    for i in range(top, bottom):
        for j in range(left, right):
            # 计算当前点到圆心的距离
            distance = math.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
            # 如果点在圆内或圆上，则添加到列表中
            if distance <= radius:
                points.append((j, i))  # 注意：图像坐标通常是(y, x)
    
    return points

class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        else:
            raise IndexError("pop from empty stack")

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        else:
            raise IndexError("peek from empty stack")

    def size(self):
        return len(self.items)
    def clear(self):
        self.items = []

    def __str__(self):
        return str(self.items)