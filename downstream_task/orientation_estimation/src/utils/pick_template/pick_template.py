"""用于给每个类别挑选模板

Returns:
    输入： dataroot ； categories ； save_root
    1. 读取，每个物体，可视化出来
    2. 若看中作为模板，给个指令，保存到对应的位置; 并跳到下一个类
"""

import tkinter as tk
import shutil
import os
import json
from tqdm import tqdm


from src.structure.pytorch_mesh import PytorchMesh
from src.structure.point_cloud import PointCloud

'''def prompt_save_and_handle_keys(mesh_dir, save_dir):
    def save_to_file(mesh_dir, save_dir):
        # 定义目标路径
        dst_path = os.path.join(save_dir, os.path.basename(mesh_dir))
        # 拷贝文件夹
        shutil.copytree(mesh_dir, dst_path, dirs_exist_ok=True)
        print(f'Data saved to {save_dir}')

    def on_enter(event):
        save_to_file(mesh_dir, save_dir)
        root.quit()

    def on_esc(event):
        print('Skipped saving.')
        root.quit()

    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    top = tk.Toplevel(root)
    top.title("提示")

    label = tk.Label(top, text="save or not? (Enter/Escape)")
    label.pack(pady=20)

    top.bind('<Return>', on_enter)
    top.bind('<Escape>', on_esc)
    top.focus_force()

    root.mainloop()

    top.destroy()
    root.destroy()
'''

'''class SavePrompt:
    def __init__(self, mesh_dir, save_dir):
        self.mesh_dir = mesh_dir
        self.save_dir = save_dir
        self.result = None

    def save_to_file(self):
        dst_path = os.path.join(self.save_dir, os.path.basename(self.mesh_dir))
        shutil.copytree(self.mesh_dir, dst_path, dirs_exist_ok=True)
        print(f'Data saved to {self.save_dir}')
        self.result = 'enter'

    def on_enter(self, event):
        self.save_to_file()
        self.root.quit()

    def on_esc(self, event):
        print('Skipped saving.')
        self.result = 'escape'
        self.root.quit()

    def prompt_save_and_handle_keys(self):
        self.root = tk.Tk()
        self.root.withdraw()  # 隐藏主窗口

        top = tk.Toplevel(self.root)
        top.title("提示")

        label = tk.Label(top, text="save or not? (Enter/Escape)")
        label.pack(pady=20)

        top.bind('<Return>', self.on_enter)
        top.bind('<Escape>', self.on_esc)
        top.focus_force()

        self.root.mainloop()

        top.destroy()
        self.root.destroy()

        return self.result

'''

def save_to_file(mesh_dir, save_dir):
    dst_path = os.path.join(save_dir, os.path.basename(mesh_dir))
    shutil.copytree(mesh_dir, dst_path, dirs_exist_ok=True)
    print(f'Data saved to {save_dir}')

def prompt_save_and_handle_keys(mesh_dir, save_dir):
    while True:
        response = input("save or not? (y/n): ").strip().lower()
        if response == 'y':
            save_to_file(mesh_dir, save_dir)
            return 'y'
        elif response == 'n':
            print('Skipped saving.')
            return 'n'
        else:
            print("Invalid input. Please type 'y' or 'n'.")

if __name__ == "__main__":
    '''# objaverse 6k 配置
    data_json_root = '/data4/jl/project/one-shot/data/interim/objaverse_dataset/newObj/obj_info'
    save_root = 'data/interim/objaverse_dataset/temp'
    from src.dataset.objaverse_6k_cateNames import get_categories
    cates_all = get_categories()
    dataType = 'objaverse_6k'
    '''

    # objaverse 4w 配置
    data_json_root = '/data4/jl/project/one-shot/data/interim/objaverse_lvis_dataset/newObj/obj_info'
    save_root = '/data4/jl/project/one-shot/data/interim/objaverse_lvis_dataset/temp'
    from src.dataset.objaverse_4w_cateNames import get_categories
    cates_all = get_categories()
    dataType = 'objaverse_4w'

    # 从cates中选出save_root没有的类别
    cates_inDir = os.listdir(save_root)
    # 选出cates_inDir 下不为空的类别
    undealCates = []
    for cate in cates_all:
        if dataType == 'objaverse_6k':
            path = os.path.join(save_root, cate, 'glbs')
        elif dataType == 'objaverse_4w':
            path = os.path.join(save_root, cate)
        if os.path.exists(path) and os.path.isdir(path):
            files = os.listdir(path)
            if not files:
                undealCates.append(cate)
        else:
            undealCates.append(cate)
    cates = undealCates
    # print('precess cates num', len(cates))
    # print('cates:', cates)
    # asdf
    # print('cates:', cates)
    # cates = cates[::-1]
    print('precess cates num', len(cates))
    # asdf


    # 获得处理类别的路径
    for cate in tqdm(cates):
        # if cate == 'airplane':
        #     continue
        cateData_root = os.path.join(data_json_root, cate)
        cateJson_path = cateData_root + '.json'
        print('cateData_root:', cateJson_path)
        
        # 读取json
        with open(cateJson_path, 'r') as f:
            cate_data = json.load(f)

        # 读取对应的点云并可视化
        for ins_name, values in cate_data.items():
            mesh_path = values['mesh']
            print('obj_path:', mesh_path)
            # 读取obj，并采样5000个点云，可视化
            meshIns = PytorchMesh.read_from_obj(mesh_path)
            if not meshIns.check_mesh():
                print('Error: mesh is empty.')
                continue
            pts = meshIns.to_pts(5000)
            PointCloud(pts).vis()

            # 保存
            if dataType == 'objaverse_6k':
                save_dir = os.path.join(save_root, cate, 'glbs')
            elif dataType == 'objaverse_4w':
                save_dir = os.path.join(save_root, cate)
            mesh_dir = os.path.dirname(mesh_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print('mesh_dir:', mesh_dir)
            print('save_dir:', save_dir)
            # asdf
            # prompt = SavePrompt(mesh_dir, save_dir)
            # result = prompt.prompt_save_and_handle_keys()
            result = prompt_save_and_handle_keys(mesh_dir, save_dir)
            print(f"Result: {result}")
            # prompt_save_and_handle_keys(mesh_dir, save_dir)
            if result == 'y':
                break





    '''data_root = '/data4/LJ/workstation/PartSLIP2/results/0506-omni3d/'
    results_root = '/data4/LJ/workstation/PartSLIP2/results/0506-omni3d_eval/'
    obj_root = '/data4/jl/datasets/omni3d/obj/'
    save_root = '/data4/LJ/workstation/PartSLIP2/results/0506-omni3d_demo/'
    categorys = ["teapot", "biscuit", "box", "burrito", "dinosaur", "eraser", "flash_light", "tooth_paste"]

    for cate in categorys:
        cate_path = data_root + '/templates/' + cate + '/results/'
        temps = os.listdir(cate_path)
        cateTemp_path = cate_path + temps[0] + '/rot0/'
        temp_all_pts = read_ply(cateTemp_path + 'normalized_pc.ply', num_samples=1500)

        aligned_rots_path = results_root + cate + '/r_can.npy'
        aligned_rots = np.load(aligned_rots_path)

        test_root = data_root + cate + '/results/'
        objs_names = os.listdir(test_root)
        choose_rots = []
        random_rots_list = []
        mesh_names_list = []

        for i, obj_name in enumerate(objs_names):
            
            obj_path = results_root + cate + '/' + obj_name + '.ply'
            obj_pts = read_ply(obj_path, num_samples=1500)

            display_two_point_clouds(temp_all_pts, obj_pts)

            
            mesh_path = obj_root + cate + '/' + obj_name.split('_random')[0] + '/'
            if not os.path.exists(save_root + cate + '/'):
                os.makedirs(save_root + cate + '/')
            # 从外面获得一个input
            input_flag = input('input: ')
            print(input_flag)
            if input_flag == '1':
                # # debug 读取变换前的，aligned pose 和读取的align pts 是否一致
                # mesh_read_path = mesh_path + 'Scan/Scan.obj'
                # objPts = read_obj_andBackup(mesh_read_path, num_points=3000)
                # print('objPts:', objPts.shape)
                # print('aligned_rots:', aligned_rots[i][0].shape)

                # ali_pts = objPts@aligned_rots[i][0].T
                # print('obj_pts:', obj_pts.shape)
                # print('ali_pts:', ali_pts.shape)
                # display_two_point_clouds(obj_pts, ali_pts)
                 

                mesh_names_list.append(obj_name)
                choose_rots.append(aligned_rots[i])
                random_rots_list.append(Rotation.random().as_matrix())
                save_path=save_root + cate + '/'
                shutil.copytree(mesh_path, save_path + mesh_path.split('/')[-2] + '/')
                print(f'Data saved to {save_path}')
        choose_rots = np.array(choose_rots)
        random_rots_list = np.array(random_rots_list)

        if not os.path.exists(save_root + 'rots/'):
            os.makedirs(save_root + 'rots/')
        np.save(save_root + 'rots/' + cate + '_random.npy', random_rots_list)
        np.save(save_root + 'rots/' + cate + '_aligned.npy', choose_rots)

        file_path = save_root + 'rots/' + cate + '_mesh_names.txt'
        with open(file_path, "w") as file:
            for string in mesh_names_list:
                file.write(string + "\n")'''
