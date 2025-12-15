# coding=utf-8
# mask, depth， 以及自定义可视化
# 点云可视化
# 点云投影到2D
# 模型可视化
# 模型投影到2D
import time
import numpy as np
import open3d as o3d
import os
import cv2
import time
import numpy as np
import open3d as o3d
import os
import cv2

def pjoin(*a):
    """
    适用于ubuntu和windows的路径拼接函数
    :param a: *a: The path waitting for join 使用示例: path=pjoin('/data1/','001.png')
    :return: 拼接后的路径
    """
    path = a[0]
    for i in range(len(a)):
        if i == 0:
            continue
        else:
            path = os.path.join(path, a[i]).replace('\\', '/')
    return path



# coding=utf-8
# mask, depth， 以及自定义可视化
# 点云可视化
# 点云投影到2D
# 模型可视化
# 模型投影到2D
class PointCloudRender:
    def __init__(self, result_dir=None, window_shape=(512, 512), window_pos=(50, 25)):
        self.result_dir = result_dir
        self.window_width = window_shape[0]
        self.window_height = window_shape[1]
        self.left = window_pos[0]
        self.top = window_pos[1]
        self.rotate_value = 10.0
        self.rotate_count = [0]
        self.coordinate = False
        self.coordinateMesh = None

    def __rotate_view(self, vis):
        """
        旋转回调函数
        :param vis:  vis = o3d.visualization.Visualizer()
        :return:
        """
        ctr = vis.get_view_control()
        # ctr.set_zoom(2)
        ctr.rotate(self.rotate_value, 0)
        if self.result_dir and 1 < self.rotate_count[0] < 525:
            save_path = os.path.join(self.result_dir, f'{self.rotate_count[0]}.png')
            vis.capture_screen_image(save_path, False)
        self.rotate_count[0] += 1
        return False

    def __capture_image(self, vis):
        img_save_path = os.path.join(self.result_dir, "{0}.png".format(time.asctime().replace(' ', '-').replace(':', '-')))
        image_buffer = vis.capture_screen_float_buffer()
        img_tmp = np.asarray(image_buffer)
        img = img_tmp.copy()
        img[:, :, 0] = img_tmp[:, :, 2]
        img[:, :, 2] = img_tmp[:, :, 0]

        cv2.imshow("captured image", img)
        cv2.waitKey(1)
        if self.result_dir:

            cv2.imwrite(img_save_path, img * 255.0)
            print("[OK] img save to {0}".format(img_save_path))
        return False

    def open_coordinate(self, open):
        """
        是否开启坐标系
        :param open: True 在可视化中显示坐标系; False 在可视化中关闭坐标系
        :return:
        """
        self.coordinate = open

    def coordinate_setting(self, scale=1.0, center=(0, 0, 0)):
        """
        调整坐标系的scale, 坐标原点位置
        x, y, z 坐标轴将分别渲染为红色, 绿色, 蓝色
        :param scale: 坐标系尺寸
        :param center: tuple(x, y, z) 坐标系原点位置
        :return:
        """
        mesh_ = o3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh_.scale(scale, center=(0, 0, 0))
        self.coordinateMesh = mesh_

    """
    Args:
        name: windows name 窗口名
        pts_list: list of pointcloud (numpy.array, nx3) 存放点云的list
        color_list: list of color (every color like np.array([255,0,0])) 存放颜色的list，list长度应该与pts_list相同
        # 修复, color_list 现在可以接收单独定义每个点的颜色(1,3) -> (1, 3) 或 (n, 3)
        result_dir: 存储图片的路径, 如果为None则不会保存 例如: "/data/cat"
    """
    def render_multi_pts(self, win_name, pts_list, color_list, save_dir=None, save_img=False, show_coord=True):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=win_name, width=512, height=512, left=300, top=300)
        opt = vis.get_render_option()
        opt.show_coordinate_frame = show_coord
        assert len(pts_list) == len(color_list)
        pcds = []
        for index in range(len(pts_list)):
            print(1)
            pcd = o3d.geometry.PointCloud()
            pts = pts_list[index]
            color = color_list[index]
            if color.shape != pts.shape:
                print("color shape != pts.shape")
                colors = np.tile(color, (pts.shape[0], 1))
            else:
                colors = color
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            pcds.append(pcd)
        if self.coordinate:
            vis.add_geometry(self.coordinateMesh)
        for pcd in pcds:
            vis.add_geometry(pcd)
        ctr = vis.get_view_control()

        key_to_callback = {}
        key_to_callback[ord("C")] = self.__capture_image
        o3d.visualization.draw_geometries_with_key_callbacks([pcd for pcd in pcds], key_to_callback, window_name=win_name,
                                                                width=self.window_width, height=self.window_height,
                                                                left=self.left, top=self.top)
        if save_img:
            img_save_path = os.path.join(save_dir, "{0}.png".format(time.asctime().replace(' ', '-').replace(':', '-')))
            vis.capture_screen_image(img_save_path, False)
            print("[OK] img save to {0}".format(img_save_path))


    def visualize_shape(self, name, pts, result_dir=None):
        """
        最简单的可视化函数
        :param name: 窗口名
        :param pts: (numpy.array, nx3) 点云
        :param result_dir: 存储图片的路径, 如果为None则不会保存 例如: "/data/cat"
        :return:
        """
        """ The most simple function, for visualization pointcloud and save image.
        Args:
            name: window name 
            pts: list of pointcloud 
            result_dir: if not None, save image to this path 图片保存路径

        """
        self.result_dir = result_dir
        vis = o3d.visualization.Visualizer()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        vis.add_geometry(pcd)
        ctr = vis.get_view_control()

        key_to_callback = {}
        key_to_callback[ord("C")] = self.__capture_image
        o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback, window_name=name,
                                                                width=self.window_width, height=self.window_height,
                                                                left=self.left, top=self.top)


    """
    Args:
        name: windows name 窗口名
        pts_list: list of pointcloud (numpy.array, nx3) 存放点云的list
        color_list: list of color (every color like np.array([255,0,0])) 存放颜色的list，list长度应该与pts_list相同
        # 修复, color_list 现在可以接收单独定义每个点的颜色(1,3) -> (1, 3) 或 (n, 3)
        result_dir: 存储图片的路径, 如果为None则不会保存 例如: "/data/cat"
    """
    def render_multi_pts(self, win_name, pts_list, color_list, save_path=None, show_coord=True):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=win_name, width=512, height=512, left=300, top=300)
        opt = vis.get_render_option()
        opt.show_coordinate_frame = show_coord
        assert len(pts_list) == len(color_list)
        pcds = []
        for index in range(len(pts_list)):
            pcd = o3d.geometry.PointCloud()
            pts = pts_list[index]
            color = color_list[index]
            if color.shape != pts.shape:
                print("color shape != pts.shape")
                colors = np.tile(color, (pts.shape[0], 1))
            else:
                print("color shape == pts.shape")
                colors = color
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            # pcds.append(pcd)
            vis.add_geometry(pcd)
        ctr = vis.get_view_control()
        ctr.rotate(-300.0, 150.0)
        # vis.run()
        if save_path is not None:
            vis.capture_screen_image(save_path, False)
        vis.destroy_window()


    def render_multi_pts_rotation(self, name, pts_list, color_list,
                                    rotate_value=8.0, angle_offset=(0, 0, 0), result_dir=None):
        """
        旋转可视化点云, 自动截图并保存到指定目录
        :param name:
        :param pts_list: list of pointcloud (numpy.array, nx3) 点云
        :param color_list: 点云颜色 例如[np.array([255,0,0]), np.array([255,255,0])], 此参数如果不为空, 长度必须和pts_list一致
        :param rotate_value: 该值的大小控制每次旋转的幅度 默认8.0
        :param angle_offset: 调整点云绕xyz轴的初始旋转。(值的范围为-2~2,对应-2*PI~2*PI) (numpy.float)
        :param result_dir: 存储图片的路径, 如果不提供则不会保存 例如: "/data/cat"
        :return:
        """
        self.rotate_value = rotate_value
        self.result_dir = result_dir
        vis = o3d.visualization.Visualizer()
        assert len(pts_list) == len(color_list)

        pcds = []
        for index in range(len(pts_list)):
            pcd = o3d.geometry.PointCloud()
            pts = pts_list[index]
            color = color_list[index]
            if len(color.shape) == 1 or color.shape[0] != pts.shape[0]:
                if np.any(color > 1.0):
                    color = color.astype(float) / 255
                color = np.tile(color, (pts.shape[0], 1))  # 将color扩大为 (pts.shape[0], 1)
            else:
                print("Error in viz_pts_with_rotation: color shape {0} wrong".format(color.shape))
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(color)
            # 调整pts角度
            x_r, y_r, z_r = angle_offset
            R = pcd.get_rotation_matrix_from_xyz((x_r * np.pi, y_r, z_r * np.pi))
            pcd = pcd.rotate(R, center=(0, 0, 0))

            pcds.append(pcd)
            vis.add_geometry(pcd)

        ctr = vis.get_view_control()
        o3d.visualization.draw_geometries_with_animation_callback(pcds, self.__rotate_view, window_name=name,
                                                                width=self.window_width, height=self.window_height,
                                                                left=self.left, top=self.top)
        self.rotate_count[0] = 0


if __name__ == '__main__':
    # example of usage
    pts1 = np.random.randn(1024, 3)
    pts2 = np.random.randn(1024, 3)
    red = np.array([255, 0, 0])
    green = np.array([0, 255, 0])
    blue = np.array([0, 0, 255])

    pcr = PointCloudRender()
    pcr.render_multi_pts("multi", [pts1, pts2], [red, green], "M:/test/")


    pcr.visualize_shape("simple", pts1, "M:/test/")
    pcr.render_multi_pts_rotation("multi-rotate", [pts1, pts2], [red, green])