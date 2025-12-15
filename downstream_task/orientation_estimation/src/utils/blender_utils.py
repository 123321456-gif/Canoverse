import bpy
import bmesh
import random
import numpy as np

def assign_random_color(vertices):
    # 创建一个新的 mesh 对象
    mesh = bpy.data.meshes.new(name="RandomPolygon")
    obj = bpy.data.objects.new(name="RandomPolygonObj", object_data=mesh)

    # 将对象链接到当前场景
    bpy.context.collection.objects.link(obj)

    # 设置多边形顶点和面
    mesh.from_pydata(vertices, [], [[i for i in range(len(vertices))]])
    mesh.update()

    # 生成随机颜色
    random_color = (random.random(), random.random(), random.random(), 1)

    # 创建材质
    mat = bpy.data.materials.new(name="RandomColorMat")
    mat.diffuse_color = random_color

    # 将材质赋予对象
    obj.data.materials.append(mat)

    # 确保对象拥有正确的材质槽
    if len(obj.material_slots) == 0:
        bpy.ops.object.material_slot_add()

    # 分配材质给面片
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.object.material_slot_assign()
    bpy.ops.object.mode_set(mode='OBJECT')