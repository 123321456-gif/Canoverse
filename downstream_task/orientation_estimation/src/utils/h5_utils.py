
import h5py

############针对2D分割的结果设计的读取和写入的h5
def save_masks_hdf5(masks, filename):
    """
    保存masks数据到HDF5文件
    """
    with h5py.File(filename, 'w') as file:
        for i, group in enumerate(masks):
            group_dataset = file.create_group(f"group_{i}")
            for j, (tensor, integer, tuple_val) in enumerate(group):
                # 将Tensor转换为numpy数组保存
                group_dataset.create_dataset(f"tensor_{j}", data=tensor)
                group_dataset[f"tensor_{j}"].attrs['integer'] = integer
                group_dataset[f"tensor_{j}"].attrs['tuple'] = tuple_val

def load_masks_hdf5(filename):
    """
    从HDF5文件加载masks数据
    """
    masks_loaded = []
    with h5py.File(filename, 'r') as file:
        for group_name in file:
            group_dataset = file[group_name]
            group = []
            for tensor_name in group_dataset:
                tensor = group_dataset[tensor_name][...]
                integer = group_dataset[tensor_name].attrs['integer']
                tuple_val = group_dataset[tensor_name].attrs['tuple']
                group.append((tensor, integer, tuple_val))
            masks_loaded.append(group)
    return masks_loaded