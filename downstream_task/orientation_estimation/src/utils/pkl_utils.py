import numpy as np
import pickle
import joblib

def save_data(data, file_path, use_joblib=False):
    """
    将数据保存到文件。

    参数:
    data: 要保存的数组列表。
    file_path: 保存文件的路径。
    use_joblib: 是否使用joblib进行保存。如果为False，则使用pickle。
    """
    if use_joblib:
        joblib.dump(data, file_path)
    else:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(file_path, use_joblib=False):
    """
    从文件中加载数据。

    参数:
    file_path: 加载文件的路径。
    use_joblib: 是否使用joblib进行加载。如果为False，则使用pickle。

    返回:
    加载的数组列表。
    """
    if use_joblib:
        return joblib.load(file_path)
    else:
        with open(file_path, 'rb') as f:
            return pickle.load(f)

if __name__=="__main__":
    # 示例数据
    data = [np.array([[0.18359375, -0.61572266, -0.22119141],
                    [0.05709839, -0.59619141, -0.71875],
                    [0.38110352, -0.61572266, -0.22253418]]),
            np.array([[0.05709839, -0.59619141, -0.71875],
                    [0.35473633, -0.60205078, -0.32348633],
                    [0.38110352, -0.61572266, -0.22253418]])]

    # 保存数据
    save_data(data, 'results/data.pkl')

    # 从文件中加载数据
    loaded_data = load_data('results/data.pkl')

    print(loaded_data)
