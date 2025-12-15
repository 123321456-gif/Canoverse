import time
from functools import wraps

####### 计时器 修饰器
# 计时器
def timer(func):
    """
    装饰器，用于测量并打印函数执行所需的时间（以秒为单位）。
    
    参数:
    func -- 要计时的函数
    
    返回:
    wrapper -- 包装后的函数
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 开始计时
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 结束计时
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper


# 统计每个函数中主要步骤的
def timeit_section(section_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"[INFO] Execution time of {section_name}: {end_time - start_time:.2f} seconds")
            return result
        return wrapper
    return decorator