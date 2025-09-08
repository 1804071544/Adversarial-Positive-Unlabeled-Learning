import numpy as np
from scipy.interpolate import make_interp_spline

def smooth_arr(x,cdf):
    spline = make_interp_spline(x, cdf, k=3)  # 三次样条拟合
    x_smooth = np.linspace(x.min(), x.max(), 100)  # 创建平滑的 x 轴
    cdf_smooth = spline(x_smooth)  # 计算平滑的 CDF
    return x_smooth, cdf_smooth