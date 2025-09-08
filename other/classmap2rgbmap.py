import numpy as np
import PIL.Image as Image


def classmap2rgbmap(classmap: np.ndarray, palette, cls):
    palette = np.asarray(palette)
    (h, w) = classmap.shape
    rgb = np.zeros((h, w, 3))

    if cls == 'mcc':
        for i in range(h):
            for j in range(w):
                rgb[i, j, :] = palette[classmap[i, j], :]
    else:
        for i in range(h):
            for j in range(w):
                rgb[i, j, :] = palette[classmap[i, j] * cls, :]

    r = Image.fromarray(rgb[:, :, 0]).convert('L')
    g = Image.fromarray(rgb[:, :, 1]).convert('L')
    b = Image.fromarray(rgb[:, :, 2]).convert('L')

    rgb = Image.merge("RGB", (r, g, b))

    return rgb


def pro2grayim(input_matrix,):
    """
    Normalize the input 2D matrix and convert it to gray scale values, then save the image.

    Parameters:
    input_matrix (np.ndarray): The input 2D matrix to be normalized and converted.
    filename (str): The filename to save the gray scale image.

    Returns:
    np.ndarray: The normalized gray scale values.
    """
    # 确保输入为 NumPy 数组
    input_matrix = np.array(input_matrix)

    # 归一化处理
    min_val = np.min(input_matrix)
    max_val = np.max(input_matrix)
    normalized_array = (input_matrix - min_val) / (max_val - min_val)

    # 转换为灰度值 (范围 [0, 255])
    gray_image = (normalized_array * 255).astype(np.uint8)

    # 保存为图像
    img = Image.fromarray(gray_image)


    return img
