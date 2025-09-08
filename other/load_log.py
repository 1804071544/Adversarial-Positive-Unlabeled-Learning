# 假设日志文件名为 'log_file.log'
import os
import pandas as pd


def load_log(path):
    log_file = os.path.join(path, 'log.log')
    data = []
    line_index = 0
    # 打开并读取文件
    with open(log_file, 'r') as file:
        for line in file:
            line_index = line_index + 1
            if line.startswith("INFO:root:"):
                # 去掉首尾空白
                content = line.strip()
                #print(content)  # 打印整行内容

                # 提取损失值和其他指标
                parts = content.split(",")
                metrics = {part.split()[0]: float(part.split()[1]) for part in parts[1:]}
                metrics['epoch'] = int(line_index)  # 提取 epoch
                data.append(metrics)
    df = pd.DataFrame(data)
    return df
