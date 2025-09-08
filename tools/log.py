from other.load_log import load_log

path = r'D:\MasterProgram\Graduation_thesis\paper2\MyNET\log\kog\ex14_20240819-180331'
log_data = load_log(path)
log_data = log_data.sort_values(by="F1", ascending=False)
print('end')
