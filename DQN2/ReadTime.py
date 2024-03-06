# 假设文本文件名为log.txt

# 打开并读取文件
with open('main_DQN_log.txt', 'r',encoding='utf8') as file:
    # 遍历文件中的每一行
    for line in file:
        # 如果找到'Time elapsed:'字样，则打印该行内容
        if 'Time elapsed:' in line:
            print(line.strip())  # strip()用于移除行尾的换行符