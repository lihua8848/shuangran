import os

#统计以ndpi结尾的文件，将文件名做一个excel，gbk编码
root_dir = "/mnt/d/data/shuangran"
for dir, sub_dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.ndpi'):
            with open(root_dir + '/ndpi_name.txt', 'a') as f:
                f.writelines(f'{file}\n')