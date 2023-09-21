import torch
import csv
features = torch.load('/home/omnisky/workspace/shuangran/features_vitb.pth')
from sklearn.cluster import KMeans
import pandas as pd
cluster_labels = KMeans(n_clusters=4).fit_predict(features)

# read each_slide_patch_num.csv
pngs_order = csv.reader(open('/home/omnisky/workspace/shuangran/each_slide_patch_num.csv', 'r'))
slide_pcr_labels = csv.reader(open('/home/omnisky/workspace/shuangran/silde_pcr_label.csv', 'r'))
#slide_pcr_labels to dict
slide_pcr_labels = dict(slide_pcr_labels)
#pngs_order 里面 每个key的value是一个list，list里面cluster_labels的0，1，2，3的数量
#将label与png对应起来
slide_cluster_labels = {}
before = 0
for png_order in pngs_order:
    nums = png_order[1]
    nums = int(nums)
    
    name = png_order[0]
    num_0 = cluster_labels[before:before+nums].tolist().count(0)
    num_1 = cluster_labels[before:before+nums].tolist().count(1)
    num_2 = cluster_labels[before:before+nums].tolist().count(2)
    num_3 = cluster_labels[before:before+nums].tolist().count(3)
    label = 1 if slide_pcr_labels[str(name) + '.ndpi'] == '1' else 0
    before += nums
    slide_cluster_labels[name] = [num_0, num_1, num_2, num_3, label]
#save as new csv
with open('/home/omnisky/workspace/shuangran/each_slide_patch_num_label_vitb.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for key, value in slide_cluster_labels.items():
        writer.writerow([key, value[0], value[1], value[2], value[3], value[4]])
print("")