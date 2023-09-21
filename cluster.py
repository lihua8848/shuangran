import torch
import csv
features = torch.load('/home/omnisky/workspace/shuangran/features_vitb.pth')
from sklearn.cluster import KMeans
cluster_labels = KMeans(n_clusters=4).fit_predict(features)

pngs_order = csv.reader(open('/home/omnisky/workspace/shuangran/patch_coordinates.csv', 'r'))
pngs_order = list(pngs_order)
#将label与png对应起来
pngs_order = [pngs_order[i] + [cluster_labels[i]] for i in range(len(pngs_order))]
#save as new csv
with open('/home/omnisky/workspace/shuangran/patch_coordinates_label_vitb.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for png_order in pngs_order:
        writer.writerow(png_order)
print(len(pngs_order))
