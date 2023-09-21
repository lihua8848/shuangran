import csv
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
pngs_path = csv.reader(open('/home/omnisky/workspace/shuangran/patch_coordinates_png_path.csv', 'r'))
pngs_path = list(pngs_path)

pngs_label = csv.reader(open('/home/omnisky/workspace/shuangran/patch_coordinates_label_vitb.csv', 'r'))
pngs_label = list(pngs_label)

#给不同label的png上不同的颜色
class_color = {
    0: (0, 0, 255), #red
    1: (0, 255, 0), #green
    2: (255, 0, 0), #blue
    3: (255, 255, 0) #yellow
}
import cv2
for png, label in tqdm(list(zip(pngs_path, pngs_label)), desc="Colorfy png", unit="png"):
    label = int(label[5])
    img = cv2.imread(png[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = np.zeros_like(img, dtype=np.uint8)
    overlay[:] = class_color[label]
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    # #可视化
    # plt.imshow(img)
    # plt.show()
    
    colorfy_png_path = png[0].replace('data', 'colorfy_data_vit')
    if not os.path.exists(os.path.dirname(colorfy_png_path)):
        os.makedirs(os.path.dirname(colorfy_png_path))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(colorfy_png_path, img)
