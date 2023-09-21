import os
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import csv
# 加载预训练的ResNet模型
def load_resnet_model():
    model = models.resnet50(pretrained=True)
    model = model.eval()  # 设置为评估模式，即不进行训练
    return model

# 使用ResNet模型提取特征
def extract_features(image_path, model):
    # 图像预处理步骤
    preprocess = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 读取图像并进行预处理
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # 增加batch维度

    # 使用ResNet模型提取特征
    with torch.no_grad():
        features = model(image_tensor)

    return features.squeeze().numpy()  # 去除batch维度并转换为numpy数组

# 聚类特征
def perform_clustering(features, num_clusters=4):
    kmeans = KMeans(n_clusters=num_clusters)
    return kmeans.fit_predict(features)

# 可视化聚类结果
def visualize_clusters(cluster_labels, image_list):
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(image_list[i])

    for label, images in clusters.items():
        plt.figure(figsize=(10, 10))
        for i, image in enumerate(images):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        plt.show()

# 主程序
if __name__ == "__main__":
    # root_dir = '/mnt/d/data/shuangran'
    save_root_dir = "/home/omnisky/workspace/shuangran/data"
    save_abs_dirs = []
    sub_dirs = os.listdir(save_root_dir)
    for sub_dir in sub_dirs:
        # abs_dir.append(os.path.join(root_dir, sub_dir))
        save_abs_dirs.append(os.path.join(save_root_dir, sub_dir))
    
    # 创建文件夹来保存patch
    pngs = []
    for dir in save_abs_dirs:
        sub_sub_dir = os.listdir(dir)
        files_dirs = [os.path.join(dir, sub_sub_dir[i]) for i in range(len(sub_sub_dir))]
        for file_dir in files_dirs:
            #if file_dir is dir
            if os.path.isdir(file_dir): 
                file_paths = os.listdir(file_dir)
                pngs.add = [os.path.join(file_dir, file_paths[i]) for i in range(len(file_paths))]
                # pngs[file_dir.split('/')[-1]] = len(file_paths)
    #pngs sum
    
    # print(sum(pngs.values()))
    # #save dict as csv
    # with open('/home/omnisky/workspace/shuangran/each_slide_patch_num.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for key, value in pngs.items():
    #         writer.writerow([key, value])
    feature_list = []
    model = load_resnet_model()
    for png in tqdm(pngs, desc="Extracting Features", unit="image"):
        feature_list.append(extract_features(png, model))
    features = np.array(feature_list)
    features_path = "/home/omnisky/workspace/shuangran/features.pth"
    torch.save(features, features_path)
    # cluster_labels = perform_clustering(features)
    # visualize_clusters(cluster_labels, features)
