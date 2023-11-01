# shuangran
## 文件说明
histolab_crop.py：从npda文件中获取标注区域，切成256size的patch
extract_features.py：使用预训练的resnet50进行patch特征的提取，使用kmeans++进行聚类
colorfy_patch.py：对分类好的patch，上色，总共分成四类
paste_cluster_patch.py：将上好色的patch，映射至病理图像上
其他的为辅助函数
逻辑回归的代码稍后更新
