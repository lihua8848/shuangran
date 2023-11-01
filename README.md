# shuangran
## 文件说明
- histolab_crop.py：从npda文件中获取标注区域，切成256size的patch
- extract_features.py：使用预训练的resnet50进行patch特征的提取，使用kmeans++进行聚类
- colorfy_patch.py：对分类好的patch，上色，总共分成四类
- paste_cluster_patch.py：将上好色的patch，映射至病理图像上
- analysis_result.py：对聚类的结果进行分析并绘图
- 其他的为辅助函数
## Analysis
采用描述性统计分析，相关性分析，逻辑回归分析
- ![image](https://github.com/lihua8848/shuangran/assets/54617440/5cab65ff-384b-40f7-af35-b834a65aa34a)
- ![image](https://github.com/lihua8848/shuangran/assets/54617440/1f37e641-584c-4d20-a677-e864536b65de)
- ![image](https://github.com/lihua8848/shuangran/assets/54617440/25915ecf-6e2b-4cde-8614-bda8e5e34e4a)
- ![image](https://github.com/lihua8848/shuangran/assets/54617440/16405895-4571-488f-92f8-b9696627c709)



