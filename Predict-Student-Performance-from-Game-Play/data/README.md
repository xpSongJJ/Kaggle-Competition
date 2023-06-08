## 数据集准备  
先下载好源数据集：https://www.kaggle.com/competitions/predict-student-performance-from-game-play/data  
运行`generate`程序，生成训练时用到的数据集，`.parquet`文件是以列为独立单位进行存储，
只读取相应的列时，速度比较快。  
