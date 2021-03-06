import os
import cv2
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import seaborn as sb

# 导入数据
data = pd.read_csv("./项目/数据分析/kaggle_mnist/train.csv").astype('float32')
# 显示头五条数据
data.head()
print(data.shape)   # 显示数据形状是(42000, 785)，图片的大小为28*28的灰度图，第一列为标签


# 查看我们有多少标签
data.label.nunique()    # 有10个



#  设置数据集
y = data.label
x = data.drop('label',axis=1)

print(y.shape)
print(x.shape)
x  = x/255.0    # 对数值进行归一化处理使得原本0~255的灰度值变成0到1的数值，使得数据变小,降低学习难度

#划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# 展示图像
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
plt.subplot(1, 2, 1)
plt.imshow(x_train[:1].values.reshape((28,28)),cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(x_train[1:2].values.reshape((28,28)),cmap='gray')
plt.show()

# 标准化
scaler = MinMaxScaler()
scaler.fit(x_train)
X_train = scaler.transform(x_train)
X_test = scaler.transform(x_test)

# 将图片样本转换形状
X_train = np.reshape(X_train, (X_train.shape[0], 28,28,1)).astype('float32')
X_test = np.reshape(X_test, (X_test.shape[0], 28,28,1)).astype('float32')
print("Train data shape: ", X_train.shape)
print("Test data shape: ", X_test.shape)

# 将标签转换为独热编码
y_train = tf.keras.utils.to_categorical(y_train,num_classes=10,dtype=int)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=10,dtype=int)
print(y_train.shape,y_test.shape)



#定义映射字典         # 参考别人的，这里懒得改了-.-
number_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',
                7:'7',8:'8',9:'9'}

#展示
fig, axis = plt.subplots(3, 3, figsize=(20, 20))
for i, ax in enumerate(axis.flat):
    ax.imshow(x_train[:][i:i+1].values.reshape(28,28),cmap='gray')
    ax.axis('off')
    ax.set(title = f"mnist : {number_dict[y_train[i].argmax()]}")
plt.show()



# 计算一共有各个标签有多少个
sb.set_style('whitegrid')
df=data.copy()
df['label'] = df['label'].map(number_dict)

labels_count = df.groupby('label').size()
labels_count.plot.bar(figsize=(15,10))
plt.ylabel("Count")
plt.xlabel("mnist")
plt.show()
