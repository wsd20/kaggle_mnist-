from PIL import Image
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
model_save_path = './项目/数据分析/kaggle_mnist/checkpoint/mnist.ckpt'


# 复现模型
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),


    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

# 加载参数
model.load_weights(model_save_path)




# 将任意大小的图片转换为输入图片
img = Image.open('./项目/数据分析/kaggle_mnist/pictures/' + '5.PNG')
img = img.resize((28, 28), Image.ANTIALIAS)
# 转换为灰度图
img_arr = np.array(img.convert('L'))
# 将灰度图进行二值化处理，将图片变成只有黑色和白色的高对比度图片，
# 同时进行降噪处理，(这里也可以使用opencv来操作)
# 灰度值大于200的设置为255，小于的设置为0
for i in range(28):
    for j in range(28):
        if img_arr[i][j] < 200:
            img_arr[i][j] = 255
        else:
            img_arr[i][j] = 0


# 由于我们训练时故意将图片的数值进行过处理，这里我们用同样的操作来使训练图像保持一致
img_arr = img_arr / 255.0


img_arr.shape
# 将数据转换为二维数组,展示图像
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
plt.subplot(1, 2, 1)
plt.imshow(img_arr,cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(img_arr,cmap='gray')
plt.title('示例图片')
plt.show()


img_arr = img_arr.reshape(1, 28, 28, 1)

# img_arr = img_arr[tf.newaxis, ...]
img_arr.shape
result = model.predict(img_arr)
pred = tf.argmax(result, axis=1)
tf.print(pred)
