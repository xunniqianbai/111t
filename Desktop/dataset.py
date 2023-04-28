import os
import random
img_path_dir = r"D:\level-set\dataset\image"
label_path_dir = r"D:\level-set\dataset\label_seg"
ratio = 0.9

img_name_list = os.listdir(img_path_dir)
label_name_list = os.listdir(label_path_dir)

output_train_file = "./train.csv"
output_test_file = "./test.csv"

line_list = []
for img_path, label_path in zip(img_name_list, label_name_list):
    line_ = img_path_dir + "/" + img_path + "," + label_path_dir + "/" + label_path
    line_list.append(line_)

random.shuffle(line_list)
#1.shuffle 就是为了避免数据投入的顺序对网络训练造成影响。

# 2.增加随机性，提高网络的泛化性能，避免因为有规律的数据出现，导致权重更新时的梯度过于极端，避免最终模型过拟合或欠拟合。
# shuffle 就是为了避免数据投入的顺序对网络训练造成影响。
# 增加随机性，提高网络的泛化性能，避免因为有规律的数据出现，导致权重更新时的梯度过于极端，避免最终模型过拟合或欠拟合。

len_ = len(line_list)

with open(output_train_file,"w",encoding="utf-8") as f:
    for line_ in line_list[:int(ratio*len_)]:
        f.write(line_+"\n")

with open(output_test_file,"w",encoding="utf-8") as f:
    for line_ in line_list[int(ratio*len_):]:
        f.write(line_+"\n")

