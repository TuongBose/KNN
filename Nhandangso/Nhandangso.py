import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def distance(test_sample, train_sample):
    total_distance = 0
    for index in range(400):
        if ((test_sample[0][index] <= (train_sample[0][index] + 10)) and (test_sample[0][index] >= (train_sample[0][index] - 10))):
            pass
        else:
            total_distance += 1
    return total_distance


def sortNeighbors(neighbor):
    return neighbor['dis']


def KNN(dataset, test, K):
    list_test = test.reshape(1, 400)
    k_list = []
    for i in range(50):
        for j in range(50):
            x = np.array(dataset[i][j])
            list_data = x.reshape(1, 400)
            k_list.append({'label': i // 5, 'dis': distance(list_test, list_data)})

    k_list.sort(key=sortNeighbors)
    k_list = k_list[:K]
    m_list = [k_list[i]['label'] for i in range(K)]

    print("Nhãn dự đoán:", max(set(m_list), key=m_list.count))


def select_image_file(title="Select an image file"):
    # Mở cửa sổ chọn file
    Tk().withdraw()  # Ẩn cửa sổ chính của tkinter
    file_path = askopenfilename(title=title, filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    return file_path


# Mở cửa sổ chọn ảnh train (digits.png)
print("Select the Train image file...")
image_train_path = select_image_file("Select Train image file")
imagetrain = cv2.imread(image_train_path, 0)

# Mở cửa sổ chọn ảnh test (test.jpg)
print("Select the test image file...")
image_test_path = select_image_file("Select test image file")
imagetest = cv2.imread(image_test_path, 0)

# Chia ảnh train thành các ô nhỏ 50x50
cells = [np.hsplit(row, 50) for row in np.vsplit(imagetrain, 50)]

# Thực hiện phân loại KNN với K = 13
KNN(cells, imagetest, 13)
