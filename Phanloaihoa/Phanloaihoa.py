import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from PIL import Image
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

# Tải dữ liệu hoa Iris (hoặc sử dụng hình ảnh đã được phân loại khác)
def load_iris_images(folder):
    X_train = []
    y_train = []
    # Giả sử tên file là "<label>_<index>.jpg"
    labels = {'setosa': 0, 'versicolor': 1, 'virginica': 2}  # Nhãn cho các loại hoa

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        img = img.resize((64, 64))  # Thay đổi kích thước hình ảnh
        img_array = np.array(img)
        X_train.append(img_array.flatten())  # Chuyển đổi thành vector
        label = filename.split("_")[0]  # Gán nhãn từ tên file
        y_train.append(labels[label])  # Chuyển đổi nhãn thành số

    return np.array(X_train), np.array(y_train)

# Hàm để chọn thư mục chứa ảnh và tải dữ liệu
def choose_folder_and_load_images():
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ Tkinter chính
    folder_path = askdirectory(title="Chọn thư mục chứa hình ảnh")
    
    if folder_path:
        X_train, y_train = load_iris_images(folder_path)
        print(f"Đã tải {len(X_train)} hình ảnh từ thư mục: {folder_path}")
        return X_train, y_train
    else:
        print("Không có thư mục nào được chọn!")
        return None, None

# Tạo mô hình KNN
k = 3
X_train, y_train = choose_folder_and_load_images()

if X_train is not None and y_train is not None:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Hàm để dự đoán loại hoa từ hình ảnh mới
    def predict_flower(image_path):
        img = Image.open(image_path)
        img = img.resize((64, 64))  # Thay đổi kích thước hình ảnh
        img_array = np.array(img)
        image_features = img_array.flatten()  # Chuyển đổi thành vector
        predicted_label = knn.predict([image_features])
        labels = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}  # Chuyển đổi nhãn số về tên hoa
        return labels[predicted_label[0]]

    # Sử dụng Tkinter để mở hộp thoại chọn ảnh
    def choose_image_and_predict():
        root = Tk()
        root.withdraw()  # Ẩn cửa sổ Tkinter chính
        image_path = askopenfilename(title="Chọn hình ảnh", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        
        if image_path:
            predicted_flower = predict_flower(image_path)
            print(f"Loại hoa dự đoán: {predicted_flower}")
        else:
            print("Không có tệp hình ảnh nào được chọn!")

    # Dự đoán loại hoa từ hình ảnh mới
    choose_image_and_predict()
else:
    print("Không thể tạo mô hình KNN vì không có dữ liệu!")
