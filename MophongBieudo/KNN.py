import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from matplotlib.widgets import TextBox, Button

# Dữ liệu tùy chỉnh ban đầu (có thể rỗng để chờ thêm điểm)
X_custom = np.array([[5.5, 2.5], [6.5, 3.0], [5.8, 2.8], [6.0, 3.4], [7.2, 3.6], [4.8, 3.0]])
y_custom = np.array([0, 1, 1, 2, 2, 0])

# Tham số K ban đầu
k_init = 1

# Tạo lưới điểm trên không gian 2D
def create_meshgrid(X):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

xx, yy = create_meshgrid(X_custom)

# Cờ để chọn nhãn cho điểm mới (mặc định là màu đỏ)
current_label = 0

# Hàm vẽ biểu đồ KNN với giá trị K tùy chỉnh
def plot_knn(k):
    try:
        k = int(k)  # Chuyển đổi giá trị k nhập vào thành số nguyên
    except ValueError:
        return  # Nếu nhập không phải là số nguyên thì không cập nhật

    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_custom, y_custom)

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])  # Màu nền
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])  # Màu của các điểm

    ax.clear()
    ax.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Vẽ các điểm dữ liệu tùy chỉnh
    ax.scatter(X_custom[:, 0], X_custom[:, 1], c=y_custom, cmap=cmap_bold,
               edgecolor='k', s=25) # s la kich thuoc cua cac dau cham
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(f"K = {k}")
    plt.draw()

# Hàm xử lý sự kiện chuột
def on_click(event):
    if event.inaxes != ax:
        return
    global X_custom, y_custom, current_label
    X_custom = np.vstack([X_custom, [event.xdata, event.ydata]])  # Thêm điểm mới vào tọa độ
    y_custom = np.append(y_custom, current_label)  # Gán nhãn cho điểm mới
    plot_knn(text_box.text)  # Vẽ lại biểu đồ với điểm mới và giá trị K hiện tại

# Hàm để chọn nhãn màu cho điểm mới (đỏ, xanh lá, xanh dương)
def set_label_red(event):
    global current_label
    current_label = 0  # Màu đỏ

def set_label_green(event):
    global current_label
    current_label = 1  # Màu xanh lá

def set_label_blue(event):
    global current_label
    current_label = 2  # Màu xanh dương

# Hàm tải dữ liệu mẫu từ bộ Iris
def load_sample_data(event):
    global X_custom, y_custom, xx, yy
    iris = datasets.load_iris()
    X_custom = iris.data[:, :2]  # Chỉ sử dụng 2 đặc trưng đầu tiên để dễ biểu diễn
    y_custom = iris.target
    xx, yy = create_meshgrid(X_custom)
    plot_knn(text_box.text)

# Thiết lập đồ thị
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.1)

# Vẽ KNN với giá trị k ban đầu
plot_knn(k_init)

# Tạo ô nhập liệu (TextBox) để thay đổi giá trị K
axbox = plt.axes([0.1, 0.2, 0.1, 0.05])
text_box = TextBox(axbox, 'Nhập giá trị K:', initial=str(k_init))

# Cập nhật biểu đồ khi giá trị K thay đổi
text_box.on_submit(plot_knn)

# Tạo nút chọn màu đỏ
ax_red = plt.axes([0.1, 0.9, 0.1, 0.075])
btn_red = Button(ax_red, 'Chấm Đỏ', color='red', hovercolor='lightcoral')
btn_red.on_clicked(set_label_red)

# Tạo nút chọn màu xanh lá
ax_green = plt.axes([0.1, 0.8, 0.1, 0.075])
btn_green = Button(ax_green, 'Chấm Xanh Lá', color='green', hovercolor='lightgreen')
btn_green.on_clicked(set_label_green)

# Tạo nút chọn màu xanh dương
ax_blue = plt.axes([0.1, 0.7, 0.1, 0.075])
btn_blue = Button(ax_blue, 'Chấm Xanh Dương', color='blue', hovercolor='lightblue')
btn_blue.on_clicked(set_label_blue)

# Tạo nút để tải bộ dữ liệu mẫu
ax_load = plt.axes([0.1, 0.6, 0.1, 0.075])
btn_load = Button(ax_load, 'Tải dữ liệu mẫu', color='gray', hovercolor='lightgray')
btn_load.on_clicked(load_sample_data)

# Gắn sự kiện click chuột vào biểu đồ
cid = fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()
