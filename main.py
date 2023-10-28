import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Ví dụ về việc sử dụng dữ liệu iris từ scikit-learn
data = datasets.load_iris()
X = data.data
y = data.target

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Khởi tạo một mô hình SVM
model = SVC(kernel='linear', C=1.0)

# Huấn luyện mô hình trên dữ liệu huấn luyện
model.fit(X_train, y_train)

# Dự đoán kết quả trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

from sklearn.model_selection import GridSearchCV

# Thiết lập các giá trị tham số bạn muốn thử nghiệm
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

# Sử dụng Grid Search để tìm giá trị tốt nhất cho các tham số
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# In ra các giá trị tốt nhất
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best accuracy: {grid_search.best_score_}')

# Đánh giá mô hình tốt nhất trên tập kiểm tra
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Accuracy on test set using the best model: {test_accuracy}')
