import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from models.model_struct import SimpleRNN, train_model, save_model
from models.model_struct import load_data, preprocess_data, create_sequences

print('начало')

json_file_path = r'C:\Users\anton\Downloads\Telegram Desktop\normalized_frames.json'
df = load_data(json_file_path)

# Предобработка данных
target_column = "nx_tmp"
x_train, x_test, y_train, y_test = preprocess_data(df, target_column)

# Создание последовательностей
sequence_length = 10  # Выберите подходящую длину последовательности
x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = create_sequences(
    x_train, y_train, x_test, y_test, sequence_length
)

# Определение параметров модели
input_size = x_train_tensor.shape[2]  # Входной размер равен количеству признаков
hidden_size = 64  # Например, 64 нейрона в скрытом слое
output_size = 1  # Выходной размер, предположим, у вас одномерный выход

# Инициализация модели
model = SimpleRNN(input_size, hidden_size, output_size).to('cuda')

# Определение параметров обучения
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Обучение модели
train_model(model, x_train_tensor, y_train_tensor, num_epochs, batch_size, criterion, optimizer)

# Тестирование модели
model.eval()
with torch.no_grad():
    test_predictions = model(x_test_tensor)
    test_mse = mean_squared_error(test_predictions.cpu().squeeze().numpy(), y_test_tensor.cpu().numpy())
    print(f'Test MSE: {test_mse:.4f}')

# Сохранение модели
model_path = "trained_model.pt"
save_model(model, model_path)
