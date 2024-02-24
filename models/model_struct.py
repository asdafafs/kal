import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(json_file_path):
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    data_list = []
    for frame in json_data['frames']:
        row_dict = {}
        for key, value in frame['fields'].items():
            column_name = f"{key}"
            row_dict[column_name] = value.get('value')
        data_list.append(row_dict)

    df = pd.DataFrame(data_list)
    df = df.drop(['dest_callsign', 'src_callsign', 'src_ip_addr', 'src_port',
                  'dst_ip_addr', 'pid', 'time', 'boottime'], axis=1)

    return df


def preprocess_data(df, target_column):
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

    return x_train, x_test, y_train, y_test


def create_sequences(x_train, y_train, x_test, y_test, sequence_length):
    x_train_sequences = [x_train.iloc[i:i + sequence_length].values for i in range(len(x_train) - sequence_length)]
    y_train_sequences = [y_train.iloc[i + sequence_length] for i in range(len(y_train) - sequence_length)]

    x_test_sequences = [x_test.iloc[i:i + sequence_length].values for i in range(len(x_test) - sequence_length)]
    y_test_sequences = [y_test.iloc[i + sequence_length] for i in range(len(y_test) - sequence_length)]

    x_train_tensor = torch.FloatTensor(np.array(x_train_sequences))
    y_train_tensor = torch.FloatTensor(np.array(y_train_sequences))

    x_test_tensor = torch.FloatTensor(np.array(x_test_sequences))
    y_test_tensor = torch.FloatTensor(np.array(y_test_sequences))

    x_train_tensor = x_train_tensor.to('cuda')
    y_train_tensor = y_train_tensor.to('cuda')
    x_test_tensor = x_test_tensor.to('cuda')
    y_test_tensor = y_test_tensor.to('cuda')

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=15):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[-1, :])
        return out


def train_model(model, x_train_tensor, y_train_tensor, num_epochs, batch_size, criterion, optimizer):
    X_train_batches = [x_train_tensor[i:i + batch_size] for i in range(0, len(x_train_tensor), batch_size)]
    y_train_batches = [y_train_tensor[i:i + batch_size] for i in range(0, len(y_train_tensor), batch_size)]

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()

        for i in range(len(X_train_batches)):
            outputs = model(X_train_batches[i])
            loss = criterion(outputs.squeeze(), y_train_batches[i])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_predictions = model(x_train_tensor)
                train_mse = mean_squared_error(train_predictions.cpu().squeeze().numpy(), y_train_tensor.cpu().numpy())
            print(
                f'Epoch [{epoch + 1}/{num_epochs} (Batch {i + 1})], Loss: {loss.item():.4f}, Train MSE: {train_mse:.4f}')


def test_model(model, x_test_batches, y_test_batches, criterion):
    model.eval()
    with torch.no_grad():
        for i in range(len(x_test_batches)):
            test_inputs = x_test_batches[i]
            predictions = model(test_inputs)
            predictions_np = predictions.cpu().squeeze().numpy()
            y_test_np = y_test_batches[i].cpu().numpy()
            test_loss = criterion(torch.from_numpy(predictions_np), torch.from_numpy(y_test_np))
            test_mse = mean_squared_error(predictions_np, y_test_np)
            print(f'Test Loss (Batch {i + 1}): {test_loss:.4f}, Test MSE: {test_mse:.4f}')


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Модель сохранена в файл {model_path}")


class RNNPredictor:
    def __init__(self, model_path, input_size, hidden_size, output_size, num_layers=15):
        self.model_path = model_path
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.model = SimpleRNN(input_size, hidden_size, output_size, num_layers=num_layers)  # Обратите внимание на
        # параметр num_layers
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print("Веса успешно загружены!")
        else:
            print("Файл с весами не найден.")
        self.model.eval()

    def predict(self, new_data):
        with torch.no_grad():
            new_data_tensor = torch.FloatTensor(new_data)
            model_output = self.model(new_data_tensor)
            predicted_value = model_output.item()
        return predicted_value
