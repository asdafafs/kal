import torch
from fastapi import FastAPI
from pydantic import BaseModel
from models.model_struct import load_data, RNNPredictor

# Загрузка модели
path = 'models/trained_model.pt'  # Укажите путь к вашей модели
predictor = RNNPredictor(model_path=path, input_size=202, hidden_size=64, output_size=1)

all_columns = load_data(r'C:\Users\anton\Downloads\Telegram Desktop\normalized_frames.json').columns

print(all_columns, 'в мейне')

# Укажите колонки, которые вы хотите исключить
excluded_columns = ['dest_callsign', 'src_callsign', 'src_ip_addr', 'src_port', 'dst_ip_addr', 'pid', 'time',
                    'boottime', 'wheel_rpm', ]

# Создайте список полей, исключив указанные колонки
field_names = [col for col in all_columns if col not in excluded_columns]

app = FastAPI()


class UserInput(BaseModel):
    __annotations__ = {field: float for field in field_names}


@app.post('/predict/')
async def predict(user_input: UserInput):
    input_data = torch.FloatTensor([[user_input.__dict__[field] for field in field_names]])
    prediction = predictor.predict(input_data)
    prediction_value = prediction
    return {"predict value nx_tmp": float(prediction_value)}
