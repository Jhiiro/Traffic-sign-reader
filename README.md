# Traffic Sign Classifier

A CNN trained on the GTSRB dataset to recognize 43 traffic sign categories.

## Results

| Split      | Accuracy |
|------------|----------|
| Validation | ~99.67%  |
| Test       | 94.81%   |

## Setup
```bash
pip install tensorflow opencv-python scikit-learn pandas matplotlib numpy

## Dataset

Download [GTSRB from Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) and place it like this:


## Model

Three conv blocks (Conv2D → BatchNorm → MaxPool), then Flatten → Dropout(0.5) → Dense(128) → Dense(43).

Trained for 20 epochs with Adam and categorical crossentropy.

## Usage

Train the model:

bash
python train.py

Predict a single image:

python
from tensorflow.keras.models import load_model
from predict import predict_and_show

model = load_model("traffic_sign_model.h5")
predict_and_show("path/to/sign.png", model)

## Notes

- Input images are resized to 32×32 — heavily degraded images may misclassify
- Trained and evaluated on GTSRB only; other regional sign standards aren't covered
- No data augmentation was used during training
