# AISE4020-group3

# AI-Based Fire Hazard Prediction

## Overview
In recent years, the integration of artificial intelligence (AI) into fire hazard detection has become increasingly vital due to the rising need for early-warning systems in industrial and residential environments. Fire hazards pose a significant risk to human safety and infrastructure, and traditional detection methods, such as smoke and heat sensors, often provide alerts only after combustion has already begun. To improve early detection and predictive capabilities, AI-driven models can analyze environmental data trends to assess fire risk dynamically.

This project leverages AI models, including Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks, to predict fire hazards based on various environmental parameters. The model continuously analyzes trends in temperature, pressure, humidity, and gas concentrations to assess the likelihood of a fire hazard before ignition occurs. By incorporating machine learning techniques for time-series forecasting, this system enables proactive fire risk management and enhances safety measures in industrial and household settings.

## Features
- **Time-Series Prediction**: Uses LSTM and GRU models to predict fuel concentration in the air.
- **Real-Time Hazard Detection**: Evaluates key environmental parameters to determine fire risks.
- **Data Preprocessing**: Implements normalization, feature selection, and sequence creation for time-series analysis.
- **Threshold-Based Warning System**: Identifies hazardous conditions based on predefined flammability limits.
- **Multiple AI Models**: Compares the performance of LSTM and GRU for hazard prediction.

## Dataset
The dataset used for training and testing is derived from IoT-based smoke detection data. It includes:
- **Temperature (°C)**
- **Humidity (%)**
- **Pressure (hPa)**
- **Ethanol Concentration (ppm)**
- **Particulate Matter Levels**
- **Adj_Fuel% (Target variable for prediction)**

## Data Preprocessing
1. Load the dataset from CSV files.
2. Convert timestamps to datetime format.
3. Normalize the features using MinMaxScaler.
4. Create sequences for time-series forecasting.
5. Split the dataset into training and testing sets.

## Model Implementation
### GRU Model (PyTorch)
```python
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
```

### LSTM Model (TensorFlow/Keras)
```python
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
```

## Training and Evaluation
- **GRU Model**: Trained using Mean Squared Error loss and Adam optimizer for 20 epochs.
- **LSTM Model**: Trained using Mean Squared Error loss and Adam optimizer for 10 epochs.

### Performance Metrics
- Training and validation loss plots.
- Prediction vs. actual values visualization.
- RMSE and MAE for model evaluation.

## Flammability Equations
The project uses the following equations to determine the flammability limits based on temperature:
```
LFLt = LFL25 + 0.75/dH (T-25)
UFLt = UFL25 + 0.75/dH (T-25)
```
Where:
- LFLt/UFLt = Lower and Upper Flammability Limits at temperature T.
- LFL25/UFL25 = Flammability limits at 25°C.
- dH = Enthalpy change.
- T = Temperature in °C.

## Hazard Warning System
A predefined warning system alerts users when predictions fall within a critical range:
```python
lower_bound = 4.3  # Example lower bound
upper_bound = 19   # Example upper bound

for i, pred in enumerate(predictions_rescaled):
    if lower_bound <= pred <= upper_bound:
        print(f"Warning: Prediction {pred[0]:.5f} at index {i} is within the specified range ({lower_bound} - {upper_bound}).")
```

## Installation and Usage
### Prerequisites
- Python 3.x
- PyTorch
- TensorFlow
- Scikit-learn
- Pandas
- Matplotlib

### Installation
```bash
pip install numpy pandas torch torchvision torchaudio scikit-learn tensorflow matplotlib
```

### Running the Model
1. Clone the repository:
   ```bash
   git clone https://github.com/eeshasharmaa/AISE4020-group3.git
   cd AISE4020-group3
   ```
2. Run the preprocessing script.
3. Train the model using `train.py`.
4. Evaluate and visualize predictions.

## Authors
Evan Park, Aysha Nuh, Trent Jones, Eesha Sharma, Mitchell Strasman, Connor Gilbert
