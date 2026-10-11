#import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
import time

csv_path = r'data\100 data_sample.csv'  # put your path here
nData = int(csv_path.replace('\\', ' ').split(' ')[1]) #number of data points in the csv file

df = pd.read_csv(csv_path) #dataframe from csv file

x = df[['alpha','Lambda', 'sigma','Kappa','Kappa_','A','B','m1','m2']].values.astype(np.float64) #x columns as input features
y = df[['E0','E1','E2','E3']].values.astype(np.float64) #y columns as output features

# data shuffling and splitting into training and validation sets
x_train_raw, x_val_raw, y_train_raw, y_val_raw = train_test_split(
    x, y, test_size=0.2, random_state=42, shuffle=True
)

# normalize using train stats 
x_mean, x_std = x_train_raw.mean(axis=0), x_train_raw.std(axis=0)
y_mean, y_std = y_train_raw.mean(axis=0), y_train_raw.std(axis=0)

x_train = (x_train_raw - x_mean) / x_std
x_val = (x_val_raw - x_mean) / x_std
y_train = (y_train_raw - y_mean) / y_std
y_val = (y_val_raw - y_mean) / y_std

#defining model architecture, 9 neuron in the input layer, 3 hidden layers with 64 neurons each and SiLU activation function, output layer with 4 neurons
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(9,)),
    tf.keras.layers.Dense(64, activation = 'gelu'),
    tf.keras.layers.Dense(64, activation = 'gelu'),
    tf.keras.layers.Dense(64, activation = 'gelu'),
    tf.keras.layers.Dense(4)
])

model.summary() #summarize the model

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error')

# adding early stopping to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
)

BATCH_SIZE = min(256, max(16, nData//20)) #dynamically set batch size based on dataset size, with limits

start_time = time.time()
history = model.fit(
    x_train, y_train,
    epochs=400,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    callbacks=[early_stop],
    verbose=1,
)
training_time = time.time() - start_time #training time in seconds
print(f"Training time: {training_time:.2f} seconds")
print(f"Stopped at epoch {len(history.history['loss'])} (best weights restored)")

# Save the model
model.save(rf'data\{nData}model.h5')
model.save_weights(rf'data\{nData}model.weights.h5')

# trace inference time for the validation set
start_time = time.time()
predictions = model.predict(x_val)
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.4f} seconds for {len(x_val)} points "
      f"({inference_time / len(x_val) * 1000:.4f} ms/point)")

# Unscale predictions and ground-truth back to physical units (MeV)
pred_phys = predictions * y_std + y_mean
y_val_phys = y_val * y_std + y_mean

mae_per_output = np.mean(np.abs(pred_phys - y_val_phys), axis=0)
overall_mae_mev = np.mean(mae_per_output)

mae_sentence = (f"Test MAE = {overall_mae_mev:.5f} MeV (per-output MAE: "
                + ", ".join([f"{m:.5f}" for m in mae_per_output]) + " MeV)")
print(mae_sentence)

# plot training loss
plt.figure(figsize=(10.8, 7.2))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.title(f'Training and Validation Loss for {nData} samples\n'
          f'Training Time: {training_time:.2f}s, Inference Time: {inference_time:.4f}s '
          f'(stopped at epoch {len(history.history["loss"])})')

plt.text(0.5, 1.12, mae_sentence, ha='center', va='bottom',
          transform=plt.gca().transAxes, fontsize=12)

plt.tight_layout()
plt.show()