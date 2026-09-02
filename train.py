#import necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = r'..\data.csv' #put your path here

df = pd.read_csv(csv_path) #dataframe from csv file

x = df[['alpha','Lambda', 'sigma','Kappa','Kappa_','A','B','m1','m2']].values.astype(np.float64) #x columns as input features
y = df[['E0','E1','E2','E3']].values.astype(np.float64) #y columns as output features

#normalizing the data
x_mean, x_std = x.mean(axis=0), x.std(axis=0)
y_mean, y_std = y.mean(axis=0), y.std(axis=0)
x_scaled = (x - x_mean) / x_std
y_scaled = (y - y_mean) / y_std


#defining model architecture, 9 neuron in the input layer, 3 hidden layers with 64 neurons each and SiLU activation function, output layer with 4 neurons
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(9,)),
    tf.keras.layers.Dense(64, activation = 'silu'),
    tf.keras.layers.Dense(64, activation = 'silu'),
    tf.keras.layers.Dense(64, activation = 'silu'),
    tf.keras.layers.Dense(4)
])

model.summary() #summarize the model

#compiling the model with Adam optimizer and mean squared error loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error')
model.fit(x_scaled, y_scaled, epochs = 200, verbose=1)

#shuffling the data and splitting into training and testing sets
x_train, x_val, y_train, y_val = train_test_split(x_scaled, y_scaled, test_size = 0.2, random_state = 42, shuffle = True)
#Add validation split
history = model.fit(x_train, y_train, epochs =200, batch_size =  256, validation_data = (x_val, y_val), verbose = 1,)

model.save(r'..\model.h5') #save the model
model.save_weights(r'..\model.weights.h5') #save the model weights 

#plot training loss
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
