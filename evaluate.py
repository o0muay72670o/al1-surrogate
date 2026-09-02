# import essential libraries
import numpy as np
import pandas as pd
import tensorflow as tf

df = pd.read_csv(r"..\heavy-quark-twobody\data\100MWOgen.csv") #read csv file

feature_cols = [
    "alpha",
    "Lambda",
    "sigma",
    "Kappa",
    "Kappa_",
    "A",
    "B",
    "m1",
    "m2",
]
target_cols = ["E0", "E1", "E2", "E3"] # columns for target values

x = df[feature_cols].values.astype(np.float32) 
y = df[target_cols].values.astype(np.float32)
x_mean, x_std = x.mean(axis=0), x.std(axis=0) # normalizing the data
y_mean, y_std = y.mean(axis=0), y.std(axis=0)

model = tf.keras.models.load_model((r"..\models\100MWOgen_model.h5")) # load the model
m1_var = float(input("enter mass one=")) # user input for mass one
m2_var = float(input("enter mass two=")) # user input for mass two
sigma_var = float(input("enter sigma=")) # user input for spin coupling constant

#coefficients for AL1 
alpha = 0.8321
Lambda = 0.1653
kappa = 0.5069
kappa_ = 1.8609
A = 1.6553
B = 0.2204

# validate model with user inputs
test_inputs = np.array(
    [[alpha, Lambda, sigma_var, kappa, kappa_, A, B, m1_var, m2_var]],
    dtype=np.float32,
)
# scale the test inputs and make predictions
test_scaled = (test_inputs - x_mean) / x_std
preds_scaled = model.predict(test_scaled)
preds_GeV = (preds_scaled * y_std) + y_mean
preds_MeV = (preds_GeV * 1000.0).flatten()
# print the predictions in MeV
print(
    f"\n1S (E0): {preds_MeV[0]:.1f} MeV | 2S (E1): {preds_MeV[1]:.1f} MeV | 3S (E2): {preds_MeV[2]:.1f} MeV | 4S (E3): {preds_MeV[3]:.1f} MeV"
)
