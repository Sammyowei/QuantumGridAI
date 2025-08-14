import tensorflow as tf

# Path to your old model
old_model_path = "../models/lstm_load_forecast.h5"

# Path to your new model
new_model_path = "../models/lstm_load_forecast.keras"

# Load old model (ignore compilation state)
model = tf.keras.models.load_model(old_model_path, compile=False)

# Save new model in modern format without optimizer
model.save(new_model_path, include_optimizer=False)

print("✅ Model successfully converted to .keras format!")
