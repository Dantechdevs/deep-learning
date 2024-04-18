import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Generate some synthetic data (you can replace this with your own dataset)
num_samples = 1000
input_dim = 10
latent_dim = 2

# Create random data
X = np.random.rand(num_samples, input_dim)

# Define the autoencoder architecture
input_layer = Input(shape=(input_dim,))
encoded = Dense(latent_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Create the autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.2)

# Get the encoder part (for dimensionality reduction)
encoder = Model(inputs=input_layer, outputs=encoded)

# Encode the data
encoded_data = encoder.predict(X)

# Visualize the original and encoded data
plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c='b', label='Encoded Data')
plt.scatter(X[:, 0], X[:, 1], c='r', label='Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Autoencoder for Dimensionality Reduction')
plt.legend()
plt.show()
