import numpy as np
import matplotlib.pyplot as plt

# 1. Generate dummy data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 2. Initialize parameters
learning_rate = 0.1
n_epochs = 50
m = len(X)
theta = np.random.randn(2,1) # Random initialization
X_b = np.c_[np.ones((100, 1)), X] # Add bias term

# List to store loss values
loss_history = []

# 3. Training Loop
for epoch in range(n_epochs):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients
    
    # Calculate and store loss (MSE)
    prediction = X_b.dot(theta)
    loss = np.mean((prediction - y) ** 2)
    loss_history.append(loss)

# 4. Plotting the Loss Curve
plt.plot(range(n_epochs), loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Progress: Loss per Epoch')
plt.show()