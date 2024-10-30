import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Manually inputting the data as per the image
# data = {
#     'SIZE': [32.50234527,53.42680403,61.53035803,47.47563963,59.81320787,55.14218841,52.21179669,39.29956669,48.10504169,52.55001444,45.41973014,54.35163488,44.1640495,58.16847072],
#     'PRICE': [31.70701, 68.7776, 62.56238, 71.54663, 87.23093, 78.21152, 79.64197,
#               59.17149, 75.33124, 71.30088, 55.16568, 82.47885, 62.00892, 75.39287]
# }
df = pd.read_csv('NairobiOfficePrice Ex.csv')


# Convert data to a pandas DataFrame
df = pd.DataFrame(df)

# Separate features and target variable
X = df['SIZE'].values
y = df['PRICE'].values


# Define Mean Squared Error (MSE) function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Define Gradient Descent function
def gradient_descent(X, y, m, c, learning_rate, epochs):
    n = len(y)
    errors = []

    for epoch in range(epochs):
        # Make predictions
        y_pred = m * X + c

        # Compute the current error
        error = mean_squared_error(y, y_pred)
        errors.append(error)
        print(f"Epoch {epoch + 1}, Mean Squared Error: {error}")

        # Compute gradients
        m_gradient = (-2 / n) * np.sum(X * (y - y_pred))
        c_gradient = (-2 / n) * np.sum(y - y_pred)

        # Update weights
        m -= learning_rate * m_gradient
        c -= learning_rate * c_gradient

    return m, c, errors


# Initialize slope (m) and intercept (c) with small random values
m_initial = 0
c_initial = 0

# Set learning rate and number of epochs
learning_rate = 0.0001
epochs = 10

# Train model using gradient descent
m_final, c_final, errors = gradient_descent(X, y, m_initial, c_initial, learning_rate, epochs)


# Predict the price for an office size of 100 sq. ft
predicted_price = m_final * 100 + c_final
print("Predicted price for 100 sq. ft office:", predicted_price)

# Plot data points and the line of best fit
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, m_final * X + c_final, color='red', label='Line of best fit')
plt.xlabel('Office Size (sq. ft)')
plt.ylabel('Office Price')
plt.title('Nairobi Office Price Prediction')
plt.legend()
plt.show()

