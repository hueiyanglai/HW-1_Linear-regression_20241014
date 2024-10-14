# streamlit_app.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit App
def main():
    st.title("House Price Prediction with Linear Regression")

    # User inputs
    st.sidebar.header("User Inputs")
    num_houses = st.sidebar.slider("Number of Houses", min_value=10, max_value=1000, value=100, step=10)
    std_dev_noise = st.sidebar.slider("Standard Deviation of Noise", min_value=10000, max_value=100000, value=30000, step=5000)
    train_size = st.sidebar.slider("Percentage of Training Data", min_value=10, max_value=90, value=80, step=10) / 100.0

    # Generate random data for houses
    np.random.seed(42)  # For reproducibility
    sizes = np.random.randint(1000, 4000, size=num_houses)  # House size between 1000 and 4000 sqft
    bedrooms = np.random.randint(1, 6, size=num_houses)     # Bedrooms between 1 and 5

    # Add normally distributed noise with user-defined standard deviation
    noise = np.random.normal(0, std_dev_noise, size=num_houses)

    # Calculate prices with a base formula and noise
    prices = sizes * 150 + bedrooms * 10000 + noise

    # Create DataFrame
    df = pd.DataFrame({
        'Size': sizes,
        'Bedrooms': bedrooms,
        'Price': prices
    })

    # Prepare data
    X = df[['Size', 'Bedrooms']]
    y = df['Price']

    # Split data into training and testing sets based on user-defined percentage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Fix the number of bedrooms for predictions to ensure points are on the regression line
    bedrooms_constant = 3
    X_test_fixed = X_test.copy()
    X_test_fixed['Bedrooms'] = bedrooms_constant  # Set bedrooms to constant value

    # Predict using the fixed number of bedrooms
    y_pred_fixed = model.predict(X_test_fixed)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred_fixed)
    r2 = r2_score(y_test, y_pred_fixed)

    # Display results
    st.subheader("Model Evaluation Metrics")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")

    # Plotting
    fig, ax = plt.subplots()
    sorted_idx = X_test['Size'].argsort()
    X_test_sorted = X_test.iloc[sorted_idx]
    y_test_sorted = y_test.iloc[sorted_idx]
    y_pred_sorted = y_pred_fixed[sorted_idx]

    # Plot the actual vs predicted values for the test data
    ax.scatter(X_test_sorted['Size'], y_test_sorted, color='black', label='Actual Price (Test Points)', alpha=0.6)
    ax.scatter(X_test_sorted['Size'], y_pred_sorted, color='red', label='Predicted Price (Test Points)', marker='x', alpha=0.8)

    # Generate a regression line using the 'Size' feature while keeping 'Bedrooms' constant
    size_range = np.linspace(X_test_sorted['Size'].min(), X_test_sorted['Size'].max(), 100)
    predicted_regression_line = model.predict(pd.DataFrame({'Size': size_range, 'Bedrooms': [bedrooms_constant] * 100}))

    # Plot the regression line
    ax.plot(size_range, predicted_regression_line, color='blue', label='Regression Line')

    # Add labels and legend
    ax.set_xlabel('Size of House (sqft)')
    ax.set_ylabel('Price')
    ax.set_title('House Price Prediction')
    ax.legend()

    # Show the plot in Streamlit
    st.pyplot(fig)

if __name__ == "__main__":
    main()
