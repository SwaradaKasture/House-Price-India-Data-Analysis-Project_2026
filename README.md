Student Details - 
•	Name : Kasture Swarada Pramod

•	E-mail : swaradakasture0206@gmail.com

•	Class : T.Y. ECM

•	Roll Number : 33

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# House Price India EDA & Modeling

This project provides an interactive web application built with **Streamlit** to perform Exploratory Data Analysis (EDA) and apply Machine Learning models to the House Price India dataset.

## Features

- **Exploratory Data Analysis (EDA)**
  - View the initial rows of the dataset and its shape.
  - Explore dataset summary statistics and check for missing values.
  - Interactive Visualizations:
    - Histogram representing the distribution of house prices.
    - Scatter plots comparing key features like Living Area vs. Price.
    - An annotated Correlation Heatmap detailing the strongest predictors of house pricing.

- **Machine Learning Models**
  - Adjust hyper-parameters interactively, such as the Train/Test Split fraction and the Random Seed state.
  - Compare the performance of two popular algorithms:
    - Linear Regression
    - Random Forest Regressor
  - Evaluate the model using essential performance metrics:
    - R-squared ($R^2$) Score
    - Mean Absolute Error (MAE)
  - Visualize predictions vs. actual data points to understand the model accuracy.

## Prerequisites

Ensure you have Python installed. You can install all project dependencies by making use of the provided `requirements.txt` file.

1. Open a terminal or command prompt inside the project folder.
2. Run the following command:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

1. Check that the dataset file named **`House Price India.csv`** is situated in the exact same directory as `app.py`.
2. Start the Streamlit server from your terminal:
   ```bash
   streamlit run app.py
   ```
3. Your default web browser will open and automatically route to the running application (typically at `http://localhost:8501`).

## Data Preprocessing Details

For the baseline machine learning predictions, non-predictive or redundant properties such as the unique `id` and `Date` fields are automatically dropped. The `Price` functions as the target variable ($y$). 

> Note: For heavier algorithms like *Random Forest Regressor*, training against local environments on the entire dataset might take several moments.
