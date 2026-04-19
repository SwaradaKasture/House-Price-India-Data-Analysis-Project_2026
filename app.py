import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

st.set_page_config(page_title="House Price India EDA & Modeling", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("House Price India.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset 'House Price India.csv' not found. Please ensure it is in the same directory as this script.")
    st.stop()

st.sidebar.title("Navigation")
menu = ["Exploratory Data Analysis", "Machine Learning Models"]
choice = st.sidebar.selectbox("Select a Module", menu)

if choice == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis 📊")
    
    st.header("Dataset Overview")
    st.write("First 5 rows of the dataset:")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset Shape:", df.shape)
    with col2:
        missing_vals = df.isnull().sum()
        st.write("Missing Values:")
        st.write(missing_vals[missing_vals > 0] if not missing_vals[missing_vals > 0].empty else "No missing values found.")
        
    st.header("Summary Statistics")
    st.write(df.describe())
    
    st.header("Data Visualizations")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Price'], bins=50, kde=True, ax=ax, color='royalblue')
        ax.set_title("Distribution of House Prices")
        st.pyplot(fig)
    
    with col4:
        st.subheader("Living Area vs Price")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='living area', y='Price', data=df, ax=ax, alpha=0.5, color='darkorange')
        ax.set_title("Living Area vs House Price")
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    st.write("Showing correlations for features most correlated to Price (>0.4)")
    corr = df.corr()
    top_corr_features = corr.index[abs(corr["Price"]) > 0.4]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    st.pyplot(fig)

elif choice == "Machine Learning Models":
    st.title("Machine Learning Models 🤖")
    st.write("Train models to predict House `Price`.")
    
    # Preprocessing
    # Dropping non-predictive or redundant columns like 'id', 'Date' for a baseline.
    drop_cols = ['id', 'Date']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns] + ['Price'])
    y = df['Price']
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Hyperparameters")
    test_size = st.sidebar.slider("Test Size (Fraction)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.sidebar.slider("Random State", min_value=0, max_value=100, value=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    st.write(f"Training data shape: {X_train.shape}")
    st.write(f"Testing data shape: {X_test.shape}")
    
    model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest Regressor"])
    
    if st.button("Train Model"):
        with st.spinner(f"Training {model_choice}..."):
            if model_choice == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
                
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            
            st.success("Model trained successfully!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("R-squared Score", f"{r2:.4f}")
            col2.metric("Mean Absolute Error", f"{mae:,.2f}")
            
            st.subheader("Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, predictions, alpha=0.5, color='seagreen')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_xlabel("Actual Price")
            ax.set_ylabel("Predicted Price")
            ax.set_title("Actual vs Predicted Prices")
            st.pyplot(fig)
