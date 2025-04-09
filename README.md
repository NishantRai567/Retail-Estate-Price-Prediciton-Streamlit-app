# Retail-Estate-Price-Prediction-Streamlit-app

## Overview
This web app predicts the price of a property based on various user inputs related to the property. The user enters details such as median income,number of bedrooms etc., and the app uses a trained RandomForestRegressor model to predict the estimated price of the property. The inputs are stored in a pandas dataframe and passed through a data preprocessing pipeline to generate predictions.

## Features
- Predict Property Prices: The app predicts the price of real estate based on user inputs.
- User-Friendly Interface: Built with Streamlit for a smooth, interactive user experience.
- Data Preprocessing: Inputs are processed and transformed before being passed through the prediction model.
- Machine Learning Model: Utilizes a RandomForestRegressor model for accurate price predictions.

## Technologies Used

- Streamlit:For creating the web app interface.
- Pandas:For handling data input and manipulation.
- Scikit-learn:For implementing the RandomForestRegressor model.
- Python:Programming language for backend logic.

## How to Use the App

### 1.Run the App Locally

To run the app on your local machine, follow these steps:

1.Clone the Repository: Clone the repository to your local machine:

```bash
git clone https://github.com/NishantRai567/Retail-Estate-Price-Prediciton-Streamlit-app.git
```
2.Navigate to the Project Directory:

```bash
cd Retail-Estate-Price-Prediciton-Streamlit-app
```
3.Install Dependencies: Install the required dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

4.Run the App: Launch the Streamlit app locally:

```bash
streamlit run "Predicting retail prices.py"
```

5.Access the App: The app will open in your browser at http://localhost:8501, where you can input the property details and get price predictions.

### 2. Access the App Online

If you don’t want to run the app locally, you can access the app directly via its deployment on Streamlit Sharing or another cloud platform.

Visit the deployed app at: https://retail-estate-price-prediciton-app-app-gvqobwekarnppdm2a9bdj8.streamlit.app/

Once on the site, you can enter the required property details and receive the predicted price.
