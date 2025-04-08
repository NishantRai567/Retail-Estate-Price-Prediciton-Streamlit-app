#############################################################################
# Making Streamlit app for Predicting Retail Prices
#############################################################################


# Import required packages
import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import time

# Import housing data

california_housing=fetch_california_housing()

df=pd.DataFrame(california_housing.data,columns=california_housing.feature_names)

df["Price"]=california_housing.target

# Split data into inputs and outputs

X=df.drop(["Price"],axis=1)
y=df["Price"]

# Split data into train and test sets

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)


########################################################
# Set up Pipelines
########################################################

transformer=Pipeline(steps=[("imputer",SimpleImputer()),
                            ("scaler", StandardScaler())])

##############################################################
# Apply the Pipeline
##############################################################

# LinearReegression

model=Pipeline(steps=[("preprocessing_pipeline",transformer),
               ("regressor", LinearRegression())])

model.fit(X_train,y_train)
y_pred=model.predict(X_test)
r2_score(y_pred,y_test)



# Random Forest

model=Pipeline(steps=[("preprocessing_pipeline", transformer),
                    ("regressor", RandomForestRegressor(random_state=42))])

model.fit(X_train,y_train)
y_pred_class=model.predict(X_test)
r2_score(y_test,y_pred_class)

# Save Pipeline

joblib.dump(model,"retail_model.joblib")


###############################################################################
# Streamlit app
###############################################################################

# Set page config
st.set_page_config(page_title="Retail-Estate Price Prediction", layout="wide")
# add title and instructions
st.title("🏠 Retail-Estate Price Prediction")
st.subheader("Enter details for house and submit to predict Price")


st.markdown("""
**Note**: This model is trained on a dataset from the 1990s. As such, the predictions may not fully account for changes in housing market trends, inflation, or other modern factors. Use the predictions with caution as they may not reflect current market conditions.
""")

# User inputs


st.subheader("House Features")
    

input_data = {
    'MedInc': st.number_input("Median Income(in thousands if dollars)", min_value=0, max_value=15, value=8),
    'AveOccup': st.number_input("Average Occupancy(people per household)", min_value=0, max_value=10, value=2),
    'AveRooms': st.number_input("Average Rooms", min_value=0, max_value=10, value=6),
    'AveBedrms': st.number_input("Average Bedrooms", min_value=0, max_value=10, value=1),
    'Population': st.number_input("Population(number of people in the area)", min_value=0, max_value=1000000, value=500),
    'HouseAge': st.number_input("House Age(years)", min_value=0, max_value=100, value=20),
    'Latitude': st.number_input("Latitude(degrees)", min_value=32, max_value=34, value=34),
    'Longitude': st.number_input("Longitude(degrees)", min_value=-120, max_value=-118, value=-118)
}



# submit inputs to model


if st.button("🔍 Submit For Prediction"):
       
   # Show spinner while predicting
   with st.spinner("Making prediction..."):
        
        # Simulating delay (remove this in production)
        time.sleep(2)  # Simulate a delay
    
        # store our data in a datframe for prediction
        new_data=pd.DataFrame([input_data])
    
        # apply model pipeline to the input data and extract probability prediction
        predicted_price=model.predict(new_data)*1000


        # Show result
        st.write(f"Predicted House Price: ${predicted_price[0]:,.2f}")








