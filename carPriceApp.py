import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib
import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv('automobile.csv')


# Add header and Subheader 
st.markdown("<h1 style = 'color: #124076; text-align: center; font-size: 60px; font-family: Helvetica'>CAR PRICE PREDICTOR APP</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Esther Nzekwe</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

# Add an Image 
st.image('pngwing.com(5).png' , caption = 'Built By Esther Nzekwe')

# Add Project Problem Statement
st.markdown("<h2 style = 'color: #132043; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)
st.markdown("<p>The aim of this project is to develop a machine learning model to predict car prices based on various attributes and features. The goal of this project is to create a reliable and accurate pricing model that can assist both car sellers and buyers in estimating the fair market value of a vehicle. By leveraging historical data on car attributes such as make, model, year, mileage, fuel type, and other relevant features, the model aims to provide valuable insights into pricing trends and factors influencing the value of cars in the market. Ultimately, the project seeks to empower stakeholders within the automotive industry with a robust tool for making informed pricing decisions and enhancing the overall car buying and selling experience</p>", unsafe_allow_html= True)


# Sidebar Designs 
st.sidebar.image('pngwing.com-15.png')
st.divider()

display_data = data.groupby('make')[['price']].mean()
st.dataframe(display_data , use_container_width= True)
st.caption('Car Models and their average prices')


st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

st.divider()


st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.sidebar.markdown("<br>", unsafe_allow_html= True)

st.sidebar.subheader('Input Variables)
# User Inputs 
make = st.sidebar.selectbox('Car Model', data['make'].unique())
sym = st.sidebar.number_input('Symboling', data['symboling'].min(), data['symboling'].max())
curb_weight = st.sidebar.number_input('Curb Weight', data['curb-weight'].min(), data['curb-weight'].max())
body = st.sidebar.selectbox('Body Style', data['body-style'].unique())
length = st.sidebar.number_input('Car Length', data['length'].min(), data['length'].max())
hp = st.sidebar.number_input('Horse Power', 0, 5000)
wheel = st.sidebar.number_input('Wheel Base', data['wheel-base'].min(), data['wheel-base'].max())
eng = st.sidebar.number_input('Engine Size', data['engine-size'].min(), data['engine-size'].max())
engType = st.sidebar.selectbox('Engine Type', data['engine-type'].unique())
city = st.sidebar.number_input('City MPG', data['city-mpg'].min(), data['city-mpg'].max())

sel_make = make

# import transformers 
body_encoder = joblib.load('body-style_encoder.pkl')
enginetype_encoder = joblib.load('engine-type_encoder.pkl')
make_encoder = joblib.load('make_encoder.pkl')

# make	symboling	curb-weight	body-style	length	city-mpg	horsepower	wheel-base	engine-size	engine-type

# user input Dataframe 
user_input = pd.DataFrame()
user_input['make'] = [make]
user_input['symboling'] = [sym]
user_input['curb-weight'] = [curb_weight]
user_input['body-style'] = [body]
user_input['length'] = [length]
user_input['city-mpg'] = [city]
user_input['horsepower'] = [hp]
user_input['wheel-base'] = [wheel]
user_input['engine-size'] = [eng]
user_input['engine-type'] = [engType]


st.markdown("<br>", unsafe_allow_html = True)
st.header('Input Variable')
st.dataframe(user_input, use_container_width = True)

# transform users input according to training scale and encoding
user_input['body-style'] = body_encoder.transform(user_input[['body-style']])
user_input['engine-type'] = enginetype_encoder.transform(user_input[['engine-type']])
user_input['make'] = make_encoder.transform(user_input[['make']])

# modelling  ---
model = joblib.load('carPricePredictor.pkl')

if make:
    if st.button(f'Predict the price of {sel_make}'):
        predicted_price = model.predict(user_input)
        st.success(f"The Predicted Price For {sel_make} Is {predicted_price[0].round(2)}")
