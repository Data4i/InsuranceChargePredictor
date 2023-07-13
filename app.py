import pickle

import streamlit as st

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title='Insurance Charge Predictor', layout = 'wide', page_icon='heart')

left_col, center_col, right_col = st.columns((1,2,1))


with center_col:
    st.header('Health Insurance Charge Predictor  ❤️')


@st.cache_resource
def get_model(filename:str):
    model = pickle.load(open(filename, 'rb'))
    return model

@st.cache_data
def get_used_df(df_filename):
    return pd.read_csv(df_filename)

df_filename = 'data/insurance.csv'
used_df = get_used_df(df_filename)

filename = 'models/insurance_forest2_model.pkl'
model = get_model(filename)

def feauture_engineering(df):
    # ordinalEncoder = OrdinalEncoder()
    df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0 )
    df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
    df['region'] = df['region'].apply(lambda x: 3 if x == 'southwest' else 2 if x == 'southeast' else 1 if x == 'northwest' else 0)

    return df




with center_col:
    sex = st.selectbox('Sex/Gender', options = used_df.sex.unique(), key = 'sex')
    smoker = st.selectbox('Smoker', options = used_df.smoker.unique())
    region = st.selectbox('Region', options = used_df.region.unique())
    children_range = sorted(used_df['children'].unique())
    children = st.selectbox('No Of Children', options = children_range)
    bmi_range = sorted(used_df['bmi'].unique())
    bmi = st.select_slider('BMI', options = bmi_range)
    age_range = sorted(used_df['age'].unique())
    age = st.slider('Age', min_value = int(min(age_range)), max_value = int(max(age_range)))
    button = st.button('Get Estimated Charges')

    if button:

        infos = {
            'age': age,
            'sex': st.session_state.sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }


        info_df = pd.DataFrame(infos, index = [1])
        # info_df['region'] = info['region'].apply(lambda: )
        good_df = feauture_engineering(info_df)
        # st.write(info_df)
        estimated_price = model.predict(good_df)
        st.write(f"Estimated Charges To Pay: ${estimated_price[0]: .2f}")


