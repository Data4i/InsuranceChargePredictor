import pickle

# import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# @st.cache_resource
def load_model(filename: str):
    pickle.load(open(filename, 'rb'))

filename = 'models/insurance_forest_model.pkl'

model = load_model(filename)
print(model)

