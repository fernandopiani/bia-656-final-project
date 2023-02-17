# packages
import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMRegressor
from flask import Flask, request, render_template
from model_transformer import ModelTransformer

# Sklearn imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin

import __main__
__main__.ModelTransformer = ModelTransformer


# List translation function
def translate(inputs):

    # Translation dictionaries
    property_type_dict = {'0': 'Aparthotel', '1': 'Apartment', 
                            '2': 'Bed and breakfast', '3': 'Boutique hotel', \
                                '4': 'Condominium', '5': 'Guest suite', \
                                    '6': 'Guesthouse', '7': 'Hostel',
                                        '8': 'Hotel', '9': 'House',\
                                            '10': 'Loft', '11': 'Other',\
                                                '12': 'Serviced apartment', '13': 'Townhouse'}

    room_type_dict = {'0':'Entire home/apt', '1': 'Private room', '2':'Shared room', '3': 'Hotel room'}

    input_df = pd.DataFrame(inputs, index=[0])
    input_df['property_type'] = input_df['property_type'].replace(property_type_dict)
    input_df['room_type'] = input_df['room_type'].replace(room_type_dict)
    input_df['id'] = 89

    # Creating dummy ID column as expected by ColumnTransformer
    first_column = input_df.pop('id')
    input_df.insert(0, 'id', first_column)

    return input_df


# Preprocessing function:
def preprocess(input_df):

    import model_transformer

    input_df['bedrooms'] = input_df['bedrooms'].astype('float')
    input_df['bathrooms'] = input_df['bathrooms'].astype('float')
    input_df['minimum_nights'] = input_df['minimum_nights'].astype('float')

    with open('pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    
    input_df_transformed = pipeline.transform(input_df)
    return input_df_transformed

    

# prediction function
def predict(inputs_transformed_df):

    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    result = loaded_model.predict(inputs_transformed_df)
    return result[0]

app = Flask(__name__) 

@app.route('/')
def index():
    return render_template('interface.html')
 
@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        inputs = request.form.to_dict()
        input_df = translate(inputs)
        input_df_transformed = preprocess(input_df)
        prediction = predict(input_df_transformed)
        prediction = np.round(prediction, 2)       
        return render_template("output.html", prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True)