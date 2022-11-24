import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd

y_predict = np.load('./y_predicted_42.npy')
y_test = np.load('./y_test_42.npy')

a = []
for i in range(100):
    b = []
    b.append(y_predict[i][0])
    b.append(y_test[i][0])
    a.append(b)

a_np = np.array(a)

image = Image.open('out.png')


st.title('Blood Pressure Estimation using Neural Network 🩸')

code = '''# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(ppg, bp, test_size=0.30)'''
st.code(code, language='python')

code = '''# Import Libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers'''
st.code(code, language='python')

code = ''' # Create Deep Learning Model
def Model(input_dim, activation, num_class):
    model = Sequential()

    model.add(Dense(1024, input_dim = input_dim))
    model.add(Activation(activation))
    model.add(Dropout(0.5))

    model.add(Dense(512)) 
    model.add(Activation(activation))
    model.add(Dropout(0.5))

    model.add(Dense(64))    
    model.add(Activation(activation))
    model.add(Dropout(0.25))

    model.add(Dense(num_class))    
    model.add(Activation('linear'))
    
    model.compile(loss='Huber',
                  optimizer=optimizers.Adam(lr = 0.001),
                  metrics=['MeanAbsoluteError']
                 )
    return model'''
st.code(code, language='python')

st.image(image, caption='Mean Absolute Error of First 10,000 Sample')

chart_data = pd.DataFrame(
    a_np,
    columns=['Predicted Blood Pressure', 'Actual Blood Pressure'])


st.line_chart(chart_data)