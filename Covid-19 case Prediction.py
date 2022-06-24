# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:48:43 2022

@author: caron
"""

#%% Imports
import os
import pandas as pd
from Covid19_module import EDA
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from Covid19_module import model_evaluation
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
#%% Constant

CSV_PATH = os.path.join(os.getcwd(),('cases_malaysia_train.csv'))



#%% EDA STEPS

# Step 1) Data Loading

df=pd.read_csv(CSV_PATH)

# Step 2) Data inspection
df.info()
df.describe().T


eda=EDA()
eda.plot_graph(df)



# Step 3) Data Cleaning
df['cases_new'] = pd.to_numeric(df['cases_new'],errors='coerce')
df.info()
df.isna().sum()
df['cases_new'].interpolate(method='polynomial', order=2,inplace=True)

temp = df['cases_new']
df.isna().sum()
# Step 4) Features Selectiion
# selecting cases_new data only

#%% Step 5)  Data Preprocessing
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
df = mms.fit_transform(np.expand_dims(df['cases_new'],axis=-1))

# save pickle
import pickle

MSS_FILE_NAME = os.path.join(os.getcwd(),'covid_cases_scalar.pkl')  
with open(MSS_FILE_NAME,'wb') as file:
    pickle.dump(mms,file)
    
    
X_train=[]
y_train=[]

win_size = 30

for i in range(win_size,np.shape(df)[0]):
    X_train.append(df[i-win_size:i,0])
    y_train.append(df[i,0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)


#%% model development
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
import numpy as np

model = Sequential()
model.add(Input(shape=(np.shape(X_train)[1],1))) # input_length, # number of features
model.add(LSTM(64,return_sequences=((True)))) # LSTM
model.add(Dropout(0.3))
model.add(LSTM(64)) # LSTM
model.add(Dropout(0.3))
model.add(Dense(1,activation='relu')) # Output Layer
model.summary()

model.compile(optimizer='adam',loss='mse',metrics='mape')

# callbacks
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import datetime
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
COVID_LOG_FOLDER_PATH = os.path.join(os.getcwd(),'covid_log',log_dir)

tensorboard_callback = TensorBoard(log_dir=COVID_LOG_FOLDER_PATH)

early_stopping_callback = EarlyStopping(monitor='loss',patience=3)

X_train = np.expand_dims(X_train,axis=-1)
hist = model.fit(X_train,y_train,batch_size=32,epochs=100,
                 callbacks=[tensorboard_callback,early_stopping_callback])

#%% TensorBoard

plot_model(model,show_layer_names=(True),show_shapes=(True))
#%% Model Evaluation

hist.history.keys()
plt.figure()
plt.plot(hist.history['mape'])
plt.show()

plt.figure()
plt.plot(hist.history['loss'])
plt.show()

#%% model deployment and analysis
CSV_TEST_PATH = os.path.join(os.getcwd(),'cases_malaysia_test.csv')

test_df = pd.read_csv(CSV_TEST_PATH)
test_df['cases_new']=pd.to_numeric(test_df['cases_new'],errors='coerce')
test_df.info() # got 1 Nans

# use interpolate for NaNs value
test_df['cases_new'].interpolate(method='polynomial',order=2,inplace=True) # to fill NaN for timeseries data
test_df.isna().sum() # 0 Nans


test_df = mms.transform(np.expand_dims(test_df.iloc[:,1],axis=-1)) 
con_test = np.concatenate((df,test_df),axis=0)
con_test = con_test[-(win_size+len(test_df)):] 

plt.figure()
plt.plot(test_df)
plt.show()

X_test = []

for i in range(win_size,len(con_test)): 
    X_test.append(con_test[i-win_size:i,0])
    
X_test = np.array(X_test)

predicted = model.predict(np.expand_dims(X_test,axis=-1))

# save the model
#%% plotting the graphs

me = model_evaluation()
me.plot_predicted_graph(test_df, predicted, mms)

#%% MSE, MAPE
print(mean_absolute_error(test_df, predicted))
print(mean_squared_error(test_df, predicted))

test_df_inverse = mms.inverse_transform(test_df)
predicted_inverse = mms.inverse_transform(predicted)

print(mean_absolute_error(test_df_inverse,predicted_inverse))
print(mean_squared_error(test_df_inverse,predicted_inverse))
print(mean_absolute_percentage_error(test_df_inverse,predicted_inverse))
print((mean_absolute_error(test_df, predicted)/sum(abs(test_df))) *100)

#%% Model saving
import os
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
model.save(MODEL_SAVE_PATH)
