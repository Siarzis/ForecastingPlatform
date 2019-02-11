import tensorflow as tf
from online_data import online_data_processing
import pandas as pd
from scaler import p_scaler

class model_forecast():
    
    def __init__(self):
        
        return
    
    def forecast(self,path_nwp,nominal_p,horizon,forecasting_date):
        
        # Retrieving nwp data for online forecasting
        
        data_model = online_data_processing(path_nwp)
        x = data_model.fetch_data(horizon,forecasting_date)
        
        # Retrieving model parameters        
        
        w_in_par=pd.read_csv('G:\\phd\\forecasting_platform\\Parameters\\w_in.csv').iloc[:,1:].values
        w_out_par=pd.read_csv('G:\\phd\\forecasting_platform\\Parameters\\w_out.csv').iloc[:,1:].values
        b_in_par=pd.read_csv('G:\\phd\\forecasting_platform\\Parameters\\b_in.csv').iloc[:,1:].values
        b_out_par=pd.read_csv('G:\\phd\\forecasting_platform\\Parameters\\b_out.csv').iloc[:,1:].values
    
        # Building NN model
        sess=tf.Session()
        x_in =tf.placeholder(dtype=tf.float32,shape=(None,x.shape[1]))
        
        w_in=tf.Variable(initial_value=w_in_par,dtype=tf.float32)
        w_out=tf.Variable(initial_value=w_out_par,dtype=tf.float32)
    
        b_in=tf.Variable(initial_value=b_in_par,dtype=tf.float32)
        b_out=tf.Variable(initial_value=b_out_par,dtype=tf.float32)
    
        hl=tf.nn.sigmoid(tf.matmul(x_in,w_in)+b_in)
        
        y_hat =tf.nn.sigmoid((tf.matmul(hl,w_out)+b_out))
    
        init=tf.global_variables_initializer()
        sess.run(init)
        
        # Making the prediction
        
        pred=sess.run(y_hat,{x_in:x})
        
        # Descaling prediction
        
        sc_p=p_scaler(nominal_p)
        pred=sc_p.descale(pred)
        
        return pred
    
    
