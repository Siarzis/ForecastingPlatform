import tensorflow as tf
import numpy as np
from data_processing import data_processing
import pandas as pd

# Training model
class model_train():
    
    def __init__(self,path_data,path_nwp,nominal_p,horizon,start_date,end_date):
        
        # Retrieving training data
        data_model = data_processing(path_nwp,path_data)
        self.x_train,self.y_train = data_model.train_data(nominal_p,horizon,start_date,end_date)
        
        # User parameters
        self.nominal_p = nominal_p
        self.horizon = horizon
        
        return
    
    def train(self,h,epochs):
        
        # Building NN model
        
        x_train = self.x_train 
        y_train = self.y_train 
        
        
        sess=tf.Session()
        
        ## Defining models input layer parameters
        
        x=tf.placeholder(dtype=tf.float32,shape=(None,x_train.shape[1]))
        y=tf.placeholder(dtype=tf.float32,shape=(None,y_train.shape[1]))

        ## Defining models hidden layer parameters
        
        w_in=tf.Variable(initial_value=np.random.randn(x_train.shape[1],h),dtype=tf.float32)
        w_out=tf.Variable(initial_value=np.random.randn(h,y_train.shape[1]),dtype=tf.float32)
    
        b_in=tf.Variable(initial_value=np.random.randn(1,h),dtype=tf.float32)
        b_out=tf.Variable(initial_value=np.random.randn(1,y_train.shape[1]),dtype=tf.float32)
        
        
        hl=tf.nn.sigmoid(tf.matmul(x,w_in)+b_in)
        hl_drpt=tf.nn.dropout(hl,0.5)
        
        ## Defining models output layer parameters
        y_hat =tf.nn.sigmoid((tf.matmul(hl_drpt,w_out)+b_out))
            
        squared_delta=tf.squared_difference(y_hat,y)

        loss=tf.reduce_mean(squared_delta)

        optimizer=tf.train.AdamOptimizer(0.001)
        train=optimizer.minimize(loss)
        
        # Running training
        init=tf.global_variables_initializer()
        sess.run(init)
    
        for i in range(epochs):  
                if (sess.run(loss,{x:x_train,y:y_train})>0):
                    sess.run(train,{x:x_train,y:y_train})
                    print('Epoch : '+str(i+1)+' / '+str(epochs)+' loss = ',sess.run(loss,{x:x_train,y:y_train}))
                else:
                    break
        
        # Storing model parameters to csv
        self.win = sess.run(w_in)
        self.wout = sess.run(w_out)
        self.bin = sess.run(b_in)
        self.bout = sess.run(b_out)
        
        win_df=pd.DataFrame(self.win, dtype=None, copy=False)
        win_df.to_csv('G:\\phd\\forecasting_platform\\Parameters\\w_in.csv') 
    
        wout_df=pd.DataFrame(self.wout, dtype=None, copy=False)
        wout_df.to_csv('G:\\phd\\forecasting_platform\\Parameters\\w_out.csv') 
        
        bin_df=pd.DataFrame(self.bin, dtype=None, copy=False)
        bin_df.to_csv('G:\\phd\\forecasting_platform\\Parameters\\b_in.csv') 
    
        bout_df=pd.DataFrame(self.bout, dtype=None, copy=False)
        bout_df.to_csv('G:\\phd\\forecasting_platform\\Parameters\\b_out.csv') 
        
        return 

class model_evaluation():

    def __init__(self,path_data,path_nwp,nominal_p,horizon,start_date,end_date):
        
        # Retrieving evaluation data
        data_model = data_processing(path_nwp,path_data)
        self.x_test,self.y_test = data_model.evaluation_data(nominal_p,horizon,start_date,end_date)
        
        # User parameters

        self.nominal_p = nominal_p
        self.horizon = horizon
        
        return 
        
    def evaluate(self):
        
        
        x_test = self.x_test 
        y_test = self.y_test 
        
        # Retrieving model parameters
        
        w_in_par=pd.read_csv('G:\\phd\\forecasting_platform\\Parameters\\w_in.csv').iloc[:,1:].values
        w_out_par=pd.read_csv('G:\\phd\\forecasting_platform\\Parameters\\w_out.csv').iloc[:,1:].values
        b_in_par=pd.read_csv('G:\\phd\\forecasting_platform\\Parameters\\b_in.csv').iloc[:,1:].values
        b_out_par=pd.read_csv('G:\\phd\\forecasting_platform\\Parameters\\b_out.csv').iloc[:,1:].values
        
        # Building NN model using the parameters
        
        sess=tf.Session()

        ## Defining models input layer parameters
        x=tf.placeholder(dtype=tf.float32,shape=(None,x_test.shape[2]))

        ## Defining models hidden layer parameters

        w_in=tf.Variable(initial_value=w_in_par,dtype=tf.float32)
        w_out=tf.Variable(initial_value=w_out_par,dtype=tf.float32)
    
        b_in=tf.Variable(initial_value=b_in_par,dtype=tf.float32)
        b_out=tf.Variable(initial_value=b_out_par,dtype=tf.float32)
    
        hl=tf.nn.sigmoid(tf.matmul(x,w_in)+b_in)
        
        ## Defining models output layer parameters
        
        y_hat =tf.nn.sigmoid((tf.matmul(hl,w_out)+b_out))
    
        init=tf.global_variables_initializer()
        sess.run(init)
        
        # Making prediction using the model
        
        pred=[]
        
        for i in range(x_test.shape[0]):
            pred.append(sess.run(y_hat,{x:x_test[i]}))
            
        pred=np.array(pred)
        
        # Calculating mean absolute error
        
        mae=[]
        for i in range(self.horizon):
            mae.append(np.mean(np.abs(y_test[:,i]-pred[:,i])))
        
        return mae
