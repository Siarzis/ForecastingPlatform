import numpy as np
import pandas as pd
from scaler import nwp_scaler,p_scaler

class data_processing():
    
    def __init__(self,path_nwp,path_data):
        
        self.path_nwp = path_nwp
        self.path_data = path_data
        
        return

    def train_data(self,nominal_p,horizon,start_date,end_date):

        # Open power timeseries csv
        
        dataset_p=pd.read_csv(self.path_data,',')
        dates=dataset_p.iloc[:,0]

        times_index = pd.date_range(start=start_date,end=end_date)
        
        # Reading power and nwp data
        nwp=[]
        p=[]

        for t in times_index:
            
            try:
                
                file_name=t.strftime('%Y')+'-'+t.strftime('%m')+'-'+t.strftime('%d')+'.csv'
                dataset_temp=pd.read_csv(self.path_nwp+file_name,',')
        
                nwp.append(dataset_temp.iloc[:horizon,:].values)
                time_stamp=t.strftime('%Y-%m-%d 00:00:00+00:00')
    
                ind=np.argmax(dates.str.find(time_stamp))
                
                p.append(dataset_p.iloc[ind:ind+horizon,1:2].values)
                
            except: 
                
                continue
    
        # Scaling data in range (0,1)
        p=np.array(p)
        p=np.reshape(p,(p.shape[0]*p.shape[1],1))

        sc_p=p_scaler(nominal_p)
        p_sc=sc_p.scale(p)

        nwp=np.array(nwp)
        nwp=np.reshape(nwp,(nwp.shape[0]*nwp.shape[1],nwp.shape[2]))

        sc_nwp=nwp_scaler()
        nwp_sc=sc_nwp.scale(nwp)
        
        
        return nwp_sc,p_sc
    
    def evaluation_data(self,nominal_p,horizon,start_date,end_date):

        # Open power timeseries csv

        dataset_p=pd.read_csv(self.path_data,',')
        dates=dataset_p.iloc[:,0]

        times_index = pd.date_range(start=start_date,end=end_date)

        # Reading power and nwp data

        nwp=[]
        p=[]

        for t in times_index:
            
            try:
                file_name=t.strftime('%Y')+'-'+t.strftime('%m')+'-'+t.strftime('%d')+'.csv'
                dataset_temp=pd.read_csv(self.path_nwp+file_name,',')
        
                nwp.append(dataset_temp.iloc[:horizon,:].values)
                time_stamp=t.strftime('%Y-%m-%d 00:00:00+00:00')
    
                ind=np.argmax(dates.str.find(time_stamp))
    
                p.append(dataset_p.iloc[ind:ind+horizon,1:2].values)
            except:
                continue
    

        # Scaling data in range (0,1)

        p=np.array(p)
        p=np.reshape(p,(p.shape[0],p.shape[1],1))

        sc_p=p_scaler(nominal_p)
        p_sc=sc_p.scale(p)
        
        nwp=np.array(nwp)
        nwp_sc=np.reshape(nwp,(nwp.shape[0]*nwp.shape[1],nwp.shape[2]))
        
        sc_nwp=nwp_scaler()
        nwp_sc=sc_nwp.scale(nwp_sc)

        nwp_sc=np.reshape(nwp_sc,(nwp.shape[0],nwp.shape[1],nwp.shape[2]))


        return nwp_sc,p_sc