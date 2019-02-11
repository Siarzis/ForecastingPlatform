import pandas as pd
from scaler import nwp_scaler

class online_data_processing():
    
    def __init__(self,path_nwp):
        
        self.path_nwp = path_nwp
        
        return
    
    def fetch_data(self,horizon,forecasting_date):
        
        # Reading nwp data for online forecasting
        t = pd.date_range(start=forecasting_date,end=forecasting_date)
        
        file_name=t.strftime('%Y')+'-'+t.strftime('%m')+'-'+t.strftime('%d')+'.csv'

        dataset=pd.read_csv(self.path_nwp+file_name[0],',')
        
        nwp=dataset.iloc[:horizon,:].values
        
        # Scaling nwp in range (0,1)
        sc_nwp= nwp_scaler()
        nwp=sc_nwp.scale(nwp)
        
        return nwp
    
    
    

