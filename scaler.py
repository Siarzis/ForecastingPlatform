# Variables scaler
class nwp_scaler():
    
    #Scaling nwp data only for u,v components
    
    def __init__(self):
        
        #Min and Max of u an v values
        self.d_max=25
        self.d_min=-25
        
        return
    
    def scale(self,data):
        
        d_scaled = (data - self.d_min) / (self.d_max - self.d_min)

        return d_scaled
    
    
    def descale(self,d_scaled):
         
        data= d_scaled*(self.d_max - self.d_min) + self.d_min
        
        return data


class p_scaler():
    
    #Scaling power data

    def __init__(self,nominal_p):
        
        self.d_max=nominal_p
        self.d_min=0
        
        return
        
    def scale(self,data):
        
        d_scaled = (data - self.d_min) / (self.d_max - self.d_min)

        return d_scaled
    
    
    def descale(self,d_scaled):
         
        data= d_scaled*(self.d_max - self.d_min) + self.d_min
        
        return data
        
        