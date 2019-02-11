import pandas as pd
from offline_process import model_train,model_evaluation
import numpy as np
from online_process import model_forecast


# Input parameters

path_nwp="G:\\phd\\forecasting_platform\\Nwp_2\\"
path_data_train="G:\\phd\\forecasting_platform\\Data\\offline\\off_data_train.csv"
path_data_evaluation="G:\\phd\\forecasting_platform\\Data\\offline\\off_data_evaluation.csv"

nominal_p=28000
horizon=48


# Training

start_date_train='01/01/2011'
end_date_train='10/30/2011'

mdl_train=model_train(path_data_train,path_nwp,nominal_p,horizon,start_date_train,end_date_train)
mdl_train.train(h=50,epochs=300)


# Evaluation

start_date_eval='11/01/2011'
end_date_eval='11/30/2011'

mae_h=model_evaluation(path_data_evaluation,path_nwp,nominal_p,horizon,start_date_eval,end_date_eval).evaluate()

for i in range(horizon):
    print('Mean absolute error for hour '+str(i+1)+' = ',mae_h[i])

# Forecasting

forecasting_date='12/05/2011'

mdl_forecast=model_forecast()

pred=mdl_forecast.forecast(path_nwp,nominal_p,horizon,forecasting_date)
print(pred)
