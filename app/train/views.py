from django.http import HttpResponse
from django.shortcuts import render

from OriginalModel.offline_process import model_train


# Create your views here.
def index(request):

    context = {
        'user_name': 'Andrew Syrmakesis',
        'company': 'SmartRue',
        'ship_date': '04/04/2019',
        'util_list': ['train', 'evaluate', 'test'],
        'ordered_warranty': True
    }

    return render(request, 'train/index.html', context)


def train(request):

    path_nwp = r'C:\Users\User\WorkSpace\ForecastApp\app\OriginalModel\forecasting_platform\Nwp_2'
    path_data_train = r'C:\Users\User\WorkSpace\ForecastApp\app\OriginalModel\forecasting_platform\Data\offline\off_data_train.csv'

    nominal_p = 28000
    horizon = 48

    start_date_train = '01/01/2011'
    end_date_train = '10/30/2011'

    h = 50
    epochs = 50

    mdl_train = model_train()
    win_df, wout_df, bin_df, bout_df = mdl_train.delay(path_data_train, path_nwp, nominal_p, horizon, start_date_train, end_date_train, h, epochs)

    context = {
    }

    return render(request, 'train/train.html', context)
