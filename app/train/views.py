from django.http import HttpResponse
from django.shortcuts import render
from celery.result import AsyncResult

from train.tasks import train_task
from OriginalModel.online_process import model_forecast

import json


# Create your views here.
def index(request):

    context = {
        'user_name': 'Andrew Syrmakesis',
        'company': 'SmartRue',
        'ship_date': '04/04/2019',
        'util_list': ['train', 'evaluate', 'predict'],
        'ordered_warranty': True
    }

    return render(request, 'train/index.html', context)


def train(request):

    job = train_task.delay()
    return render(request, 'train/train.html', context={'task_id': job.task_id})


# TODO this view can be updated to handle errors
# check https://www.codementor.io/uditagarwal/asynchronous-tasks-using-celery-with-django-du1087f5k
def task_state(request, task_id):

    result = AsyncResult(task_id)
    response_data = {
        'state': result.state,
        'details': result.info
    }
    return HttpResponse(json.dumps(response_data), content_type='application/json')


def predict(request):
    path_nwp = 'C:\\Users\\User\\WorkSpace\\ForecastApp\\app\OriginalModel\\forecasting_platform\\Nwp_2\\'
    nominal_p = 28000
    horizon = 48
    forecasting_date = '12/05/2011'

    mdl_forecast = model_forecast()
    pred = mdl_forecast.forecast(path_nwp, nominal_p, horizon, forecasting_date)

    return render(request, 'train/predict.html', context={'predictions': pred.tolist()})
