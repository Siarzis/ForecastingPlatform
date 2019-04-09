from django.http import HttpResponse
from django.shortcuts import render
from celery.result import AsyncResult

from train.tasks import train_task

import json
from time import sleep


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

    job = train_task.delay()
    result = AsyncResult(job.id)
    sleep(5)
    response_data = {
        'state': result.state,
        'details': result.info,
    }

    return HttpResponse(json.dumps(response_data), content_type='application/json')

    # context = {}

    # return render(request, 'train/train.html', context)
