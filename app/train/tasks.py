from celery import task, current_task
from time import sleep

from OriginalModel.offline_process import model_train


@task()
def train_task():

    # path_nwp = r'C:\Users\User\WorkSpace\ForecastApp\app\OriginalModel\forecasting_platform\Nwp_2'
    # path_data_train = r'C:\Users\User\WorkSpace\ForecastApp\app\OriginalModel\forecasting_platform\Data\offline\off_data_train.csv'

    # nominal_p = 28000
    # horizon = 48

    # start_date_train = '01/01/2011'
    # end_date_train = '10/30/2011'

    # mdl_train = model_train(path_data_train, path_nwp, nominal_p, horizon, start_date_train, end_date_train)
    # mdl_train.train(h=50, epochs=300)

    for i in range(10):
        sleep(1)

        process_percent = int(100 * float(i) / float(10))
        current_task.update_state(state='PROGRESS',
                                  meta={'process_percent': process_percent})
