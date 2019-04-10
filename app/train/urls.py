from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('train/', views.train, name='train'),
    path('train/<task_id>', views.task_state, name='train_state')
]
