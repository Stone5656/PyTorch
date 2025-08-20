from django.urls import path
from .views import regress_predict, classify_predict

urlpatterns = [
    path("regress/predict", regress_predict),
    path("classify/predict", classify_predict),
]
