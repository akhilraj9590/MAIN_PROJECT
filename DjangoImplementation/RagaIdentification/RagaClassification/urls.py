from django.urls import path
from .views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', home,name='home'),
    path('uploadAudio', uploadAudio,name='uploadAudio'),
    path('predictByRecord',predictByRecord,name='predictByRecord'),

]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
