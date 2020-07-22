from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('',views.homePage,name='home'),
    path('about',views.aboutPage,name='about'),
    path('contact',views.contactPage,name='contact'),
    path('face',views.facePage,name='face'),
    path('language',views.languagePage,name='language'),
    path('project',views.projectPage,name='project'),
    path('review',views.reviewPage,name='review'),
    path('smile',views.smilePage,name='smile'),
    path('thug',views.thugPage,name='thug'),
    path('spam',views.spamPage, name='spam'),
    path('thugrun',views.thugrun,name='thugrun'),
    path('lang',views.langrun,name = 'langrun'),
    path('rev',views.reviewrun,name='reviewrun'),
    path('spa',views.spamrun,name='spamrun'),
    path('facerun', views.facerun, name='facerun'),
    path('smilerun', views.smilerun, name='smilerun'),
    path('blog',views.blogPage, name='blog'),
    path('con',views.contectmail, name='conmail'),

]
