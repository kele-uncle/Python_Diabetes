"""定义Graduation_design的URL模式"""

from django.urls import path
from django.conf.urls import url
from . import views

app_name = 'Graduation_design'
urlpatterns = [
    path('', views.Wel_Login,name='wel_login'),
    path('Welpage/', views.Welpage, name='Welpage'),
    path('welreigister/', views.Wel_Register, name="welreigister"),
    path('login/', views.Login, name='login'),
    path('login/hello/register', views.Register, name='Register'),
    path('Introduce/', views.Introduce, name='Introduce'),
    path('Chart/', views.Chart, name='Chart'),
    path('index/', views.index, name='index'),
    path(r'^form/$', views.submitform, name='formUrl'),
    path(r'^showform/$', views.showform, name='showformUrl'),
    path(r'^Submail/$', views.Submail, name='Submail'),
    path(r'^Sendemail/$', views.Sendemail, name='Sendemail'),
    path(r'^Searchdetail/$',views.Searchdetail,name='Searchdetail'),
    path(r'^Doingdetail/$',views.Doingdetail,name='Doingdetail'),
    path(r'^Algorithm/$', views.Algorithm,name='Algorithm'),
    path(r'^KNN/$', views.Alo_Knn, name='knn'),
    path(r'^SVM/$', views.Alo_Svm, name='svm'),
    path(r'^TREE/$', views.Alo_Tree, name='tree'),
    path(r'^MO_TREE/$', views.Alo_more_tree, name='more_tree'),
    path(r'^LINEAR/$', views.Alo_linear, name='linear'),
    path(r'^Data/$', views.Alo_data, name='data'),

]
