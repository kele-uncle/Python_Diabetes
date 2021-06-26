from django.contrib.auth.decorators import login_required
from warnings import simplefilter
from django.shortcuts import render, redirect
from .models import Login, UserMessage
import hashlib
from . import models
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.dates import DateFormatter
import numpy as np
from io import BytesIO
import base64
from numpy.random import randn
from django.http import HttpResponse
import datetime
import pandas as pd
import json
import random
from random import choice
import string
from django.core.mail import send_mail
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup
import requests
from django.contrib.auth.decorators import login_required
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
simplefilter(action='ignore', category=FutureWarning)

reloadModel = joblib.load('D:\Programming\Graduation\Graduation\graduation\static\\file\ModelDiabetes.pkl')
global diabetes
diabetes = pd.read_csv('D:\Programming\Graduation\Graduation\graduation\static\\file\diabetes.csv')
x_train, x_test, y_train, y_test = train_test_split(
        diabetes.loc[:, diabetes.columns != 'Outcome'],
        diabetes['Outcome'], stratify=diabetes['Outcome'],
        random_state=66) # 对数据进行分隔，随机分为训练子集和测试子集
diabetes_features = [x for i, x in enumerate(diabetes.columns) if i != 8]


def Login(request):
    if request.method == "POST":
        username = request.POST.get('name')
        password = request.POST.get('password')
        if username and password:  # 确保用户名和密码都不为空
            username = username.strip()
            try:
                user = models.Login.objects.get(user=username)  # 看前面创建的user表中是否有name值符合登录页面输入的用户名
            except:
                return render(request, 'Login.html')  # 数据库中没有相应的用户名，跳转至注册页面
            if user.password == password:
                return render(request, 'Hello.html')

    return render(request, 'Login.html')
def Register(request):
    if request.method == 'GET':
        return HttpResponse('no')
    elif request.method == 'POST':
        username = request.POST.get('name')
        repassword = request.POST.get('repassword')
        user = models.Login.objects.filter(user=username).first()
        if user:
            return HttpResponse('chunzai')
        try:
            user = models.Login(user=username,  password=repassword,)
            user.save()
            return HttpResponse('ok')
        except Exception as e:
            print('保存失败', e)
            return HttpResponse(e)
def Introduce(request):
    return render(request, 'Introduce.html')
def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)
def Chart(request):
    t1 = np.arange(0, 5, 0.1)
    t2 = np.arange(0, 5, 0.02)
    plt.figure(12)
    plt.subplot(221)
    plt.plot(t1, f(t1), 'bo', t2, f(t2), 'r--')
    plt.subplot(222)
    plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
    plt.subplot(212)
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)  # 对plot_data进行编码
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    context = {
        'img': imd,

    }
    return render(request, 'chart.html', context)
def Wel_Login(request):
    if request.method == "POST":
        login_value = request.POST.get('name')
        password = request.POST.get('password')
        if login_value and password:  # 确保用户名和密码都不为空
            login_value = login_value.strip()
            try:
                user = models.Wel_Login.objects.get(user=login_value)# 看前面创建的user表中是否有name值符合登录页面输入的用户名
            except:
                #context = {"error": error}
                #return render(request, 'Welcome.html')
                return HttpResponse('登录错误')# 数据库中没有相应的用户名，跳转至注册页面
            if user.password == password:
                return render(request, 'base.html')

    return render(request, 'welcome.html')

def Welpage(request):
    return render(request, 'welreigister.html', )
def Wel_Register(request):
    if request.method == 'GET':
        return render(request, 'welcome.html')
    elif request.method == 'POST':
        username = request.POST.get('name')
        repassword = request.POST.get('repassword')
        email = request.POST.get('email')
        user = models.Wel_Login.objects.filter(user=username).first()
        if user:
            context = {'error': '用户名已存在' }
            return render(request, 'welreigister.html', context)
        try:
            user = models.Wel_Login(user=username,  password=repassword, email=email)
            user.save()
            context = {'error': '注册成功,跳转登录页面'}
            return render(request, 'welcome.html', context)
        except Exception as e:
            print('保存失败', e)
            return HttpResponse(e)
'''
获取留言表单，填入留言信息并存储到数据库中
'''
def submitform(request):
    return render(request,'message_form.html')
'''
显示留言板信息
'''
def showform(request):
    # 上传到数据库
    if request.method == 'POST':
        usermessage = UserMessage()
        usermessage.name = request.POST.get('name','')
        usermessage.email = request.POST.get('email','')
        usermessage.address = request.POST.get('address','')
        usermessage.message = request.POST.get('message','')
        usermessage.object_id = str(random.randint(0,10000)).zfill(5) # 自动补零
        usermessage.save()
    # 取出当前数据库中所有记录，并传入到下一个html中
    all_messages = UserMessage.objects.all()
    return render(request,'message_board.html', {
        'all_messages':all_messages,
    })
def Submail(request):
    return render(request,'findpassword.html')
'''使用邮箱实现重置密码'''
def Sendemail(request):
    email = request.POST.get('email')
    if models.Login.objects.filter(email=email):
        for i in models.Login.objects.filter(email=email):
            nametemp = i.user
            idtemp = i.id
            # 生成随机密码
            def GenPassword(length=8, chars=string.ascii_letters + string.digits):
                return ''.join([choice(chars) for i in range(length)])
            pawdtemp = GenPassword(8)
            models.Login.objects.filter(email=email).delete()
            models.Login.objects.create(id=idtemp, user=nametemp, password=pawdtemp, email=email)
            send_mail(
                subject=u"这是新的密码,请使用新的密码登录", message=pawdtemp,
                from_email='daigang344@163.com', recipient_list=[email, ], fail_silently=False,
            )
            context = {'message':'重置成功，请查看邮箱登录'}
            return render(request,'Welcome.html', context)
    else:
        return HttpResponse("您的邮箱的账户注册信息没有找到")
def Searchcity(request):
    return render(request,'consult.html')
def Searchdetail(request):
    return render(request, 'searchdetail.html')
def Doingdetail(request):
    if request.method == 'POST':
        temp = {}
        arr = []
        temp['preg'] = request.POST.get('pregVal')
        temp['plasma'] = request.POST.get('plasmaVal')
        temp['bp'] = request.POST.get('bpVal')
        temp['skin'] = request.POST.get('skinVal')
        temp['insulin'] = request.POST.get('insulinVal')
        temp['bmi'] = request.POST.get('bmiVal')
        temp['pedigree'] = request.POST.get('pedigreeVal')
        temp['age'] = request.POST.get('ageVal')
        arr.append(float(temp['preg']))
        arr.append(float(temp['plasma']))
        arr.append(float(temp['bp']))
        arr.append(float(temp['skin']))
        arr.append(float(temp['insulin']))
        arr.append(float(temp['bmi']))
        arr.append(float(temp['pedigree']))
        arr.append(float(temp['age']))
        scoreval = reloadModel.predict([arr]) # 这里的意思是给定一个糖尿病数据reloadModel，预测获取的数据在这个数据集中的标签。实现对是否患有糖尿病进行一个预测。
        return render(request,'Searchresult.html',{'scoreval':scoreval})
def DocDetail(request):
    if request.method == 'POST':
        temp={}
        temp['city'] = request.POST.get('city')
        city = temp['city']
        error = False
        data = []
        try:
            url = 'https://www.practo.com/'+city+'/endocrinologist'
            html_text = requests.get(url).text
            soup = BeautifulSoup(html_text,'lxml')
            names = soup.find_all('h2',class_='doctor-name')
            places = soup.find_all('div',class_='u-bold u-d-inlineblock u-valign--middle')
            for i in range(len(names)):
                val = {'name':names[i].text,'place':places[i].text}
                data.append(val)
        except:
            error = True
        return render(request,'docdisplay.html', {'data':data})
@login_required
def index(request):
    return render(request, 'base.html')
def Algorithm(request):
    return render(request, 'Alo_base.html')

def Alo_Knn(request):
    # k_nn = request.POST.get('knn')
    # begin_value = request.POST.get('begin')
    # end_value = request.POST.get('end')
    # train_accuracy = []
    # test_accuracy = []
    # # neighbors = range(1,11)
    # # for neighbor in neighbors:
    # #     k_value = KNeighborsClassifier(n_neighbors=neighbor)  # 建立KNN模型
    # #     k_value.fit(x_train, y_train)  # 使用x_train作为训练数据，y_train作为目标值
    # #     train_value = k_value.score(x_train, y_train)
    # #     test_value = k_value.score(x_test, y_test)
    # #     train_accuracy.append(train_value)
    # #     test_accuracy.append(test_value)
    #
    # k_value = KNeighborsClassifier(n_neighbors=k_nn)  # 建立KNN模型
    # k_value.fit(x_train, y_train)  # 使用x_train作为训练数据，y_train作为目标值
    # train_value = k_value.score(x_train, y_train)
    # test_value = k_value.score(x_test, y_test)
    # context = {'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy,'train_value': train_value,'test_value': test_value}

    list = ['hello','nihao','haha','nihao','hola','coome','here']
    return render(request,'Alo_Knn.html', {'list':json.dumps(list), })


def Alo_Svm(request):
    return render(request,'Alo_Svm.html')
def Alo_Tree(request):
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(x_train, y_train)
    important = tree.feature_importances_
    plt.figure(figsize=(6, 4))
    n_features = 8
    plt.barh(range(n_features), important, align='center')
    plt.yticks(np.arange(n_features), diabetes_features)
    plt.xlabel("功能的重要程度")
    plt.ylabel("功能")
    plt.ylim(-1, n_features)
    tree_io = BytesIO()
    plt.savefig(tree_io, format='png', bbox_inches='tight', pad_inches=0.0)
    tree_data = base64.encodebytes(tree_io.getvalue()).decode()
    tree_img = 'data:image/png;base64,' + str(tree_data)
    plt.close()
    context = {'important': important,'tree_img':tree_img}
    return render(request,'Alo_Tree.html',context)
def Alo_more_tree(request):
    more_tr = RandomForestClassifier(n_estimators=100, random_state=0)
    more_tr.fit(x_train,y_train)
    plt.figure(figsize=(6, 4))
    n_features = 8
    plt.barh(range(n_features), more_tr.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), diabetes_features)
    plt.xlabel("功能的重要程度")
    plt.ylabel("功能")
    plt.ylim(-1, n_features)
    more_tr_io = BytesIO()
    plt.savefig(more_tr_io, format='png', bbox_inches='tight', pad_inches=0.0)
    more_tr_data = base64.encodebytes(more_tr_io.getvalue()).decode()
    more_tr_img = 'data:image/png;base64,' + str(more_tr_data)
    plt.close()
    context = {'more_tr': more_tr_img}
    return render(request,'Alo_more_tree.html',context)
def Alo_linear(request):
        return render(request,'Alo_linear.html')
def Alo_data(request):
    return  render(request,'Alo_data.html')
