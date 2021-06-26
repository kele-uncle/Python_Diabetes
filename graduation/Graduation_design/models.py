from django.db import models


class Login(models.Model):
    user = models.CharField(max_length=50, verbose_name='用户名')
    password = models.CharField(max_length=50, verbose_name='密码')
    email = models.CharField(max_length=100, verbose_name='邮箱')
    class Meta():
        db_table = 'login'
class Wel_Login(models.Model):
    user = models.CharField(max_length=50, verbose_name='普通用户名')
    password = models.CharField(max_length=50, verbose_name='普通密码')
    email = models.EmailField(max_length=100, verbose_name='邮箱')
    class Meta():
        db_table = 'Wel_login'
class UserMessage(models.Model):
    object_id = models.CharField(max_length=50, primary_key=True, verbose_name='主键', default='')
    name = models.CharField(max_length=20, null=True, blank=True, verbose_name=u'用户名')
    email = models.EmailField(verbose_name=u'邮箱')
    address = models.CharField(max_length=50, verbose_name=u'联系地址')
    message = models.CharField(max_length=100, verbose_name=u'留言信息')

    class Meta:
        db_table = 'user_message'
        ordering = ('-object_id',)
        verbose_name_plural = u'用户留言信息'


