# Generated by Django 3.1.7 on 2021-03-27 03:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Graduation_design', '0006_guestbook'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserMessage',
            fields=[
                ('object_id', models.CharField(default='', max_length=50, primary_key=True, serialize=False, verbose_name='主键')),
                ('name', models.CharField(blank=True, max_length=20, null=True, verbose_name='用户名')),
                ('email', models.EmailField(max_length=254, verbose_name='邮箱')),
                ('address', models.CharField(max_length=50, verbose_name='联系地址')),
                ('message', models.CharField(max_length=100, verbose_name='留言信息')),
            ],
            options={
                'verbose_name_plural': '用户留言信息',
                'db_table': 'user_message',
                'ordering': ('-object_id',),
            },
        ),
        migrations.DeleteModel(
            name='GuestBook',
        ),
    ]