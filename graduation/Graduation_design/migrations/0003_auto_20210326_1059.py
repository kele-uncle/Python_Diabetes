# Generated by Django 3.1.7 on 2021-03-26 02:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Graduation_design', '0002_wel_login'),
    ]

    operations = [
        migrations.AlterField(
            model_name='wel_login',
            name='email',
            field=models.CharField(max_length=30, verbose_name='邮箱'),
        ),
    ]
