# Generated by Django 5.0.7 on 2024-10-08 12:24

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("SchedulerApp", "0007_instructor_availability_end_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="meetingtime",
            name="time",
            field=models.TimeField(
                choices=[
                    ("6:45 - 8:20", "6:45 - 8:20"),
                    ("8:20 - 10:00", "8:20 - 10:00"),
                    ("10:00 - 10:50", "10:00 - 10:50"),
                    ("10:50 - 12:30", "10:50 - 12:30"),
                    ("12:30 - 2:10", "12:30 - 2:10"),
                ],
                default="11:30 - 12:30",
                max_length=50,
            ),
        ),
    ]
