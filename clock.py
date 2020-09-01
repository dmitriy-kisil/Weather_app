from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess
import os

cmd = "python3 weather_flask.py"
print("Run weather app")
weather_app_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
print('Update DB')
# cmd = "python3 create_new_day3.py"
# subprocess.call(cmd, shell=True)
sched = BlockingScheduler(timezone="Europe/Kiev")

# For testing purpose
# @sched.scheduled_job('interval', minutes=1)
# def timed_job():
#     """Schedule a job."""
#     print('This job is run every 1 minute.')
#     cmd = "python3 create_new_day2.py"
#     subprocess.call(cmd, shell=True)


@sched.scheduled_job('interval', hours=1)
def hourly_job():
    """Schedule a job."""
    print('This job is run every 1 hour.')
    cmd = "python3 update_hour_temp.py"
    subprocess.call(cmd, shell=True)


@sched.scheduled_job('cron', day_of_week='mon-sun', hour=12, minute=00)
def scheduled_job():
    """Schedule a job."""
    print('This job is run every one day.')
    cmd = "python3 create_new_day3.py"
    subprocess.call(cmd, shell=True)


sched.start()
