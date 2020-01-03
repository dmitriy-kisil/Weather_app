from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess
import signal
import os

cmd = "python3 weather_flask.py"
print("Run weather app")
weather_app_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)

sched = BlockingScheduler(timezone="Europe/Kiev")

# For testing purpose
# @sched.scheduled_job('interval', minutes=1)
# def timed_job():
#     """Schedule a job."""
#     print('This job is run every 1 minute.')
#     cmd = "python3 create_new_day.py"
#     subprocess.call(cmd, shell=True)

@sched.scheduled_job('cron', day_of_week='mon-sun', hour=18, minute=40)
def scheduled_job():
    """Schedule a job."""
    print('This job is run every one day.')
    cmd = "python3 create_new_day.py"
    subprocess.call(cmd, shell=True)

sched.start()
