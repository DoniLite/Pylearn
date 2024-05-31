import schedule
import time
from datetime import datetime


def job():
    now = datetime.now()
    print("Date actuelle :", now.strftime("%Y-%m-%d"))
    print("Heure actuelle :", now.strftime("%H:%M:%S"))


def start_scheduler():
    schedule.every(10).seconds.do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)
