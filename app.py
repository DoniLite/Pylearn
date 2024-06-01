from multiprocessing import Process
from config.server import app
from jobs.job import start_scheduler


def run_flask():
    app.run(port=8000, debug=True, host='0.0.0.0')


if __name__ == '__main__':
    p1 = Process(target=run_flask)
    p2 = Process(target=start_scheduler)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
