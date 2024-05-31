from multiprocessing import Process
from server import app
from job import start_scheduler


def run_flask():
    app.run(port=8000, debug=True)


if __name__ == '__main__':
    p1 = Process(target=run_flask)
    p2 = Process(target=start_scheduler)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
