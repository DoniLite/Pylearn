import schedule
import psycopg2
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello, Flask!"


@app.route('/api/data')
def get_data():
    data = {"message": "Hello, this is your data!"}
    return jsonify(data)


if __name__ == '__main__':
    app.run(port=8000, debug=True)


def job(conf_job):
    now = datetime.now()
    print("Date actuelle :", now.strftime("%Y-%m-%d"))
    print("Heure actuelle :", now.strftime("%H:%M:%S"))
    conf_job()


schedule.every(10).seconds.do(job)

# Connect to your postgres DB
conn = psycopg2.connect("""postgres://doni_data_user:yV287EL5Ju5ESwxts8th7A0EPUQfv0Dn@dpg-cpc1pbm3e1ms739dhttg-a
.oregon-postgres.render.com/doni_data""")


cur = conn.cursor()

# Execute a query
# cur.execute("SELECT * FROM my_data")

# Retrieve query results
# records = cur.fetchall()
# print(records)