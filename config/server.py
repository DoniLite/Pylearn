from flask import Flask, jsonify, request

app = Flask(__name__)


@app.get('/')
def home():
    ip_address = request.remote_addr
    print(f" adress: {ip_address}")
    return "Hello, Flask!"


@app.post('/api/data')
def get_data():
    data = {"message": "Hello, this is your data!"}
    return jsonify(data)


@app.post('/api')
def use_api():
    print('ok')


if __name__ == '__main__':
    app.run(port=8000, debug=True)
