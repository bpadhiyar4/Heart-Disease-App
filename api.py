from flask import Flask, jsonify
from project import Learning

app = Flask(__name__)
heartObj = Learning()

@app.route("/getData")
def getData():
    return jsonify(heartObj.getData().tolist())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)