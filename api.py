from flask import Flask, jsonify
from flask import request
from flask_marshmallow import Marshmallow
from flask_migrate import Migrate
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from pandas import to_datetime as td

from .predict import predict

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///airsynq.db'

socketio = SocketIO(app)
db = SQLAlchemy(app)
ma = Marshmallow(app)
migrate = Migrate(app, db)


class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime())
    prediction = db.Column(db.String())

    def __init__(self, timestamp, prediction):
        self.timestamp = timestamp
        self.prediction = prediction


class PredictionSchema(ma.Schema):
    class Meta:
        # Fields to expose
        fields = ('timestamp', 'prediction')


predictions_schema = PredictionSchema(many=True)


@app.route('/v1.0/loki', methods=["POST"])
def realtime_prediction():
    if request.method == 'POST':
        timestamp = td(request.form['timestamp'])
        image = request.files['image']

        prediction = predict(image).tolist()[0]

        p = Prediction(timestamp=timestamp, prediction=prediction)

        db.session.add(p)
        db.session.commit()
        db.session.close()

        @socketio.on('predictions')
        def get_predictions():
            all_predictions = Prediction.query.all()
            result = predictions_schema.dump(all_predictions)

            emit('response', jsonify({"prediction": result, "timestamp": timestamp}), broadcast=True)

        return jsonify({"message": "Prediction Successful"}), 201


db.create_all()

if __name__ == "__main__":
    # specify thread as false b/c of tf compatibility while running flask: flask run --without-threads
    socketio.run(app)
