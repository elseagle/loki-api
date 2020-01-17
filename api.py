from flask import Flask, jsonify
from flask import request
from predict import predict
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = ''

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


@socketio.on('predict')
def realtime_prediction(timestamp, image):

    timestamp = request.form['timestamp']
    image = request.files['image']

    prediction = predict(image).tolist()[0]

    p = Prediction(timestamp=timestamp, prediction=prediction)

    db.session.add(p)
    db.session.commit()
    db.session.close()

    all_predictions = Prediction.query.all()
    result = predictions_schema.dump(all_predictions)

    emit('response', jsonify(result), broadcast=True)


db.create_all()

if __name__ == "__main__":
    
    # specify thread as false b/c of tf compatibility while running flask: flask run --without-threads
    socketio.run(app)
