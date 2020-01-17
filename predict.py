from PIL import Image
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from numpy import expand_dims
from sklearn.preprocessing import Normalizer
from joblib import load
from keras.models import load_model

facenet_model = load_model('../facenet_keras.h5')
model = load('../bmark-one-model.joblib')


def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)

    # convert to RGB, if needed
    image = image.convert('RGB')
    # print(image)
    # convert to array
    pixels = asarray(image)

    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)

    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']

    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


def normalize(embedding):
    in_encoder = Normalizer(norm='l2')
    face = in_encoder.transform(embedding)
    return face


def predict(image):
    face = extract_face(image)
    face_embed = get_embedding(facenet_model, face)
    face_norm = normalize(face_embed.reshape(1, -1))
    yhat = model.predict(face_norm)

    return yhat