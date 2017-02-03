from helper.inception_v3 import InceptionV3
from helper.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from helper.imagenet_utils import decode_predictions
import numpy as np

model = InceptionV3(include_top=True, weights='imagenet')

img_path = 'data/cat.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))

