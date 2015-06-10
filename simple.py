from ada_dnn import DNN
import numpy
from sklearn.externals import joblib

#you need numpy data file
train_x = numpy.load('X.npy')
train_y = numpy.load('Y.npy')


dnn = DNN([69,2048, 2048, 2048, 2048, 48],
          learning_rate=0.02,
          dropouts=0.25,
          batch_size=256, epochs=30)

dnn.fit(train_x, train_y)
joblib.dump(dnn, 'model')
dnn.predict_proba(train_x)

