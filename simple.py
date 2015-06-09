from ada_dnn import DNN
import numpy

#you need numpy data file
train_x = numpy.load('X.npy')
train_y = numpy.load('Y.npy')


dnn = DNN([69,512, 39],
          learning_rate_decays=0.98, learning_rate=0.02,
          batch_size=256, epochs=10)

dnn.fit(train_x, train_y)
dnn.predict_proba(train_x)

