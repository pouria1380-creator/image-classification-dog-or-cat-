from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten 
import numpy as np
import random
import matplotlib.pyplot as plt
x_train = np.loadtxt('input.csv', delimiter=',')
y_train = np.loadtxt('labels.csv', delimiter=',')
x_test = np.loadtxt('input_test.csv', delimiter=',')
y_test = np.loadtxt('labels_test.csv', delimiter=',')


x_train =x_train.reshape(len(x_train),100, 100, 3)
y_train =y_train.reshape(len(y_train),1)
x_test = x_test.reshape(len(x_test),100, 100 , 3)
y_test = y_test.reshape(len(y_test),1)

x_train  /= 255.0
x_test /=255.0


print(f'shape of x train: {x_train.shape} \nshape of y train: {y_train.shape} \nshape of x test: {x_test.shape} \nshape of y_test: {y_test.shape}')


print(x_train[1:])
print(x_test[1:])


random_index = random.randint(0, len(x_train))
plt.imshow(x_train[random_index, :])
plt.show()


#model 
model = Sequential([
    Conv2D(32, (3,3), activation = 'relu', input_shape = (100, 100, 3)), # number of filters we want to use | size of the filter we want to use | activation function to use | what kind of shape this model is expecting
    MaxPooling2D((2,2)), # only filter size is needed to be determined

    Conv2D(32, (3,3), activation = 'relu'), # input shape is ony for the first layer
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation = 'relu'),  #64 is for determining how many neurons in a layer
    Dense(1, activation = 'sigmoid')
])
#another way is to:
# model = Sequential()
# model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (100, 100, 3)))
#model.add(MaxPooling2D((2,2)))
#model.add(Conv2D(32, (3,3), activation = 'relu'))
#model.add(MaxPooling2D((2,2)))
#model.add(Flatten())
#model.add(Dense(64, activation = 'relu'))
#model.add(Dense(1, activation= ;sigmoid))




# cost function and back propagation

model.compile(loss='binary_crossentropy',optimizer = 'adam', metrics = ['accuracy'] )
model.fit(x_train, y_train, epochs= 5, batch_size = 64)
#evaluate
model.evaluate(x_test, y_test)



# making predictions

random_index_two = random.randint(0,len(y_test))
plt.imshow( x_test[random_index_two, :])
plt.show()

y_prediction = model.predict(x_test[random_index_two, :].reshape(1, 100, 100, 3)) 
y_prediction = y_prediction >0.5

if (y_prediction ==0 ):
    prediction = 'dog'
else:
    prediction = 'cat'
print(f'the model predicted that this picture is a : {prediction}')