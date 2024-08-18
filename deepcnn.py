import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(0)

# Load and preprocess data
data = np.load('D:\\ML\\data_with_labels.npz')
train = data['arr_0'] / 255.0
labels = data['arr_1']

print(train[0])
print(labels[0])

plt.figure(figsize=(6, 6))
f, plts = plt.subplots(5, sharex=True)
c = 91
for i in range(5):
    plts[i].pcolor(train[c + i * 558], cmap=plt.cm.gray_r)
plt.show()

def to_onehot(labels, nclasses=5):
    outlabels = np.zeros((len(labels), nclasses))
    for i, l in enumerate(labels):
        outlabels[i, l] = 1
    return outlabels

onehot = to_onehot(labels)

# Split the data into training and testing sets
indices = np.random.permutation(train.shape[0])
valid_cnt = int(train.shape[0] * 0.1)
test_idx, training_idx = indices[:valid_cnt], indices[valid_cnt:]
test, train = train[test_idx, :], train[training_idx, :]
onehot_test, onehot_train = onehot[test_idx, :], onehot[training_idx, :]

# Reshape the data to match the expected input shape for Conv2D
train = train.reshape([-1, 36, 36, 1])
test = test.reshape([-1, 36, 36, 1])

# Define the convolutional neural network model
inputs = tf.keras.Input(shape=(36, 36, 1))
x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
x = tf.keras.layers.Conv2D(4, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model using a custom training loop with tqdm
epochs = 1000
batch_size = 32
train_acc = []
test_acc = []

# Create a custom training loop
for epoch in tqdm(range(epochs), desc="Training Progress"):
    history = model.fit(train, onehot_train, epochs=1, batch_size=batch_size, validation_data=(test, onehot_test), verbose=0, callbacks=[early_stopping])
    train_acc.append(history.history['accuracy'][0])
    test_acc.append(history.history['val_accuracy'][0])
    if early_stopping.stopped_epoch > 0:
        print(f"Early stopping at epoch {early_stopping.stopped_epoch}")
        break

print(train_acc[-1])
print(test_acc[-1])

# Plot the training and validation accuracy
plt.figure(figsize=(6, 6))
plt.plot(train_acc, 'bo', label='Training accuracy')
plt.plot(test_acc, 'b', label='Validation accuracy')
plt.legend()
plt.show()

# Plot the weights of the model
weights = model.layers[1].weights[0].numpy()
f, plts = plt.subplots(4, sharex=True)
for i in range(4):
    plts[i].pcolor(weights[:, :, 0, i].reshape([3, 3]))

plt.show()

# Predict on the test set
pred = np.argmax(model.predict(test), axis=1)
conf = np.zeros([5, 5])
for p, t in zip(pred, np.argmax(onehot_test, axis=1)):
    conf[t, p] += 1

# Plot the confusion matrix
plt.figure(figsize=(8, 8))
plt.imshow(conf, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.colorbar()

# Annotate the confusion matrix
for i in range(conf.shape[0]):
    for j in range(conf.shape[1]):
        plt.text(j, i, int(conf[i, j]), ha='center', va='center', color='red')

plt.show()
