from tensorflow import keras
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import matplotlib as mpl
import tensorflow as tf
from sklearn.model_selection import train_test_split

def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "autoencoders"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

def show_reconstructions(model, images=X_valid, n_images=5):
    reconstructions = model.predict(images[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(images[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])


print("## STACKED DENOISING ##")
tf.random.set_seed(42)
np.random.seed(42)

noise_layer = keras.layers.GaussianNoise(0.2)

# denoising_encoder = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.GaussianNoise(0.2),
#     keras.layers.Dense(100, activation="selu"),
#     keras.layers.Dense(30, activation="selu")
# ])


denoising_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu")
])

denoising_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
denoising_ae = keras.models.Sequential([noise_layer, denoising_encoder, denoising_decoder])
denoising_ae.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(learning_rate=1.0))

history = denoising_ae.fit(X_train_full, X_train_full, epochs=5,
                           validation_data=(X_valid, X_valid))

print(denoising_encoder.predict(X_valid[:10]))

tf.random.set_seed(42)
np.random.seed(42)

noise = keras.layers.GaussianNoise(0.2)
show_reconstructions(denoising_ae, noise(X_valid, training=True))
plt.show()

# denoising_ae.save_weights('my_model_weights.h5')

print("## VISUALIZE FEATURES ##")
X_valid_compressed = denoising_encoder.predict(X_valid)
tsne = TSNE()
X_valid_2D = tsne.fit_transform(X_valid_compressed)
X_valid_2D = (X_valid_2D - X_valid_2D.min()) / (X_valid_2D.max() - X_valid_2D.min())
# adapted from https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
plt.figure(figsize=(10, 8))
cmap = plt.cm.tab10
plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap=cmap)
image_positions = np.array([[1., 1.]])
for index, position in enumerate(X_valid_2D):
    dist = np.sum((position - image_positions) ** 2, axis=1)
    if np.min(dist) > 0.02: # if far enough from other images
        image_positions = np.r_[image_positions, [position]]
        imagebox = mpl.offsetbox.AnnotationBbox(
            mpl.offsetbox.OffsetImage(X_valid[index], cmap="binary"),
            position, bboxprops={"edgecolor": cmap(y_valid[index]), "lw": 2})
        plt.gca().add_artist(imagebox)
plt.axis("off")
save_fig("mnist_visualization_plot")
plt.show()

#print('## CONVOLUTIONAL WITH REUSED WEIGHTS ##')
## Do this with all data

X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, train_size=0.1, random_state=42)
print("FULL SHAPE", X_train_full.shape)
print("SHRINKED SHAPE", X_train.shape)
tf.random.set_seed(42)
np.random.seed(42)

inp_classifier = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(10, activation="softmax")
])
    
print('## 10% of training ##')
class_ae = keras.models.Sequential([denoising_encoder, inp_classifier])

class_ae.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=0.1),
                 metrics=["accuracy"])

class_ae.fit(X_train, y_train, epochs=10,
            validation_data=(X_valid, y_valid))


print('## Full training ##')
class_ae_full = keras.models.Sequential([denoising_encoder, inp_classifier])
class_ae_full.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=0.1),
                 metrics=["accuracy"])

class_ae_full.fit(X_train_full, y_train_full, epochs=10,
            validation_data=(X_valid, y_valid))
# conv_encoder = keras.models.Sequential([
#     keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
#     keras.layers.Conv2D(16, kernel_size=3, padding="SAME", activation="selu"),
#     keras.layers.MaxPool2D(pool_size=2),
#     keras.layers.Conv2D(32, kernel_size=3, padding="SAME", activation="selu"),
#     keras.layers.MaxPool2D(pool_size=2),
#     keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="selu"),
#     keras.layers.MaxPool2D(pool_size=2)
# ])

# load weights from prior model
# conv_encoder.load_weights('my_model_weights.h5', by_name=True)

# conv_decoder = keras.models.Sequential([
    
#     keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="VALID", activation="selu",
#                                  input_shape=[3, 3, 64]),
#     keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="SAME", activation="selu"),
#     keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="SAME", activation="sigmoid"),
#     keras.layers.Reshape([28, 28])
# ])
# conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])

# conv_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(learning_rate=1.0),
#                 metrics=[rounded_accuracy])
# history = conv_ae.fit(X_train, X_train, epochs=5,
#                       validation_data=(X_valid, X_valid))

