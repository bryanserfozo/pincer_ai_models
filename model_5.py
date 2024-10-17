import tensorflow as tf
import tensorflow_datasets as tfds

# Load the QuickDraw dataset (filtering by category)
quickdraw_ds = tfds.load('quickdraw_bitmap', split='train', as_supervised=True)

# Define a function to label "crab" as 1 and others as 0
def preprocess_doodle(image, label):
    crab_label = 1 if label == 82 else 0
    # Normalize the image (from [0, 255] to [0, 1])
    image = tf.cast(image, tf.float32) / 255.0
    # Reshape image to fit into model input (28x28)
    image = tf.reshape(image, (28, 28, 1))
    return image, crab_label

# Apply the preprocessing function to your dataset
processed_ds = quickdraw_ds.map(preprocess_doodle)

# Filter to only get 'crab' examples
crab_ds = processed_ds.filter(lambda image, label: label == 1)

# Filter to get 'not crab' examples
not_crab_ds = processed_ds.filter(lambda image, label: label == 0)

# Take equal amounts of crab and not crab data (you can adjust the ratio)
crab_count = 100_000  # Approximate number of crab samples
not_crab_ds = not_crab_ds.take(crab_count)

# Combine both datasets and shuffle
# balanced_ds = crab_ds.concatenate(not_crab_ds).shuffle(buffer_size=crab_count * 2)

balanced_ds = crab_ds.concatenate(not_crab_ds)

cached_ds = balanced_ds.cache()
cached_and_shuffled_ds = cached_ds.shuffle(buffer_size=crab_count * 2)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification output
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Split the dataset into training and validation sets
train_size = int(0.8 * crab_count * 2)
train_ds = cached_and_shuffled_ds.take(train_size).batch(32)
val_ds = cached_and_shuffled_ds.skip(train_size).batch(32)

# Add early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=[early_stopping])

# Save the model in TensorFlow SavedModel format
model.export('pincer_ai_model_2')

# Convert the saved model to TensorFlow.js format using the following command:
# tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model path_to_save_model path_to_save_tfjs_model
