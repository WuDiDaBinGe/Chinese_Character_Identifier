import os
import tensorflow as tf
import tensorflow.keras as keras
from  tensorflow.keras.callbacks import ReduceLROnPlateau
from pre_process import pics_dataset
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

DATASET_ROOT_PATH="/home/wbq/code/singleChar/dataset/"
TRAIN_PATH=DATASET_ROOT_PATH+"//train"
TEST_PATH=DATASET_ROOT_PATH+"//test"
MODEL_SAVE="./model_save"
LOG_DIR="./log"


IMG_SIZE=64
CHANNLES=3
NUM_CLASS=3755
BATCH_SIZE=32
EPOCH=20


def change_range(image,label):
    return 2*image-1,label

def loadModel(class_nums):
    base_model=keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE,IMG_SIZE,CHANNLES),
        include_top=False
    )
    # base_model.load_weights("../models_weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")
    # Fine tune from this layer onwards
    fine_tune_at = 129
    base_model.trainable =True
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model=keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(class_nums,activation="softmax")
    ])
    return model

def train():
    # Load Dataset
    train_ds,train_num=pics_dataset.get_dataSet(TRAIN_PATH)
    test_ds,test_num=pics_dataset.get_dataSet(TEST_PATH)
    # Load Model
    model=loadModel(NUM_CLASS)

    # set batch—size
    train_ds_batch=pics_dataset.set_batch_shuffle(BATCH_SIZE,train_ds,train_num)
    test_ds_batch=pics_dataset.set_batch_shuffle(BATCH_SIZE,test_ds,test_num)
    # LR Delay
    sgd=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # LR reduce with epoch
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,patience=5, mode='auto')


    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  )
    print("Tainable num is:{}".format(len(model.trainable_variables)))
    model.summary()

    train_steps_per_epoch=tf.math.ceil(train_num/BATCH_SIZE).numpy()
    valid_stpes_per_epoch=tf.math.ceil(test_num/BATCH_SIZE).numpy()

    # Creating Keras callbacks
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=LOG_DIR, histogram_freq=1)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        'training_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=5)
    os.makedirs('training_checkpoints/', exist_ok=True)
    early_stopping_checkpoint = keras.callbacks.EarlyStopping(patience=5)


    history = model.fit(train_ds_batch,
                        epochs=EPOCH,
                        validation_data=test_ds_batch,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_steps=valid_stpes_per_epoch,
                        callbacks=[reduce_lr,tensorboard_callback,model_checkpoint_callback,early_stopping_checkpoint])

    test_loss, test_accuracy = model.evaluate(test_ds_batch)
    print("initial loss: {:.2f}".format(test_loss))
    print("initial accuracy: {:.2f}".format(test_accuracy))
    model.save(MODEL_SAVE)
    return history

def show_loss_accuracy(history):
    print("start showing!")
    acc=history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    plt.plot(acc,label="Training Accuracy!")
    plt.plot(val_acc,label="Validation Accuaracy!")
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title("Training and Valiadation Accuracy!")

    plt.subplot(2,1,2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 20])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    plt.savefig('./classification_pr.png')

if __name__ == '__main__':
    history=train()
    show_loss_accuracy(history)