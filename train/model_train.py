import tensorflow as tf
import tensorflow.keras as keras
from pre_process import pics_dataset
import matplotlib.pyplot as plt
DATASET_ROOT_PATH="F://dataset//hanzi_dataset//dataset_character//dataset"
TRAIN_PATH=DATASET_ROOT_PATH+"//train"
TEST_PATH=DATASET_ROOT_PATH+"//test"

IMG_SIZE=32
CHANNLES=3
NUM_CLASS=3755
BATCH_SIZE=16
EPOCH=1


def change_range(image,label):
    return 2*image-1,label

def loadModel(class_nums):
    base_model=keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE,IMG_SIZE,CHANNLES),
        include_top=False
    )
    base_model.trainable=False
    model=keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(class_nums,activation="softmax")
    ])
    return model

def train():
    # 加载数据集
    train_ds,train_num=pics_dataset.get_dataSet(TRAIN_PATH)
    test_ds,test_num=pics_dataset.get_dataSet(TEST_PATH)
    # 加载模型
    model=loadModel(NUM_CLASS)
    # # Fine tune from this layer onwards
    # fine_tune_at = 249
    #
    # # Freeze all the layers before the `fine_tune_at` layer
    # for layer in model.layers[:fine_tune_at]:
    #     layer.trainable = False
    # 设置batch—size
    train_ds_batch=pics_dataset.set_batch_shuffle(BATCH_SIZE,train_ds,train_num)
    test_ds_batch=pics_dataset.set_batch_shuffle(BATCH_SIZE,test_ds,test_num)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"]
                  )
    print("可训练变量{}".format(len(model.trainable_variables)))
    model.summary()

    train_steps_per_epoch=tf.math.ceil(train_num/BATCH_SIZE).numpy()
    valid_stpes_per_epoch=tf.math.ceil(test_num/BATCH_SIZE).numpy()
    history = model.fit(train_ds_batch,
                        epochs=EPOCH,
                        validation_data=test_ds_batch,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_steps=valid_stpes_per_epoch)
    test_loss, test_accuracy = model.evaluate(test_ds_batch, steps=20)
    print("initial loss: {:.2f}".format(test_loss))
    print("initial accuracy: {:.2f}".format(test_accuracy))
    return history

def show_loss_accuracy(history):
    acc=history.history['accyracy']
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
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
if __name__ == '__main__':
    history=train()
    show_loss_accuracy(history)