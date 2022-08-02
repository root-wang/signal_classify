import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from signal_classfy.data_init import data_init, data_test_init
# initialize optimizer
from signal_classfy.network.ResNet import ResNet
from signal_classfy.test import test
from signal_classfy.typing.Classify_enum import classes
from signal_classfy.utils.save_model import saveModel, SAVE_MODEL

# AM FM

if __name__ == "__main__":
    adm = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    num_epochs = 10

    # set batch size
    batch = 1

    weights_path = saveModel()
    if SAVE_MODEL is True:
        filepath = weights_path + "/{epoch}.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False,
                                                     mode="auto")
        callbacks_list = [checkpoint]
    else:
        callbacks_list = []
        x_train, y_train, x_val, y_val, x_test, y_test = data_init()
        model = ResNet((1024, 2), 4)
        model.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch, callbacks=callbacks_list,
                            validation_data=(x_val, y_val))
        import itertools
        from sklearn.metrics import confusion_matrix

        predictions = model.predict(x_test)
        classes_y = np.argmax(predictions, axis=1)
        conf_matrix = confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=classes_y)
        nr_rows = conf_matrix.shape[0]
        nr_cols = conf_matrix.shape[1]

        plt.figure(figsize=(18, 18), dpi=200)
        im = plt.imshow(conf_matrix, cmap=plt.cm.Greens)
        ax = plt.gca()
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('Actual Labels', fontsize=12)
        plt.xlabel('Predicted Labels', fontsize=12)
        tick_marks = np.arange(len(classes))
        plt.yticks(tick_marks, classes)
        # plt.xticks(tick_marks, classes)

        for i, j in itertools.product(range(nr_rows), range(nr_cols)):
            plt.text(j, i, conf_matrix[i, j], horizontalalignment='center',
                     color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')
        plt.show()

        train_accuracy = history.history['accuracy']
        train_loss = history.history['loss']
        val_accuracy = history.history['val_accuracy']
        val_loss = history.history['val_loss']
        history = [[train_accuracy, val_accuracy], [train_loss, val_loss]]
        title = ['Model accuracy', 'Model loss']
        ylabel = ['accuracy', 'loss']
        fig = plt.figure(figsize=(20, 7), dpi=80)
        for i, id in enumerate(title):
            plt.subplot(1, 2, i + 1)

            plt.plot(history[i][0], label='train')
            plt.plot(history[i][1], label='val')
            plt.title(id)
            plt.xlabel('epoch')
            plt.ylabel(ylabel[i])
            plt.legend(loc='upper right')
        plt.tight_layout(pad=1.7)
        plt.show()
        idx = [0, 1, 2, 3]
        fig = plt.figure(figsize=(20, 5), dpi=80)
        for i, id in enumerate(idx):
            plt.subplot(2, 2, i + 1)
            plt.plot(x_train[id][:, 0], color='green', label='I component')
            plt.plot(x_train[id][:, 1], color='salmon', label='Q component')
            plt.title(classes[id])
            plt.xlabel('Points')
            plt.ylabel('Amplitude')
            plt.legend(loc='upper right')
        plt.tight_layout(pad=1.7)
        plt.show()
    # test(model)
