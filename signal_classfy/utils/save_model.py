import os

SAVE_MODEL = False
SAVE_HISTORY = False


# create directory for model weights
def saveModel():
    if SAVE_MODEL is True:
        weights_path = "D:\project\code\dataset\weights"

        try:
            os.mkdir(weights_path)
        except OSError:
            print("Creation of the directory %s failed" % weights_path)
        else:
            print("Successfully created the directory %s " % weights_path)
        print('\n')
        return weights_path

    # create directory for model history
    if SAVE_HISTORY is True:
        history_path = os.path.join("../", "./dataset/history/")

        try:
            os.mkdir(history_path)
        except OSError:
            print("Creation of the directory %s failed" % history_path)
        else:
            print("Successfully created the directory %s " % history_path)
        print('\n')
