import pickle
from matplotlib import pyplot as plt


class FOELossHistory:
    def __init__(self, path):
        self.path = path
        self.data = {'train_loss': [], 'val_loss': [],
                     'val_loss_gd': [], 'val_loss_bd': []}

    def append(self, train_loss, val_loss, val_loss_gd, val_loss_bd):
        self.data['train_loss'].append(train_loss)
        self.data['val_loss'].append(val_loss)
        self.data['val_loss_gd'].append(val_loss_gd)
        self.data['val_loss_bd'].append(val_loss_bd)

    def plot(self):
        plt.figure()
        plt.semilogy(self.data['train_loss'], label='Train')
        plt.semilogy(self.data['val_loss'], label='Validation')
        plt.semilogy(self.data['val_loss_gd'], '--', label='Val_Good', )
        plt.semilogy(self.data['val_loss_bd'], '--', label='Val_Bad')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.grid()
        plt.legend()
        plt.title('loss')
        plt.show()

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_path, verbose=True):
        loss_history = FOELossHistory(file_path)
        with open(file_path, 'rb') as f:
            loss_history.data = pickle.load(f)

        if verbose:
            print('Loaded loss history from {}.'.format(file_path))

        return loss_history
