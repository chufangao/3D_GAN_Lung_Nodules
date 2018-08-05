import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle


class IndexTracker(object):
    # for scrolling thru slices of a nodule
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')
        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices // 2
        self.minind = 0
        self.maxind = 17
        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')
        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up' and self.ind < self.maxind:
            self.ind = (self.ind + 1) % self.slices
        elif event.button == 'down' and self.ind > self.minind:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def saveImgs(posdat, path):
    # input dim (n, 40, 40, 18, 1)
    fig, axs = plt.subplots(3, 6)
    cnt = 0
    for k in range(0, posdat.shape[0]):
        for i in range(3):
            for j in range(6):
                axs[i,j].imshow(posdat[k,:,:,cnt,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        cnt = 0
        fig.savefig(path+'test'+str(k)+'.png')


def scrollImg(image):
    # input dim (40, 40, 18, 1)
    # can't use Agg as backend
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, image[:,:,:,0])
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


if __name__ == '__main__':
    posdat = pickle.load(open('images/gen_nod3.pickle','rb'))
    # posdat = pickle.load(open('images/PositiveAugmented.pickle','rb'))
    print(posdat.shape)
    # print('white imgs', [i for i, v in enumerate(posdat) if np.average(v) > .9])
    # print('black imgs', [i for i, v in enumerate(posdat) if np.average(v) < -.9])
    saveImgs(posdat, 'desktop/')
    # scrollImg(posdat[0])