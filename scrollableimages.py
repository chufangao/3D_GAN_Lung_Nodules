import pickle
import numpy as np
import matplotlib.pyplot as plt


threshold = .9
index = 2

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2
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
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

posdat = pickle.load(open('images/gen_nod0.pickle','rb'))
# with open('images/PositiveAugmented.pickle','rb') as f:
#     posdat = pickle.load(f)

# print(np.average(posdat, axis=(1,2,3,4))); exit()
print('white imgs', [i for i, v in enumerate(posdat) if np.average(v) > threshold])
posdat = np.array(posdat)
print(posdat.shape)

fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, posdat[index,:,:,:,0])
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()