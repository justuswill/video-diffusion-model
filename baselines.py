import sys
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import multiprocessing, threading

import torch
from video_diffusion_local import Unet3D, GaussianDiffusion

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ------------------------------
# Tools
# ------------------------------


def progress(iterable, text=None, inner=None, timed=None):
    """
    Generator for timed for loops with progress bar

    :param iterable, inner: iterable for outer and (optional) inner loop
    :param text: (optional) Task description
    :param timed: [list of (delta, f)] events that are triggered by calling <f> after <delta> seconds have passed
    """
    text = text + ' ' if text is not None else ''
    start = time.time()
    last = start
    if timed is not None:
        last_timed = {item: start for item in timed}

    def handle_events(force=False):
        for (dt, f), lt in last_timed.items():
            if now - lt > dt or force:
                f()
                last_timed[(dt, f)] = now

    # for loop
    if inner is None:
        for i, x in enumerate(iterable):
            now = time.time()
            if i == 0 or i == len(iterable) - 1 or now - last > 0.5:
                last = now
                # Progress percentage at step completion, TBD percentage shortly after step start
                perc = (i + 1) / len(iterable)
                inv_perc = len(iterable) / (i + 0.1)
                sys.stdout.write("\r%s[%.1f %%] - %d / %d - %.1fs [TBD: %.1fs]" %
                                 (text, 100 * perc, i + 1, len(iterable), now-start, (now-start) * inv_perc))
                sys.stdout.flush()
            # Call events
            if timed is not None:
                handle_events()
            yield x
    # for loop in for loop
    else:
        for i, x in enumerate(iterable):
            for j, y in enumerate(inner):
                now = time.time()
                if j == 0 or j == len(inner) - 1 or now - last > 0.5:
                    last = now
                    perc = (i * len(inner) + j + 1) / (len(iterable) * len(inner))
                    inv_perc = (len(iterable) * len(inner)) / (i * len(inner) + j + 0.1)
                    sys.stdout.write("\r%s[%.1f %%] - %d / %d (%d / %d) - %.1fs [TBD: %.1fs]" %
                                     (text, 100 * perc, i + 1, len(iterable), j + 1, len(inner),
                                      now-start, (now-start) * inv_perc))
                    sys.stdout.flush()
                # Call events
                if timed is not None:
                    handle_events()
                yield x, y

    if timed is not None:
        handle_events(force=True)
    print()

# ------------------------------
# Data
# ------------------------------


class HDF5Dataset(torch.utils.data.Dataset):
    """ Generic dataset for data stored in h5py as uint8 numpy arrays.
    Reads videos in {0, ..., 255} and returns in range [-0.5, 0.5] """
    def __init__(self, data_file, sequence_length, train=True, resolution=64, transform=None):
        """
        Args:
            data_file: path to the pickled data file with the
                following format:
                {
                    'train_data': [B, H, W, 3] np.uint8,
                    'train_idx': [B], np.int64 (start indexes for each video)
                    'test_data': [B', H, W, 3] np.uint8,
                    'test_idx': [B'], np.int64
                }
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        # read in data
        self.data_file = data_file
        self.data = h5py.File(data_file, 'r')
        self.prefix = 'train' if train else 'test'
        self._images = self.data[f'{self.prefix}_data']
        self._idx = self.data[f'{self.prefix}_idx']
        self.size = len(self._idx)
        self.transform = transform

    @property
    def n_classes(self):
        raise Exception('class conditioning not support for HDF5Dataset')

    def __getstate__(self):
        state = self.__dict__
        state['data'] = None
        state['_images'] = None
        state['_idx'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.data = h5py.File(self.data_file, 'r')
        self._images = self.data[f'{self.prefix}_data']
        self._idx = self.data[f'{self.prefix}_idx']

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        start = self._idx[idx]
        end = self._idx[idx + 1] if idx < len(self._idx) - 1 else len(self._images)
        assert end - start >= 0
        start = start + np.random.randint(low=0, high=end - start - self.sequence_length)
        assert start < start + self.sequence_length <= end
        video = torch.tensor(self._images[start:start + self.sequence_length])
        if self.transform is not None:
            video = self.transform(video)
        # return dict(video=preprocess(video, self.resolution))
        return dict(video=video)


def preprocess(x):
    # F H W C -> C F H W
    x = x.permute(3, 0, 1, 2)
    # [0, 255] -> [-1, 1]
    x = (x / 255 - 0.5) * 2
    return x


def plot_video(images, f='test.gif'):
    # [-1, 1] -> [0, 1]
    images = 0.5 * images + 0.5
    # C F H W -> F H W C
    images = images.permute(1, 2, 3, 0)
    fig, ax = plt.subplots()
    im = ax.imshow(images[0], animated=True)
    ax.axis('off')

    # Animation update function
    def update(frame):
        im.set_data(images[frame])
        return im,

    # Create animation
    ani = FuncAnimation(fig, update, frames=images.shape[0], blit=True)
    writer = PillowWriter(fps=4)
    ani.save(f, writer=writer)

    def looping():
        i = 0
        while plt.fignum_exists(fig.number):
            update(i % images.shape[0])
            plt.draw()
            time.sleep(0.25)

    # thread = multiprocessing.Process(target=looping)
    thread = threading.Thread(target=looping)
    thread.start()
    plt.show()


# ------------------------------
# Training the model
# ------------------------------

def get_data_and_model(train=True, ckpt=None):
    # Data
    dataset = HDF5Dataset(data_file='C:/Users/Justus/Downloads/bair/data/bair.hdf5', sequence_length=16, train=train,
                          resolution=64, transform=preprocess)
    data = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    batch = next(iter(data))
    print(batch['video'].shape)
    # plt.imshow(batch['video'][0, 0])
    # plot_video(batch['video'][0])

    # Model
    unet = Unet3D(
        dim=32,
        dim_mults=(1, 2, 4, 8)
    )
    model = GaussianDiffusion(
        unet,
        image_size=64,
        num_frames=16,
        timesteps=250,  # number of steps
        loss_type='l1'  # L1 or L2
    )
    model.to(device)

    if ckpt is not None:
        params = torch.load(ckpt, map_location=device)
        model.load_state_dict(params['model'])

    return data, model


def train():
    torch.manual_seed(123)
    data, model = get_data_and_model()

    # Training
    epochs = 1
    losses = []
    opt = torch.optim.Adam(model.parameters())
    for epoch, batch in progress(range(epochs), inner=data, text='Training',
                                 timed=[(1200, lambda: torch.save({'model': model.state_dict(), 'opt': opt.state_dict()}, 'checkpoint.cp'))]):
        x = batch['video'].to(device)
        opt.zero_grad()
        loss = model(x)
        loss.backward()
        losses += [loss.item()]
        opt.step()
    print('Last-epoch loss: %.2f' % sum(losses[-len(data):-1]))
    print('Finished Training')

    sample = model.sample(batch_size=4)
    print(sample.shape)
    plot_video(sample[0])


def evaluate(ckpt):
    data, model = get_data_and_model(train=False, ckpt=ckpt)

    with torch.no_grad():
        for batch in progress(data, text='Evaluating'):
            x = batch['video'].to(device)
            x_hat = model.sample(frames=x)
            plot_video(x_hat, 'tmp_pred.gif')


if __name__ == '__main__':
    # train()
    evaluate(ckpt='checkpoint_1ep.cp')
