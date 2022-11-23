from functools import lru_cache
from matplotlib import pyplot as plt
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

DATA_ROOT = Path(__file__).parent.joinpath("data").joinpath("keypoints3d")

@lru_cache(maxsize=1)
def skel():
    return [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder","right_shoulder", 
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", 
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

def load(path):
    """
    p: (time, joint, dof), joint = 17
    p_optim : (time, joint, dof), joint = 17
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    p, p_optim = obj.values()

    time, joints, dof = p.shape
    assert joints == len(skel())
    return p, p_optim

def iter():
    """
    p: (time, joint, dof), joint = 17
    p_optim : (time, joint, dof), joint = 17
    """
    for path in DATA_ROOT.glob("*.pkl"):
        p, p_optim = load(path)
        yield p, p_optim

def show():
    from itertools import chain
    from tqdm import tqdm

    def create_canvas():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        return fig, ax

    def plot_pose(p, fig=None, ax=None, scale=1):
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')

        ax.cla()
        ax.scatter(p[:, 0], p[:, 2], p[:, 1])
        ax.set_xlim(-1 * scale, 1 * scale)
        ax.set_ylim(-1 * scale, 1 * scale)
        ax.set_zlim(-0.5 * scale, 1 * scale)

        plt.draw()
        plt.pause(0.001)
    
    fig, ax = create_canvas()
    for _p in tqdm(chain.from_iterable((p_optim for p, p_optim in iter()))):
        plot_pose(_p, fig, ax, scale=200)
    plt.show()

class DataList():
    """
    doc and download page: https://google.github.io/aistplusplus_dataset/download.html
    download keypoint3d and extract to data directory of this module.
    """
    def __init__(self, data_root=None) -> None:
        if data_root is None:
            self.data_root = DATA_ROOT
        else:
            self.data_root = Path(data_root)

        self.paths = list(self.data_root.glob("*.pkl"))

    def __len__(self, ) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx):
        p, p_optiom = load(self.paths[idx])
        return p_optiom
