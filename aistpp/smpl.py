from functools import lru_cache
from matplotlib import pyplot as plt
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import torch
from smplx import SMPL

DATA_ROOT = Path(__file__).parent.joinpath("data").joinpath("motions")
SMPL_DIR = Path(__file__).parent.joinpath("data").joinpath("spml_model")

@lru_cache(maxsize=1)
def skel():
    return [
        "root",     
        "lhip", "rhip", "belly",    
        "lknee", "rknee", "spine",    
        "lankle", "rankle", "chest",     
        "ltoes", "rtoes", "neck", 
        "linshoulder", "rinshoulder",     
        "head",  "lshoulder", "rshoulder",      
        "lelbow", "relbow",      
        "lwrist", "rwrist",     
        "lhand", "rhand",
        ]

@lru_cache(maxsize=1)
def skel_conn_map():
    return np.array(
        [
            [0, 1], 
            [0, 2], 
            [0, 3], 
            [1, 4], 
            [2, 5], 
            [3, 6], 
            [4, 7], 
            [5, 8], 
            [6, 9], 
            [7, 10], 
            [8, 11], 
            [9, 12], 
            [9, 13], 
            [9, 14], 
            [12, 15], 
            [13, 16], 
            [14, 17], 
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 22],
            [21, 23],
        ]
    )

@lru_cache(maxsize=-1)
def model(smpl_dir=None) -> SMPL:
    if smpl_dir is None:
        smpl_dir = SMPL_DIR

    return SMPL(model_path=smpl_dir, gender='MALE', batch_size=1)

def load(path):
    """
    p: (time, joint, dof), joint = 24
    root_p : (time, joint, dof), joint = 24
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    smpl_loss, smpl_poses, smpl_scaling, smpl_trans = obj.values()

    smpl_output = model().forward(
            global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
            body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
            transl=torch.from_numpy(smpl_trans).float(),
            scaling=torch.from_numpy(smpl_scaling.reshape(1, 1)).float(),
    )

    # although smpl_poses has 24 joint, but the output global position will have 45 joints
    # hands and feets joints will be added at the tail 
    # so just trim the first 24 joints to maintain the amount of joints
    p = smpl_output.joints.detach().numpy()[:, :24, :]
    root_p = smpl_trans # time, dof

    return p, root_p

def iter():
    """
    p: (time, joint, dof), joint = 24
    root_p : (time, joint, dof), joint = 24
    """
    for path in DATA_ROOT.glob("*.pkl"):
        lr, root_p = load(path)
        yield lr, root_p

def show(ps=None):
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

        for idx, _ in enumerate(p):
            ax.text(p[idx, 0], p[idx, 2], p[idx, 1], idx)

        ax.set_xlim(-0.5 * scale, 0.5 * scale)
        ax.set_ylim(0 * scale, 1 * scale)
        ax.set_zlim(-0.5 * scale, 0.5 * scale)

        plt.draw()
        plt.pause(0.001)
    
    fig, ax = create_canvas()
    if ps is None:
        for _p in tqdm(chain.from_iterable((p for p, root_p in iter()))):
            plot_pose(_p, fig, ax, scale=200)
    else:
        for _p in tqdm(ps):
            plot_pose(_p, fig, ax, scale=200)
    plt.show()

class DataList():
    """
    doc and download page: https://google.github.io/aistplusplus_dataset/download.html
    download keypoint3d and extract to data directory of this module.
    """
    def __init__(self, data_root=None, include_rootp=False) -> None:
        if data_root is None:
            self.data_root = DATA_ROOT
        else:
            self.data_root = Path(data_root)

        self.paths = list(self.data_root.glob("*.pkl"))
        self.include_rootp = include_rootp

    def __len__(self, ) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx):
        lr, root_p = load(self.paths[idx])
        if self.include_rootp:
            return lr, root_p
        else:
            return lr