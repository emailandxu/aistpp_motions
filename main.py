import aistpp
import matplotlib.pyplot as plt
import numpy as np


datalist = aistpp.smpl.DataList()

smpl_loss, smpl_poses, smpl_scaling, smpl_trans = aistpp.load_smpl(datalist.paths[0])

skel = aistpp.smpl_to_p(smpl_poses, smpl_scaling, smpl_trans)


aistpp.show(np.concatenate([skel[:, [0]], smpl_trans[:, np.newaxis]], axis=1))

