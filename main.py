from re import L
from pathlib import Path
import numpy as np

from aistpp import PAEDataList
from tqdm import tqdm

#%%
f = open("Dataset/all_data.bin", "wb")
f_seq = open("Dataset/all_seq.txt", "w")
f_shape = open("Dataset/all_shape.txt","w")

wcount = 0
pae_datalist = PAEDataList()
for bvhid, bvhpath, w in tqdm(pae_datalist):
    f.write(w.ravel().astype(np.float32).tobytes())
    f_seq.write(f"{bvhid} {Path(bvhpath).stem}\n")
    wcount += 1
    
f_shape.write(f"{wcount}\n{8712}")

f.close()
f_seq.close()
f_shape.close()