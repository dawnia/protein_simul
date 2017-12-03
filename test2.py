import time
from itertools import repeat
import cryo as cr
import mrcfile
from pathos.multiprocessing import Pool
from scipy.stats import ortho_group as og
import numpy as np
# number of images
k = 3

# load data
mol = mrcfile.open("zika_153.mrc").data

t0 = time.time()
# random rotation matrices
with Pool() as p:
    orientations = p.map(og.rvs, repeat(3, k))
    images = p.map((lambda R: cr.project_fst(mol, R)), orientations)
    b = cr.reconstruct(images, orientations)
print(time.time()-t0)
