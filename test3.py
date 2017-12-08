import time
import mrcfile
import cryo as cr
from itertools import repeat
from scipy.stats import special_ortho_group as og

# number of images
k = 1000

# load data
mol = mrcfile.open("zika_153.mrc").data

t0 = time.time()
# random rotation matrices
orientations = list(map(og.rvs, repeat(3, k)))
images = cr.project_fst_parallel(mol, orientations)

b = cr.reconstruct(images, orientations)
with mrcfile.new('zika_reconstruction_1.mrc', overwrite=True) as mrc:
    mrc.set_data(b)

print("Completed reconstruction from", k, "projections in", time.time() - t0)
