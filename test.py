import mrcfile
from cryo import *

zika_file = mrcfile.open('zika_153.mrc')
rho = zika_file.data

''' this code just makes a cube to visually test things one'''
cube = np.zeros((100, 100, 100))
for i in range(40, 60):
    for m in range(40, 60):
        for k in range(40, 60):
            cube[i,m,k] = 1
#
# xy_proj = cube[:,:, 100]
smol_square = np.zeros((200, 200))
for i in range(75, 125):
    for m in range(75, 125):
        smol_square[i,m] = 1

''' Sample known orientations for testing projections '''
I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
R = np.array([[0, 0, 1],[1, 0, 0], [0, 1, 0]])

#very small tilts
small_tilt_a = np.array([1, -.005, 0.1])
small_tilt_b = np.array([-.005, 1, 0.1])
small_tilt_a = small_tilt_a / np.linalg.norm(small_tilt_a)
small_tilt_b = small_tilt_b / np.linalg.norm(small_tilt_b)
small_tilt_c = np.cross(small_tilt_a, small_tilt_b) / np.linalg.norm(np.cross(small_tilt_a, small_tilt_b))
small_tilt = np.column_stack((small_tilt_a, small_tilt_b, small_tilt_c))

#plane rotation
plane_rot = np.array([[np.sqrt(2) / 2, np.sqrt(2) / 2, 0], [-np.sqrt(2)/2, np.sqrt(2) / 2, 0], [0,0,1]])

a = np.array([-1.5, 0.5, 1])
b = np.array([0.15430335, -0.77151675, 0.6172134])
a = a / np.linalg.norm(a)
b = b / np.linalg.norm(b)
R3 = np.column_stack((a, b, np.cross(a,b) / np.linalg.norm(np.cross(a,b))))
# R2 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
#K is number of images to take
K = 0
images = []
orientations = []
for i in range(K):
    mat = ortho_group.rvs(3)
    orientations.append(mat)
    images.append(project_fst(rho, mat))
    
'''tests identity rotation'''
# awk = reconstruct([project_fst(rho, I)], [I])

'''runs reconstruction all of the images'''
# awk = reconstruct(images, orientations)

'''Saves them to the mrc files test.mrc'''
# with mrcfile.new('test2.mrc', overwrite = True) as mrc:
#     mrc.set_data(np.float32(awk))

# with mrcfile.open('test.mrc') as mrc:
#     mrc.data


