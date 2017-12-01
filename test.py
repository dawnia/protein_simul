import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import ortho_group
from matplotlib import pyplot as plt
import time
import mrcfile
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





def project_fst(mol, rot_mat):
    #FT (3d)
    rho_fft = np.fft.fftshift(np.fft.fftn(mol))
    N = mol.shape[0]
    #making the frequency range
    if (N % 2 == 0):
        freq_range = np.arange(-N/2, N/2)
    else:
        freq_range = np.arange(-(N-1)/2, (N+1)/2)

    #building the sampling plane (unrotated)
    [etas_x, etas_y] = np.meshgrid(freq_range, freq_range)
    etas_x = etas_x[..., np.newaxis]
    etas_y = etas_y[..., np.newaxis]

    #plane_vals is our rotated plane to sample on
    plane_vals = etas_x*rot_mat[:,0] + etas_y*rot_mat[:,1]

    #used to scale our FT according to PSF before interpolating
    scalings = np.stack(np.meshgrid(freq_range, freq_range, freq_range), axis = 3)
    scalings = np.sum(scalings, axis = 3)*np.pi*1j
    scalings = np.real(np.exp(scalings))* (1/(N**3))
    rho_fft = rho_fft*scalings

    #Two interpolators for the FT, one for real and one for imaginary
    FT_Re_interpolator = RegularGridInterpolator((freq_range, freq_range, freq_range), np.real(rho_fft), bounds_error = False, fill_value = 0)
    FT_Im_interpolator = RegularGridInterpolator((freq_range, freq_range, freq_range), np.imag(rho_fft), bounds_error = False, fill_value = 0)

    #interpolating the FT on our rotated grid
    plane_samples = FT_Re_interpolator(plane_vals) + FT_Im_interpolator(plane_vals)*1j

    #inverse scaling our FT according to PSF before interpolating
    inverse_scalings = np.stack(np.meshgrid(freq_range,freq_range), axis = 2)
    inverse_scalings = np.sum(inverse_scalings, axis = 2)*np.pi*1j*(-1)
    inverse_scalings = np.real(np.exp(inverse_scalings))*(N**2)
    plane_samples = inverse_scalings*plane_samples

    #inverse FT (2d)
    plane_image = np.fft.ifftn(np.fft.ifftshift(plane_samples))

    #printing largest imaginary value post inverse FT (should be very small)
    print(np.max(np.abs(np.imag(plane_image))))
    return np.real(plane_image)

def project_fst_padded(mol, rot_mat):
    #padding the image by 50 zeros on all sides
    mol = np.pad(mol, (50, 50), 'constant', constant_values = (0))

    #FT (3d)
    rho_fft = np.fft.fftshift(np.fft.fftn(mol))
    N = mol.shape[0]

    #making the frequency range
    if (N % 2 == 0):
        freq_range = np.arange(-N/2, N/2)
    else:
        freq_range = np.arange(-(N-1)/2, (N+1)/2)

    #making the sampling plane (unrotated)
    [etas_x, etas_y] = np.meshgrid(freq_range, freq_range)
    etas_x = etas_x[..., np.newaxis]
    etas_y = etas_y[..., np.newaxis]

    #plane_vals is our rotated plane to sample on
    plane_vals = etas_x*rot_mat[:,0] + etas_y*rot_mat[:,1]

    #scaling FT according to PSF before interpolating
    scalings = np.stack(np.meshgrid(freq_range, freq_range, freq_range), axis = 3)
    scalings = np.sum(scalings, axis = 3)*np.pi*1j
    scalings = np.exp(scalings)* (1/(N**3))
    rho_fft = rho_fft*scalings

    #interpolators for the FT
    FT_Re_interpolator = RegularGridInterpolator((freq_range, freq_range, freq_range), np.real(rho_fft), bounds_error = False, fill_value = 0)
    FT_Im_interpolator = RegularGridInterpolator((freq_range, freq_range, freq_range), np.imag(rho_fft), bounds_error = False, fill_value = 0)

    #interpolating the FT on our sampling grid
    plane_samples = FT_Re_interpolator(plane_vals) + FT_Im_interpolator(plane_vals)*1j

    #inverse scaling our FT before inverse FT
    inverse_scalings = np.stack(np.meshgrid(freq_range,freq_range), axis = 2)
    inverse_scalings = np.sum(inverse_scalings, axis = 2)*np.pi*1j*(-1)
    inverse_scalings = np.exp(inverse_scalings)*(N**2)
    plane_samples = inverse_scalings*plane_samples

    #nverse FT (2d)
    plane_image = np.fft.ifftn(np.fft.ifftshift(plane_samples))

    #printing largest imaginary value post inverse FT (should be very small)
    print(np.max(np.abs(np.imag(plane_image))))
    return np.real(plane_image)[50:203, 50:203]

#K is number of images to take
K = 0
images = []
orientations = []
for i in range(K):
    mat = ortho_group.rvs(3)
    orientations.append(mat)
    images.append(project_fst_padded(rho, mat))

def reconstruct(images, orientations):
    #pad each image
    for i in range(len(images)):
        images[i] = np.pad(images[i], (25, 25), 'constant', constant_values = (0))
    N = images[0].shape[0]

    #set the frequency range
    if (N % 2 == 0):
        freq_range = np.arange(-N/2, N/2)
    else:
        freq_range = np.arange(-(N-1)/2, (N+1)/2)

    #creating the sample grid
    sample_grid = np.stack(np.meshgrid(freq_range, freq_range, freq_range), axis = 3)

    #taking the FT of each image
    images_hat = []
    image_fft_scalings = np.stack(np.meshgrid(freq_range, freq_range), axis = 2)
    image_fft_scalings = np.sum(image_fft_scalings, axis = 2)*np.pi*1j
    image_fft_scalings = np.exp(image_fft_scalings)*(1/(N**2))
    for i in range(len(images)):
        images_hat.append(image_fft_scalings*np.fft.fftshift(np.fft.fftn(images[i])))

    #two lists, one to store all of the individual back projections, one to store all of the smears
    local_backproj_hat_list = []
    local_smear_list = []


    for i in range(len(images)):
        '''rotating the sample_grid by multiplying by rotation matrix transpose
        Since R is orthonormal, this is the same as inverting
        Multiplying by R transpose gives us the coordinates in the local basis
        '''
        local_x, local_y, local_z = np.tensordot(np.transpose(orientations[i]), sample_grid, axes = (1,3))

        #local_x, local_y, local_z are each N x N x N which store the coordinates
        local_coords_grid = np.stack([local_x, local_y, local_z], axis = 3)

        #smearing on the local_z coordinates
        local_smear = np.sinc(local_z)

        #interpolators for the local_x and local_y coordinates. Interpolatres on the FT
        local_real_interpolator = RegularGridInterpolator((freq_range, freq_range), np.real(images_hat[i]), bounds_error = False, fill_value = 0)
        local_imag_interpolator = RegularGridInterpolator((freq_range, freq_range), np.imag(images_hat[i]), bounds_error = False, fill_value = 0)

        #combining local_x and local_y to a plane to run the interpolators on
        local_plane = np.stack((local_x, local_y), axis = 3)

        #interpolated values on the plane
        local_image_hat = local_real_interpolator(local_plane) + local_imag_interpolator(local_plane)*1j

        #local back projection
        local_backproj_hat = local_image_hat*local_smear

        #adding local back projection and local smears to the lists.
        local_backproj_hat_list.append(local_backproj_hat)
        local_smear_list.append(local_smear)
        print('smear no.' + ' ' + str(i+1))

    print('done smearing')

    #summing the back projections
    back_proj_hat = np.sum(np.array(local_backproj_hat_list), axis = 0)

    #summing the back projections with scaling
    # back_proj_hat = np.sum(np.array(local_backproj_hat_list), axis = 0) / np.sum(np.array(local_smear_list), axis = 0)

    '''loop to replace any zeros from the scaling with 0 elements after division'''
    # for i in range(N):
    #     for m in range(N):
    #         for k in range(N):
    #             if (np.isinf(back_proj_hat[i,m,k])):
    #                 back_proj_hat[i,m,k] = 0


    #inverting scalings before inverse FT
    inverse_scalings = np.stack(np.meshgrid(freq_range,freq_range, freq_range), axis = 3)
    inverse_scalings = np.sum(inverse_scalings, axis = 3)*np.pi*1j*(-1)
    inverse_scalings = np.exp(inverse_scalings)*(N**3)
    back_proj_hat = inverse_scalings*back_proj_hat

    print('ready to inverse fourier transform')
    back_proj = np.fft.ifftn(np.fft.ifftshift(back_proj_hat))
    print(np.max(np.imag(back_proj)))
    return np.real(back_proj)[25:178, 25:178, 25:178]

'''tests identity rotation'''
# awk = reconstruct([project_fst(rho, I)], [I])

'''runs reconstruction all of the images'''
# awk = reconstruct(images, orientations)

'''Saves them to the mrc files test.mrc'''
# with mrcfile.new('test.mrc', overwrite = True) as mrc:
#     mrc.set_data(np.float32(awk))




# with mrcfile.open('test.mrc') as mrc:
#     mrc.data


















































# def project_fst_pad(mol, rot_mat):
#     N = mol.shape[0]
#     pad_width = int(np.ceil((np.ceil(np.sqrt(3)*N) - N) / 2))
#     N_padded = N + 2*pad_width
#     padded_mol = np.lib.pad(mol, (pad_width,), 'constant', constant_values = (0))
#     rho_fft = np.fft.fftshift(np.fft.fftn(padded_mol))
#     [etas_x, etas_y] = np.meshgrid(np.arange(-(N/2) , (N/2) ),np.arange(-(N/2) , (N/2) ))
#     etas_x = etas_x[..., np.newaxis]
#     etas_y = etas_y[..., np.newaxis]
#     plane_vals = etas_x*rot_mat[:,0] + etas_y*rot_mat[:,1]
#     FT_Re_interpolator = RegularGridInterpolator((np.arange(-N_padded/2, N_padded/2), np.arange(-N_padded/2, N_padded/2), np.arange(-N_padded/2, N_padded/2)), np.real(rho_fft))
#     FT_Im_interpolator = RegularGridInterpolator((np.arange(-N_padded/2, N_padded/2), np.arange(-N_padded/2, N_padded/2), np.arange(-N_padded/2, N_padded/2)), np.imag(rho_fft))
#     plane_samples = FT_Re_interpolator(plane_vals) + FT_Im_interpolator(plane_vals)*np.array([1j])
#     plane_image = np.fft.ifftn(np.fft.ifftshift(plane_samples))
#     print(np.max(np.abs(np.imag(plane_image))))
#     return np.real(np.fft.ifftn(np.fft.ifftshift(plane_samples)))
#
# def project_fst_smolplane(mol, rot_mat):
#     rho_fft = np.fft.fftshift(np.fft.fftn(mol))
#     N = mol.shape[0]
#     N_smol = np.floor(N / np.sqrt(3)) - 1
#     [etas_x, etas_y] = np.meshgrid(np.arange(-(N_smol/2) , (N_smol/2) ),np.arange(-(N_smol/2) , (N_smol/2) ))
#     etas_x = etas_x[..., np.newaxis]
#     etas_y = etas_y[..., np.newaxis]
#     plane_vals = etas_x*rot_mat[:,0] + etas_y*rot_mat[:,1]
#     FT_Re_interpolator = RegularGridInterpolator((np.arange(-N/2, N/2), np.arange(-N/2, N/2), np.arange(-N/2, N/2)), np.real(rho_fft))
#     FT_Im_interpolator = RegularGridInterpolator((np.arange(-N/2, N/2), np.arange(-N/2, N/2), np.arange(-N/2, N/2)), np.imag(rho_fft))
#     plane_samples = FT_Re_interpolator(plane_vals) + FT_Im_interpolator(plane_vals)*np.array([1j])
#     plane_image = np.fft.ifftn(np.fft.ifftshift(plane_samples))
#     print(np.max(np.abs(np.imag(plane_image))))
#     return np.real(np.fft.ifftn(np.fft.ifftshift(plane_samples)))



#sequence of tilts
# increment = 0.01
# count = 0
# list_of_images = []
# while increment < .2:
#     current_a = np.array([1, -(increment**2)/2, increment])
#     current_b = np.array([-(increment**2)/2, 1, increment])
#     current_a = current_a / np.linalg.norm(current_a)
#     current_b = current_b / np.linalg.norm(current_b)
#     current_c = np.cross(current_a, current_b) / np.linalg.norm(np.cross(current_a, current_b))
#     current_tilt = np.column_stack((current_a, current_b, current_c))
#     list_of_images.append(project_fst(cube, current_tilt))
#     increment += .01
#     count += 1
#
# for i in list_of_images:
#     plt.imshow(i)
#     plt.show()
#     plt.close("all")
#     time.sleep(1)


















# sample = np.zeros((N, N), dtype = complex)
# for i in range(N):
#     for j in range(N):
#         if np.max(plane_vals[i,j]) < N/2 - 1 and np.min(plane_vals[i,j]) >= -N/2:
#             sample[i,j] = complex(FT_Re_interpolator(plane_vals[i,j])[0], FT_Im_interpolator(plane_vals[i,j])[0])
#             # print(i,j)
# proj = np.real(np.fft.ifftn(np.fft.ifftshift(sample)))
