import numpy as np
from functools import reduce
from pathos.multiprocessing import Pool
from scipy.fftpack import next_fast_len
from scipy.interpolate import RegularGridInterpolator as rgi


def project_fst(mol, R, return_hat=False, pad=0):
    """
    simulates EM results given molecule mol and rotation matrix R
    if return_hat is True, returns FT of image
    pad specified number of zeros to pad edges with, default is 0
    """
    # shape of data
    N = next_fast_len(mol.shape[0] + pad)

    # set the frequency range
    if N % 2 == 0:
        w = np.arange(-N / 2, N / 2)
    else:
        w = np.arange(-(N - 1) / 2, (N + 1) / 2)

    # pad as roughly evenly
    mol = np.pad(mol, (int(np.ceil((N - mol.shape[0]) / 2)), int(np.floor((N - mol.shape[0]) / 2))),
                 mode='constant')
    # coordinate grid
    omega = np.meshgrid(w, w, w)

    # Fourier transform
    rho_hat = np.fft.fftshift(np.fft.fftn(mol))
    rho_hat *= np.sign((1 - (np.abs(omega[0] + omega[1] + omega[2]) % 2) * 2)) / N ** 3

    # create sampling grid
    eta_x, eta_y = np.meshgrid(w, w)
    eta_x = eta_x[:, :, None]
    eta_y = eta_y[:, :, None]
    grid = eta_x * R.T[0] + eta_y * R.T[1]

    # interpolation function
    rho_hat_f = rgi((w, w, w), rho_hat, bounds_error=False, fill_value=0)

    # values of data's FFT interpolated to points on slice, tweak dimensions to make broadcasting work
    im_hat = rho_hat_f(grid)[:, :, None]

    # scaling factor
    im_hat *= np.sign((1 - (np.abs(eta_x + eta_y) % 2) * 2)) * N ** 2
    # returns im_hat if return_hat argument is true
    if return_hat:
        return im_hat

    # apply inverse FFT to translate back to original space
    im = np.real(np.fft.ifftn(np.fft.ifftshift(im_hat[:, :, 0])))

    return im


def project_fst_parallel(mol, orientations, return_hat=False, pad=0):
    """
    works like project_fst but takes iterable of rotation matrices and runs in parallel
    does not work on Windows
    """
    # shape of data, apply desired padding and find good size for FFT
    N = next_fast_len(mol.shape[0] + pad)

    # set the frequency range
    if N % 2 == 0:
        w = np.arange(-N / 2, N / 2)
    else:
        w = np.arange(-(N - 1) / 2, (N + 1) / 2)

    # pad roughly evenly
    mol = np.pad(mol, (int(np.ceil((N - mol.shape[0]) / 2)), int(np.floor((N - mol.shape[0]) / 2))),
                 mode='constant')
    # coordinate grid
    omega = np.meshgrid(w, w, w, indexing='ij')

    # Fourier transform
    rho_hat = np.fft.fftshift(np.fft.fftn(mol))
    rho_hat *= np.sign((1 - (np.abs(omega[0] + omega[1] + omega[2]) % 2) * 2)) / N ** 3
    del mol

    # create sampling grid
    eta_x, eta_y = np.meshgrid(w, w, indexing='ij')
    eta_x = eta_x[:, :, None]
    eta_y = eta_y[:, :, None]

    # interpolation function
    rho_hat_f = rgi((w, w, w), rho_hat, bounds_error=False, fill_value=0)

    def project_(R):
        grid = eta_x * R.T[0] + eta_y * R.T[1]

        # values of data's FFT interpolated to points on slice, tweak dimensions to make broadcasting work
        im_hat = rho_hat_f(grid)[:, :, None]

        # scaling factor
        im_hat *= np.sign((1 - (np.abs(eta_x + eta_y) % 2) * 2)) * N ** 2
        # returns im_hat if return_hat argument is true
        if return_hat:
            return im_hat

        # apply inverse FFT to translate back to original space
        im = np.real(np.fft.ifftn(np.fft.ifftshift(im_hat[:, :, 0])))

        # memory saving stuff
        return im

    with Pool() as p:
        images = p.map(project_, orientations)
    return list(images)


def reconstruct(images, orientations):
    N = images[0].shape[0]

    # set the frequency range
    if N % 2 == 0:
        freq_range = np.arange(-N / 2, N / 2)
    else:
        freq_range = np.arange(-(N - 1) / 2, (N + 1) / 2)

    # creating the sample grid
    sample_grid = np.stack(np.meshgrid(freq_range, freq_range, freq_range, indexing='ij'), axis=3)
    eta_x, eta_y = np.meshgrid(freq_range, freq_range, indexing='ij')
    # PSF scaling (take advantage of integer grid, modulo stuff is faster than taking powers)
    im_fft_scale = np.sign((1 - (np.abs(eta_x + eta_y) % 2) * 2)) / (N ** 2)

    # applies FFT, runs in parallel
    with Pool() as p:
        '''
        # use this line on Windows
        images_hat = p.map((lambda im:im_fft_scale * __import__('numpy').fft.fftshift(__import__('numpy').fft.fftn(im))), images)
        '''
        images_hat = p.map((lambda im: im_fft_scale * np.fft.fftshift(np.fft.fftn(im))), images)

    # given FT of image and orientation as pair im_r, returns Fourier space back projection and convolution kernel
    def bp_smear(im_r):
        im_hat, R = im_r
        '''
        # only necessary on windows
        import numpy as np
        N = im_hat.shape[0]
        from scipy.interpolate import RegularGridInterpolator as rgi
        # set the frequency range
        if (N % 2 == 0):
            freq_range = np.arange(-N/2, N/2)
        else:
            freq_range = np.arange(-(N-1)/2, (N+1)/2)

        # creating the sample grid
        sample_grid = np.array(np.meshgrid(freq_range, freq_range, freq_range)).T
        '''
        '''rotating the sample_grid by multiplying by rotation matrix transpose
        Since R is orthonormal, this is the same as inverting
        Multiplying by R transpose gives us the coordinates in the local basis
        '''
        local_x, local_y, local_z = np.tensordot(R.T, sample_grid, axes=(1, 3))
        # smearing on the local_z coordinates
        local_smear = np.sinc(local_z / N)

        # #interpolator for the local_x and local_y coordinates. Interpolates on the FT
        local_interpolator = rgi((freq_range, freq_range), im_hat, bounds_error=False, fill_value=0)
        # local back projection
        local_backproj_hat = local_interpolator(np.stack((local_x, local_y), axis=3)) * local_smear
        print('completed smear')
        return local_backproj_hat, local_smear

    # sum local back projections
    with Pool() as p:
        back_proj_hat, smear = reduce((lambda x, y: (x[0] + y[0], x[1] + y[1])),
                                      p.imap_unordered(bp_smear, zip(images_hat, orientations)))

    # scaling before inverse FT
    #sample_grid = sample_grid.T
    back_proj_hat *= np.sign(1 - ((np.abs(sample_grid[:,:,:,0] + sample_grid[:,:,:,1] + sample_grid[:,:,:,2]) % 2) * 2)) * N ** 3
    if (smear.size - np.count_nonzero(smear)) == 0:
        back_proj_hat /= smear
    print('ready to inverse fourier transform')
    back_proj = np.fft.ifftn(np.fft.ifftshift(back_proj_hat))
    print(np.max(np.imag(back_proj)))
    return np.real(back_proj).astype('float32')


def estimate_orientations(images):
    """
    estimates orientations given list of images
    """
    N = images[0].shape[0]
    if N % 2 == 0:
        grid_range = np.arange(-N / 2, N / 2)
    else:
        grid_range = np.arange(-(N - 1) / 2, (N + 1) / 2)

    angles = np.linspace(0, np.pi, 100)
    im1, im2 = images.pop(), images.pop()

    def find_best(im_i, im_j):
        im_i_f = rgi((grid_range, grid_range), im_i, bounds_error=False, fill_value=0)
        im_j_f = rgi((grid_range, grid_range), im_j, bounds_error=False, fill_value=0)
        # tracks angle pairs and pair quality)
        best_lines = [None, -np.inf]
        for theta in angles:
            for psi in angles:
                li = im_i_f(grid_range[:, None] * np.array([np.cos(theta), np.sin(theta)], None))
                lj = im_j_f(grid_range[:, None] * np.array([np.cos(psi), np.sin(psi)], None))
                forward = np.dot(li / np.linalg.norm(li), lj / np.linalg.norm(lj))
                if forward > best_lines[1]:
                    best_lines[0] = (psi, theta)
                    best_lines[1] = forward
                backward = np.dot(np.flipud(li), lj)
                if backward > best_lines[1]:
                    best_lines[0] = (np.pi + psi, theta)
                    best_lines[1] = backward
        return best_lines[0]

    return find_best(im1, im2)


def estimate_orientations2(images):
    """
    estimates orientations given list of images
    """
    N = images[0].shape[0]
    if N % 2 == 0:
        grid_range = np.arange(-N / 2, N / 2)
    else:
        grid_range = np.arange(-(N - 1) / 2, (N + 1) / 2)

    angles = np.linspace(0, np.pi, 100)
    im1, im2 = images.pop(), images.pop()
    im3 = images[0]

    def find_best(im_i, im_j):
        im_i_f = rgi((grid_range, grid_range), im_i, bounds_error=False, fill_value=0)
        im_j_f = rgi((grid_range, grid_range), im_j, bounds_error=False, fill_value=0)
        # tracks angle pairs and pair quality)
        best_lines = [None, -np.inf]
        for theta in angles:
            for psi in angles:
                li = im_i_f(grid_range[:, None] * np.array([np.cos(theta), np.sin(theta)], None))
                lj = im_j_f(grid_range[:, None] * np.array([np.cos(psi), np.sin(psi)], None))
                forward = np.dot(li / np.linalg.norm(li), lj / np.linalg.norm(lj))
                if forward > best_lines[1]:
                    best_lines[0] = (psi, theta)
                    best_lines[1] = forward
                backward = np.dot(np.flipud(li), lj)
                if backward > best_lines[1]:
                    best_lines[0] = (np.pi + psi, theta)
                    best_lines[1] = backward
        return best_lines[0]

    # finding common lines. The lines are represented by angles in the relative plane (about origin)
    # so l_12 is the angle from the positive x axis in the im1 local coordinates where L12 is.
    # likewise l_21 is the angle from the positive x axis in the im2 local coordinates where L12 is.
    l_12, l_21 = find_best(im1, im2)
    l_13, l_31 = find_best(im1, im3)
    l_23, l_32 = find_best(im2, im3)

    # this is to test a known plane arrangement. Reconstructs correctly
    l_12 = np.pi / 4
    l_13 = 3 * np.pi / 4
    l_21 = 0
    l_23 = np.pi / 2
    l_31 = 0
    l_32 = np.pi / 2

    # finding the lengths on the spherical triangle. Ensures we take the Dihedral angle.
    C = np.abs(l_12 - l_13) % np.pi
    B = np.abs(l_21 - l_23) % np.pi
    A = np.abs(l_31 - l_32) % np.pi
    if A > np.pi / 2:
        A = np.pi - A
    if B > np.pi / 2:
        B = np.pi - B
    if C > np.pi / 2:
        C = np.pi - C

    # solves for the angles in the spherical triangle.
    gamma = np.arccos((np.cos(C) - np.cos(A) * np.cos(B)) / (np.sin(A) * np.sin(B)))
    beta = np.arccos((np.cos(B) - np.cos(A) * np.cos(C)) / (np.sin(A) * np.sin(C)))
    alpha = np.arccos((np.cos(A) - np.cos(B) * np.cos(C)) / (np.sin(B) * np.sin(C)))
    print(A, B, C)
    # print(alpha, beta, gamma)

    # original positioning of im1
    a1 = np.array([1, 0, 0])
    b1 = np.array([0, 1, 0])

    # oriented unit vectors along lines L12 and L13
    V12 = np.array([np.cos(l_12), np.sin(l_12), 0])
    V13 = np.array([np.cos(l_13), np.sin(l_13), 0])

    # rotates l_21 to match up to the x-axis
    l_21_to_xaxis = np.array([[np.cos(-l_21), -np.sin(-l_21), 0, ],
                              [np.sin(-l_21), np.cos(-l_21), 0],
                              [0, 0, 1]])

    # rotates about the x-axis(where l_21 is aligned), by alpha
    rot_xaxis_by_alpha = np.array([[1, 0, 0],
                                   [0, np.cos(alpha), -np.sin(alpha)],
                                   [0, np.sin(alpha), np.cos(alpha)]])

    # rotates the x-axis to l12 (also L12)
    xaxis_to_L12 = np.array([[np.cos(l_12), -np.sin(l_12), 0],
                             [np.sin(l_12), np.cos(l_12), 0],
                             [0, 0, 1]])

    '''these 3 multiplications take the second image, align l_21 with the x-axis,
    rotate by alpha about the x-axis(l_21), and then rotate this rotated plane
    so that l_21 lines up with l_12 and hence L12'''
    F2 = xaxis_to_L12 @ rot_xaxis_by_alpha @ l_21_to_xaxis @ F2

    '''applies the F2 multiplication to V23 (our 3D vector), to track where V23 ends up'''
    V23 = np.array([np.cos(l_23), np.sin(l_23), 0])
    V23 = F2 @ V23

    '''To determine the unique transformation which places im3, we need the unit vectors
    V13, V23, and their local coordinates in the a3, b3 basis (l31, l32)'''
    l_31_to_move = np.array([np.cos(l_31), np.sin(l_31), 0])
    l_32_to_move = np.array([np.cos(l_32), np.sin(l_32), 0])
    F3 = np.column_stack((V13, V23, np.cross(V13, V23))) @ np.linalg.inv(
        np.column_stack((l_31_to_move, l_32_to_move, np.cross(l_31_to_move, l_32_to_move))))

    # these are the angles between L13 and L23, and l31 and l32.
    # These MUST be the same for F3 to be an orthogonal matrix
    print(np.dot(V13, V23) / (np.linalg.norm(V13) * np.linalg.norm(V23)))
    print(np.dot(l_31_to_move, l_32_to_move) / (np.linalg.norm(l_31_to_move) * np.linalg.norm(l_32_to_move)))
    '''At this point we have V12, V13, V23 as where they should be in the space.'''

    return [np.identity(3), F2, F3]
    # return find_best(im1, im2)

'''
generates 3D rotation matrix given three angles u, v, w
for x, y, z axes
'''
def rotation(u, v, w):
    Rx = np.array([[1, 0, 0], [0, np.cos(u), -np.sin(u)], [0,np.sin(u), np.cos(u)]])
    Ry = np.array([[np.cos(v), 0, np.sin(v)], [0, 1, 0], [-np.sin(v), 0, np.cos(v)]])
    Rz = np.array([[np.cos(w), -np.sin(w), 0], [np.sin(w), np.cos(w), 0], [0,0,1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def add_noise(im, r):
    """
    adds Gaussian noise to image im given signal-to-noise ratio r
    """

    # image pixel intensity variance
    im_var = np.var(im)
    # calculate noise variance
    sigma = np.sqrt(im_var / r)
    # generate noise
    noise = np.random.randn(im.shape[0], im.shape[1])
    noise = noise / np.sqrt(np.var(noise)) * sigma
    # return noise
    return im + noise
