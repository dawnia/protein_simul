import numpy as np
import gc
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.stats import ortho_group
from scipy.fftpack import next_fast_len
from pathos.multiprocessing import Pool
from functools import reduce

'''
simulates EM results given molecule mol and rotation matrix R
'''
def project_fst(mol, R, return_hat=False):
    
    # shape of data
    N = next_fast_len(mol.shape[0])
 
    # set the frequency range
    if (N % 2 == 0):
        w = np.arange(-N/2, N/2)
    else:
        w = np.arange(-(N-1)/2, (N+1)/2)
    
    # pad as roughly evenly
    mol = np.pad(mol, (np.ceil((N-mol.shape[0])/2).astype('int'), np.floor((N-mol.shape[0])/2).astype('int')), mode='constant')
    # coordinate grid
    omega = np.meshgrid(w,w,w)
    
    # Fourier transform
    rho_hat = np.fft.fftshift(np.fft.fftn(mol, s=[N,N,N]))
    rho_hat =  rho_hat * np.sign((1 - (np.abs(omega[0] + omega[1] + omega[2]) % 2)*2)) / N**3
    
    # numpy FFT seems to create a lot of mess in memory, explicit garbage collection helps
    gc.collect()
    
    # create sampling grid
    eta_x, eta_y = np.meshgrid(w, w)
    eta_x = np.expand_dims(eta_x, axis=2)
    eta_y = np.expand_dims(eta_y, axis=2)
    grid = eta_x * R.T[0] + eta_y * R.T[1]
    
    # interpolation function
    rho_hat_f = rgi((w,w,w), rho_hat, bounds_error = False, fill_value=0)
   
    # values of data's FFT interpolated to points on slice, tweak dimensions to make broadcasting work
    im_hat = np.expand_dims(rho_hat_f(grid), axis=2)
    
    # scaling factor
    im_hat = np.multiply(im_hat, np.sign((1 - (np.abs(eta_x + eta_y) % 2)*2))) * N * N
    
    # returns im_hat if return_hat argument is true
    if return_hat:
        return im_hat
    
    # apply inverse FFT to translate back to original space
    im = np.real(np.fft.ifftn(np.fft.ifftshift(im_hat[:,:,0])))
    
    # memory saving stuff
    del im_hat
    gc.collect()
    return im

def reconstruct(images, orientations, verbose=True):
    N = images[0].shape[0]

    # set the frequency range
    if (N % 2 == 0):
        freq_range = np.arange(-N/2, N/2)
    else:
        freq_range = np.arange(-(N-1)/2, (N+1)/2)

    # creating the sample grid
    sample_grid = np.array(np.meshgrid(freq_range, freq_range, freq_range)).T
    eta_x, eta_y = np.meshgrid(freq_range, freq_range) 
    # PSF scaling (take advantage of integer frequences, modulo stuff is faster than taking powers)
    im_fft_scale = np.sign((1 - (np.abs(eta_x + eta_y) % 2)*2)) / (N**2)
    gc.collect()

    # applies FFT, runs in parallel
    with Pool() as p:
        images_hat = p.map((lambda im:im_fft_scale * __import__('numpy').fft.fftshift(__import__('numpy').fft.fftn(im))), images)
    # given FT of image and orientation as pair im_r, returns Fourier space back projection and convolution kernel
    def bp_smear(im_r):
        im_hat, R = im_r
        # only necessary on windows
        import numpy as np
        N = im_hat.shape[0]
        
        # set the frequency range
        if (N % 2 == 0):
            freq_range = np.arange(-N/2, N/2)
        else:
            freq_range = np.arange(-(N-1)/2, (N+1)/2)

        # creating the sample grid
        sample_grid = np.array(np.meshgrid(freq_range, freq_range, freq_range)).T
 
        '''rotating the sample_grid by multiplying by rotation matrix transpose
        Since R is orthonormal, this is the same as inverting
        Multiplying by R transpose gives us the coordinates in the local basis
        '''
        local_x, local_y, local_z = np.tensordot(R.T, sample_grid, axes = (1,3))
        #smearing on the local_z coordinates
        local_smear = np.sinc(local_z)
        
        # #interpolator for the local_x and local_y coordinates. Interpolates on the FT
        local_interpolator = rgi((freq_range, freq_range), im_hat, bounds_error = False, fill_value = 0)
        #local back projection
        local_backproj_hat = local_interpolator(np.stack((local_x, local_y), axis = 3)) * local_smear
        print('completed smear')
        return (local_backproj_hat, local_smear)

    # sum local back projections 
    with Pool() as p:
        back_proj_hat, smear = reduce((lambda x,y: (x[0] + y[0], x[1] + y[1])), p.imap_unordered(bp_smear, zip(images_hat, orientations)))
    
    #inverting scalings before inverse FT
    sample_grid = sample_grid.T
    inverse_scalings = np.sign(1-((np.abs(sample_grid[0] + sample_grid[1] + sample_grid[2]) % 2) * 2)) * N**3
    back_proj_hat = inverse_scalings * back_proj_hat
    '''
    if (smear.size - np.count_nonzero(smear)) == 0:
        back_proj_hat = back_proj_hat/smear
    '''
    print('ready to inverse fourier transform')
    back_proj = np.fft.ifftn(np.fft.ifftshift(back_proj_hat))
    print(np.max(np.imag(back_proj)))
    return np.real(back_proj).astype('float32')


'''
estimates orientations given list of images
'''
def estimate_orientations(images):
    N = images[0].shape[0]
    if (N % 2 == 0):
        grid_range = np.arange(-N/2, N/2)
    else:
        grid_range = np.arange(-(N-1)/2, (N+1)/2)

    angles = np.linspace(0,np.pi, 100)
    im1, im2 = images.pop(), images.pop()

    def find_best(im_i, im_j):
        im_i_f = rgi((grid_range, grid_range), im_i, bounds_error=False, fill_value=0)
        im_j_f = rgi((grid_range, grid_range), im_j, bounds_error=False, fill_value=0)
        # tracks angle pairs and pair quality)
        best_lines = [None, -np.inf]
        for theta in angles:
            for psi in angles:
                li = im_i_f(grid_range[:,None] * np.array([np.cos(theta), np.sin(theta)],None))
                lj = im_j_f(grid_range[:,None] * np.array([np.cos(psi), np.sin(psi)],None))
                forward = np.dot(li/np.linalg.norm(li), lj/np.linalg.norm(lj))
                if forward > best_lines[1]:
                   best_lines[0] = (psi, theta)
                   best_lines[1] = forward
                backward = np.dot(np.flipud(li), lj)
                if backward > best_lines[1]:
                   best_lines[0] = (np.pi + psi, theta)
                   best_lines[1] = backward
        return best_lines[0]
    return find_best(im1, im2)    


'''
adds Gaussian noise to image im given signal-to-noise ratio r
'''
def add_noise(im, r, mode='power'):
    mu = np.mean(im)
    # calculate noise variance
    sigma = mu / r
    # return noised image
    return im + sigma * np.random.randn(im.shape[0], im.shape[1])

