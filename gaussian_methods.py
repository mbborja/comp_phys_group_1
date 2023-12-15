import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_coherent_wrinkle(resolution,zoom):
    """Plots an example of a gaussian random field that might affect a ground based
        satelite. this actually finicky so here is code and ratios that does this
        Args:
        resolution (int): size of grid you want to look at
        zoom (int): how much you want to plot and look at of psf, full resolution is too large
        
        returns: telescope class
    """
    center = int(resolution/2)
    aperture_diameter = resolution/10
    kc = resolution/5
    alpha = -4
    func = lambda k: k**alpha*np.exp(-(k**2)/kc**2) 
    noise = resolution*gaussian_random_field(func,resolution).real
    tel = telescope_gaussian(resolution,aperture_diameter,noise)
    
    zoom_noise = tel.psf_noisy[(center-zoom):(center+zoom),(center-zoom):(center+zoom)]
    fig = plt.figure()
    ax1 = plt.subplot(1,3,1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.imshow(np.log10(zoom_noise))
    plt.scatter(zoom,zoom)
    ax2 = plt.subplot(1,3,2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.add_patch(Circle((1000,1000), radius=resolution/20, fill=None, lw=1, edgecolor = 'black'))
    ax2.legend(['Area of effect on aperture'], loc='upper right', fontsize=5.5)
    plt.imshow(noise)
    fig.tight_layout()
    plt.show()
    return tel

class telescope_gaussian:
    '''generates a telescope with specified noise and circular aperture
        Atrributes:
        
        resolution (int): Resolution of the telescope.
        grid_dim (tuple): Dimensions of the grid used for calculations.
        aperture_diameter (float): Diameter of the telescope's aperture.
        aperture (numpy.ndarray): Circular aperture mask.
        
        psf (numpy.ndarray): Point Spread Function (PSF) calculated using FFT.
        noise (numpy.ndarray): Grid of resolution X resolution dim
        aperture_noisy (numpy.ndarray): aperture multiplied by a complex phase determined by noise
        psf_noisy (numpy.ndarray): Point Spread Function (PSF) calculated using FFT.
   
    '''
    def __init__(self,resolution,aperture_diameter,noise):
        self.aperture = []
        self.noise = []
        
        
        self.resolution = resolution
        self.grid_dim = (self.resolution,self.resolution)
        self.aperture_diameter = aperture_diameter
        
        self.aperture = create_circular_aperture(self.grid_dim,self.aperture_diameter)
        self.psf = pointspread(self.aperture)
        
        self.aperture_noisy = self.aperture*np.exp(1j*noise)
        self.psf_noisy = pointspread(self.aperture_noisy)                           
    
        
def pointspread(aperture):
    """Calculate Point Spread Function (PSF) and meshgrids using FFT.
    
    Args:
        aperture (numpy.ndarray): Aperture mask.
        wavelength (float): Wavelength of light.

    Returns: PSF (np.ndarray)

    """
    aperture_fourier = np.fft.fft2(aperture)
    aperture_fourier = np.fft.fftshift(aperture_fourier)
    aperture_fourier = np.abs(aperture_fourier)
    psf = aperture_fourier**2
    return psf

def create_circular_aperture(grid_dim, diameter):
    """Create a circular aperture mask.

    Args:
        grid_dim (tuple): Dimensions of the grid.
        diameter (float): Diameter of the aperture.

    Returns:
        numpy.ndarray: Circular aperture mask.

    """
    # a cute little way to make a circular aperture for our telescope
    aperture = np.zeros(grid_dim)
    center_x, center_y = grid_dim[0] // 2, grid_dim[1] // 2
    y, x = np.ogrid[:grid_dim[0], :grid_dim[1]]
    mask = (x - center_x)**2 + (y - center_y)**2 <= (diameter / 2)**2
    aperture[mask] = 1
    return aperture

def fftIndgen(n):
    """creates indices in k space for gaussian random field creation
        Arg:
            n : grid dim for the field space you are making
    
        Returns np.ndarray: [0, -1, ..., -n/2, -n/2 + 1, ..., -1] 
    """
    # function just for indices in the gaussian random field function below
    # nice to have a function for this separate from the code
    a = range(0, int(n/2+1))
    a = [-i for i in a]
    b = reversed(range(1, int(n/2)))   
    b = [-i for i in b]
    return a + b


def gaussian_random_field(Pk = lambda k : k**-3.0, size = 100):
    """generates a gaussian random field
        Args:
        Pk: function of k that specifies the power spectrum that you want
        
        size (int): size of grid you want to specify
        
        Returns (np.ndarray: size x size gaussian random field
    """
    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx**2 + ky**2))) 
    noise = np.fft.fft2(np.random.normal(size = (size, size)))
    amplitude = np.zeros((size,size))
    for i, kx in enumerate(fftIndgen(size)):
        for j, ky in enumerate(fftIndgen(size)):            
            amplitude[i, j] = Pk2(kx, ky)              
    return np.fft.ifft2(noise * amplitude) # return


def generate_example_fields(size):
    """generate and plot a collection of gaussian random fields with power spectra
        of the form k**2*exp(-(k/kc)**2)
        
        Arg:
        size (int): grid size specified
    """
    for alpha in [-4.0,-3,-2]:
        fig = plt.figure()
        
        for n, kc in enumerate([1,10,30,100]):
            ax = plt.subplot(1, 4, n + 1)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            func = lambda k: k**alpha*np.exp(-(k)**2/kc**2) #user specified lambda k
            out = gaussian_random_field(Pk = func, size=size)
            fig.tight_layout()
            ax.imshow(out.real, interpolation='none')
    plt.show()
            
def plot_imperfections(resolution,zoom, noise_strength):
    """Plots an aperture affected by a white noise phase shift and the noise affecting it
        Args:
        resolution (int): dimensionality of system
        zoom (int): how closely you want to zoom in on the psf in plots
        noise_strength (float): how much more aggressive do you want the noise to be
        
        returns: telescope class
    """
    center = int(resolution/2)
    aperture_diameter = resolution/8
    kc = resolution/20
    alpha = 0
    func = lambda k: k**alpha*np.exp(-(k**2)/kc**2) 
    noise = gaussian_random_field(func,resolution).real*noise_strength
    tel = telescope_gaussian(resolution,aperture_diameter,noise)    
    
    #noise_zoom = noise[(center-zoom):(center+zoom),(center-zoom):(center+zoom)]
    noisy_psf_zoom = tel.psf_noisy[(center-zoom):(center+zoom),(center-zoom):(center+zoom)]
    
    fig = plt.figure()
    ax1 = plt.subplot(1,3,1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.imshow(np.log10((noisy_psf_zoom)))
    ax2 = plt.subplot(1,3,2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.imshow(np.sqrt(noisy_psf_zoom))
    fig.colorbar(mappable=None, label='Pixel Intensity (Normalized to 1)', shrink=0.43)
    fig.tight_layout()
    plt.show()
    return tel



def generate_noise_average(resolution,zoom,N):
    """simulate a long exposure by taking many realizations of a gaussian random field
        Args:
        resolution (int): grid size
        zoom (int): how closely to look at the psf
        N (int): number of realizations to average over
        
        returns: averaged over psf (np.ndarray)
    """
    center = int(resolution/2)
    aperture_diameter = resolution/10
    kc = resolution/5
    alpha = -4
    func = lambda k: k**alpha*np.exp(-(k**2)/kc**2) 
    average = np.empty([N,resolution,resolution])
    for i in np.arange(N):
        noise = resolution*gaussian_random_field(func,resolution).real
        tel = telescope_gaussian(resolution,aperture_diameter,noise)
        average[i] = tel.psf_noisy
    
    average_psf = sum((average))/N
    zoom_psf = tel.psf_noisy[(center-zoom):(center+zoom),(center-zoom):(center+zoom)]
    zoom_average = average_psf[(center-zoom):(center+zoom),(center-zoom):(center+zoom)]
    
    fig = plt.figure()
    ax1 = plt.subplot(1,2,1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.imshow(np.log10(zoom_psf))
    plt.scatter(zoom,zoom)
    plt.title('Noisy psf')
    ax2 = plt.subplot(1,2,2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.imshow(np.log10(zoom_average))
    plt.scatter(zoom,zoom)
    plt.title('Averaged psf')
    fig.tight_layout()
    plt.show
    return average

