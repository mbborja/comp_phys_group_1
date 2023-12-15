import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.special import j1
from scipy.ndimage import zoom
import scipy.integrate as integrate

def clipped_zoom(img, zoom_factor, **kwargs):
    """Zoom in or out on an image while handling boundary conditions.

    Args:
        img (numpy.ndarray): Input image.
        zoom_factor (float): Scaling factor for zooming.
        **kwargs: Additional arguments for the zoom function.

    Returns:
        numpy.ndarray: Zoomed image.

    """
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def pointspread(aperture,wavelength):
    """Calculate Point Spread Function (PSF) and meshgrids using FFT.

    Args:
        aperture (numpy.ndarray): Aperture mask.
        wavelength (float): Wavelength of light.

    Returns:
        tuple: PSF, kx meshgrid, ky meshgrid.

    """
    aperture_fourier = np.fft.fft2(aperture)
    aperture_fourier = np.fft.fftshift(aperture_fourier)
    aperture_fourier = np.abs(aperture_fourier)
    kx, ky = np.meshgrid(np.fft.fftfreq(aperture.shape[1]) / wavelength, \
                        np.fft.fftfreq(aperture.shape[0]) / wavelength)
    psf = aperture_fourier**2
    return psf, kx,ky

def generate_airy_disk_intensity_focal(grid_dim, wavelength, aperture_diameter, focal_length):
    
    """Generate the intensity distribution of an Airy disk.

    Args:
        grid_dim (tuple): Dimensions of the grid.
        wavelength (float): Wavelength of light.
        aperture_diameter (float): Diameter of the aperture.
        focal_length (float): Focal length of the telescope.

    Returns:
        numpy.ndarray: Intensity values for the given coordinates.

    """
    
    y, x = np.ogrid[:grid_dim[0], :grid_dim[1]]
    center_x, center_y = grid_dim[0] // 2, grid_dim[1] // 2
    r = np.sqrt((y-center_y)**2 + (x-center_x)**2)
    k = 2 * np.pi / wavelength
    x = k * (aperture_diameter/2) * r / focal_length
    airy_disk = (2*j1(x) / (x))**2
    
    # Identify NaN values in the array
    nan_indices = np.isnan(airy_disk)

    # Replace NaN values with 1
    airy_disk[nan_indices] = 1
    return airy_disk

def generate_airy_cassegrain(grid_dim, wavelength, aperture_diameter, focal_length, obs_ratio):
    """Generate the intensity distribution of an cassegrain telescope.

    Args:
        grid_dim (tuple): Dimensions of the grid.
        wavelength (float): Wavelength of light.
        aperture_diameter (float): Diameter of the aperture.
        focal_length (float): Focal length of the telescope.

    Returns:
        numpy.ndarray: Intensity values for the given coordinates.

    """
    eps = obs_ratio #obscuration_ratio
    y, x = np.ogrid[:grid_dim[0], :grid_dim[1]]
    center_x, center_y = grid_dim[0] // 2, grid_dim[1] // 2
    r = np.sqrt((y-center_y)**2 + (x-center_x)**2)
    k = 2 * np.pi / wavelength
    x = k * (aperture_diameter/2) * r / focal_length
    airy_disk = (1/((1-eps**2)**2))*((2*j1(x) / (x)) - ((2*eps*j1(eps*x))/(x)))**2
    
    # Identify NaN values in the array
    nan_indices = np.isnan(airy_disk)

    # Replace NaN values with 1
    airy_disk[nan_indices] = 1
    return airy_disk

def create_circular_aperture(grid_dim,aperture_diameter):
    """Create a circular aperture mask.

    Args:
        grid_dim (tuple): Dimensions of the grid.
        aperture_diameter (float): Diameter of the aperture.

    Returns:
        numpy.ndarray: Circular aperture mask.

    """
    # a cute little way to make a circular aperture for our telescope
    aperture = np.zeros(grid_dim)
    center_x, center_y = grid_dim[0] // 2, grid_dim[1] // 2
    y, x = np.ogrid[:grid_dim[0], :grid_dim[1]]
    mask = (x - center_x)**2 + (y - center_y)**2 <= (aperture_diameter / 2)**2
    aperture[mask] = 1
    return aperture

def generate_normalized_fft(psf, zoom_factor):
    """Generate the normalized and rescaled version of the PSF.
    This allows for 
    
    Args:
        psf (numpy.ndarray): Point Spread Function.
        zoom_factor (float): Zoom factor.

    Returns:
        numpy.ndarray: Normalized FFT.

    """
    zoom = clipped_zoom(psf, zoom_factor)
    normalized_zoom = zoom/np.max(zoom)
    return normalized_zoom

def generate_residuals(airy, normalized_fft):
    """Generate the residuals between Airy disk and normalized FFT.

    Args:
        airy (numpy.ndarray): Intensity distribution of the Airy disk.
        normalized_fft (numpy.ndarray): Normalized FFT.

    Returns:
        numpy.ndarray: Residuals.

    """
    residuals = airy - normalized_fft
    return residuals

def plot_center_slice(array):
    """Plot the center row and column of a 2D array.

    Args:
        array (numpy.ndarray): Input 2D array.

    """
    # Extract the center row (or column)
    center_row = array.shape[0] // 2
    center_col = array.shape[1] // 2

    center_row_slice = array[center_row, :]
    center_col_slice = array[:, center_col]

    # Plot the center row
    plt.plot(center_row_slice)
    plt.title('Center Row')
    plt.xlabel('y (m)')
    plt.ylabel('Pixel Intensity (Normalized to 1)')
    plt.show()

    # Plot the center column
    plt.plot(center_col_slice)
    plt.title('Center Column')
    plt.xlabel('x (m)')
    plt.ylabel('Pixel Intensity (Normalized to 1)')
    plt.show()

class telescope: 
    """Class representing a telescope and its properties.

    Attributes:
        resolution (int): Resolution of the telescope.
        grid_dim (tuple): Dimensions of the grid used for calculations.
        wavelength (float): Wavelength of light.
        focal_length (float): Focal length of the telescope.
        aperture_diameter (float): Diameter of the telescope's aperture.
        first_zero_diameter (float): Diameter of the first zero of the Airy disk.
        airy (numpy.ndarray): Intensity distribution of the Airy disk.
        aperture (numpy.ndarray): Circular aperture mask.
        psf (numpy.ndarray): Point Spread Function (PSF) calculated using FFT.
        kx (numpy.ndarray): Meshgrid for x-coordinate in PSF calculation.
        ky (numpy.ndarray): Meshgrid for y-coordinate in PSF calculation.
        zoom_factor (float): Zoom factor used in FFT calculation.
        normalized_fft (numpy.ndarray): Normalized FFT of the PSF.
        residuals (numpy.ndarray): Residuals between the Airy disk and FFT-based PSF.

    Methods:
        pointspread(aperture, wavelength): Calculate PSF and meshgrids using FFT.
        plot_normalized_fft(): Plot the normalized FFT of the PSF.
        plot_airy_disk(): Plot the Airy disk intensity distribution.
        plot_residuals(): Plot the residuals between two PSF calculation methods.
        plot_final(): Plot a comparison of Airy disk, normalized FFT, and residuals.
        plot_aperture_size_comparison(): Plot a comparison of FFT and Airydisk at a certain aperture size
        plot_residual_slice(): Plot a center slice of the residuals.

    """   
    def __init__(self,resolution,wavelength,focal_length,aperture_diameter):
        self.aperture = []
        self.noise = []
        self.psf_noisy = []
        self.j1 = []
        
        self.resolution = resolution
        self.grid_dim = (self.resolution,self.resolution)
        self.wavelength = wavelength
        self.focal_length = focal_length
        self.aperture_diameter = aperture_diameter
        
        self.first_zero_diameter = 1.22 * self.focal_length * self.wavelength / (self.aperture_diameter / 2)
        
        self.airy = generate_airy_disk_intensity_focal(self.grid_dim,self.wavelength,self.aperture_diameter,self.focal_length)
        self.casse = generate_airy_cassegrain(self.grid_dim,self.wavelength,self.aperture_diameter,self.focal_length, obs_ratio=1/6)
        self.aperture = create_circular_aperture(self.grid_dim,self.aperture_diameter)
        
        self.psf, self.kx, self.ky = pointspread(self.aperture, self.wavelength)
        
        self.zoom_factor = self.wavelength*self.focal_length/self.resolution
        
        self.normalized_fft = generate_normalized_fft(self.psf, self.zoom_factor)

        self.residuals = generate_residuals(self.airy, self.normalized_fft)
        
    def pointspread(aperture,wavelength):
        aperture_fourier = np.fft.fft2(aperture)
        aperture_fourier = np.fft.fftshift(aperture_fourier)
        aperture_fourier = np.abs(aperture_fourier)
        kx, ky = np.meshgrid(np.fft.fftfreq(aperture.shape[1]) / wavelength, \
                            np.fft.fftfreq(aperture.shape[0]) / wavelength)
        psf = aperture_fourier**2
        return psf, kx,ky
        
    def plot_normalized_fft(self):
        plt.imshow(np.log10(self.normalized_fft))
        plt.title("Point Spread Function Calculated using FFT")
        cbar = plt.colorbar(label='Log of (Pixel Intensity (Normalized to 1))')
        aperture_circle = plt.Circle((self.resolution/2, self.resolution/2), self.aperture_diameter/2, color='blue', fill=False, linestyle='solid', label='Aperture Circle')
        plt.gca().add_patch(aperture_circle)
        first_zero_circle = plt.Circle((self.resolution/2,self.resolution/2), self.first_zero_diameter/2, color='red', fill=False, linestyle='dashed', label='First Zero of Airy Disk')
        plt.gca().add_patch(first_zero_circle)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.legend()
        plt.show()
    
    def plot_airy_disk(self):
        plt.imshow(np.log10(self.airy))
        aperture_circle = plt.Circle((self.resolution/2, self.resolution/2), self.aperture_diameter/2, color='blue', fill=False, linestyle='solid', label='Aperture Circle')
        cbar = plt.colorbar(label='Log of (Pixel Intensity (Normalized to 1))')
        plt.gca().add_patch(aperture_circle)
        first_zero_circle = plt.Circle((self.resolution/2,self.resolution/2), self.first_zero_diameter/2, color='red', fill=False, linestyle='dashed', label='First Zero of Airy Disk')
        plt.gca().add_patch(first_zero_circle)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Airy Disk PSF")
        plt.legend()
        plt.show()
        
    def plot_casse_disk(self):
        plt.imshow(np.log10(self.casse))
        aperture_circle = plt.Circle((self.resolution/2, self.resolution/2), self.aperture_diameter/2, color='blue', fill=False, linestyle='solid', label='Aperture Circle')
        cbar = plt.colorbar(label='Log of (Pixel Intensity (Normalized to 1))')
        plt.gca().add_patch(aperture_circle)
        first_zero_circle = plt.Circle((self.resolution/2,self.resolution/2), self.first_zero_diameter/2, color='red', fill=False, linestyle='dashed', label='First Zero of Airy Disk')
        plt.gca().add_patch(first_zero_circle)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Cassegrain Telescope PSF")
        plt.legend()
        plt.show()
        
    def plot_residuals(self):
        plt.imshow(self.residuals)
        plt.title("Residuals of Point spread function calculated 2 ways")
        cbar = plt.colorbar(label='Pixel Intensity (Normalized to 1)')
        aperture_circle = plt.Circle((self.resolution/2, self.resolution/2), self.aperture_diameter/2, color='blue', fill=False, linestyle='solid', label='Aperture Circle')
        plt.gca().add_patch(aperture_circle)
        first_zero_circle = plt.Circle((self.resolution/2,self.resolution/2), self.first_zero_diameter/2, color='red', fill=False, linestyle='dashed', label='First Zero of Airy Disk')
        plt.gca().add_patch(first_zero_circle)
        plt.legend()
        plt.show()
    
    def plot_final(self):
        fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(131) 
        aperture_circle = plt.Circle((self.resolution/2, self.resolution/2), self.aperture_diameter/2, color='blue', fill=False, linestyle='solid', label='Aperture Circle')
        plt.gca().add_patch(aperture_circle)
        first_zero_circle = plt.Circle((self.resolution/2,self.resolution/2), self.first_zero_diameter/2, color='red', fill=False, linestyle='dashed', label='First Zero of Airy Disk')
        plt.gca().add_patch(first_zero_circle)
        plt.legend()
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        ax3.set_xlabel('x (m)')
        ax3.set_ylabel('y (m)')

        img1 = ax1.imshow(self.airy)
        img2 = ax2.imshow(self.normalized_fft)
        img3 = ax3.imshow(self.residuals)

        ax1.set_title("Airy Disk PSF")
        ax2.set_title("FFT PSF")
        ax3.set_title("Residuals")
        # Add color bar to ax3 with adjusted size and position
        
        cbar3 = fig.colorbar(img1, ax=ax1,label = "Pixel Intensity (Normalized to 1)", fraction=0.05, pad=0.05)
        cbar3.set_label('Pixel Intensity (Normalized to 1)', fontsize=12)  # Adjust the font size as needed
        
        cbar3 = fig.colorbar(img2, ax=ax2,label = "Pixel Intensity (Normalized to 1)", fraction=0.05, pad=0.05)
        cbar3.set_label('Pixel Intensity (Normalized to 1)', fontsize=12)  # Adjust the font size as needed
        
        
        cbar3 = fig.colorbar(img3, ax=ax3,label = "Pixel Intensity (Normalized to 1)", fraction=0.05, pad=0.05)
        cbar3.set_label('Pixel Intensity (Normalized to 1)', fontsize=12)  # Adjust the font size as needed

        fig.tight_layout()
        plt.show()
        
    def plot_aperture_size_comparison(self):
        fig = plt.figure(figsize=(8,4))
        fig.suptitle("Aperture Diameter = " + str(self.aperture_diameter))
        ax1 = fig.add_subplot(121) 
        aperture_circle = plt.Circle((self.resolution/2, self.resolution/2), self.aperture_diameter/2, color='blue', fill=False, linestyle='solid', label='Aperture Circle')
        plt.gca().add_patch(aperture_circle)
        first_zero_circle = plt.Circle((self.resolution/2,self.resolution/2), self.first_zero_diameter/2, color='red', fill=False, linestyle='dashed', label='First Zero of Airy Disk')
        plt.gca().add_patch(first_zero_circle)
        plt.legend()

        ax2 = fig.add_subplot(122)        
        aperture_circle_ax2 = plt.Circle((self.resolution/2, self.resolution/2), self.aperture_diameter/2, color='blue', fill=False, linestyle='solid', label='Aperture Circle')
        first_zero_circle_ax2 = plt.Circle((self.resolution/2, self.resolution/2), self.first_zero_diameter/2, color='red', fill=False, linestyle='dashed', label='First Zero of Airy Disk')
        ax2.add_patch(aperture_circle_ax2)
        ax2.add_patch(first_zero_circle_ax2)
        ax2.legend()
        
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        
        ax1.set_title("Airy Disk PSF")
        ax2.set_title("FFT PSF")
        
        img1 = ax1.imshow((self.airy))
        img2 = ax2.imshow((self.normalized_fft))
        
        cbar3 = fig.colorbar(img1, ax=ax1,label = "Pixel Intensity (Normalized to 1)", fraction=0.05, pad=0.05)
        cbar3.set_label('Pixel Intensity (Normalized to 1)', fontsize=12)  # Adjust the font size as needed
        
        cbar3 = fig.colorbar(img2, ax=ax2,label = "Pixel Intensity (Normalized to 1)", fraction=0.05, pad=0.05)
        cbar3.set_label('Pixel Intensity (Normalized to 1)', fontsize=12)  # Adjust the font size as needed
        
        fig.tight_layout()
        plt.show()
        
    def plot_residual_slice(self):
        plot_center_slice(self.residuals)
        
    
        
        
def fftIndgen(n):
    """Generate indices for FFT calculations.

    Args:
        n (int): Size of the grid.

    Returns:
        list: List of indices for FFT calculations.

    """
    # function just for indices in the gaussian random field function below
    # nice to have a function for this separate from the code
    a = range(0, int(n/2+1))
    a = [-i for i in a]
    b = reversed(range(1, int(n/2)))   
    b = [-i for i in b]
    return a + b


# Function to calculate integral
def simpson_int(n):
    """ Integration of Airy DIsk Using Simpson's method

    Args:
        number n

    Returns:
        Value of J1

    """
    
    i_numeric = 0.0

    for i in range(0,n):

        x = a +i*h

        if (i==0 or i==n):
            i_numeric = i_numeric + (1.0/3.0)*J1(x,rf(x))

        elif (np.remainder(i,2.)==0):
            i_numeric = i_numeric + (2.0/3.0)*J1(x,rf(x))

        else:
            i_numeric =i_numeric +(4.0/3.0)*J1(x,rf(x))

    J1_value = i_numeric*h

    return J1_value

def J1(theta, rf):
    """ Calculates J1
    Args:
        theta (): Used in calculation
        rf (): Rf in formula

    Returns: J1
    """
    return np.cos(theta - np.pi*rf*np.sin(theta))

def gaussian_random_field(Pk = lambda k : k**-3.0, size = 100):
    """Generate a 2D Gaussian random field.

    Args:
        Pk (function): Power spectrum function.
        size (int): Size of the grid.

    Returns:
        numpy.ndarray: 2D Gaussian random field.

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

def create_cassegrain_aperture(grid_dim,diam_ap,diam_ob,strut_width):
    """Create a Cassegrain telescope aperture mask.

    Args:
        grid_dim (tuple): Dimensions of the grid.
        diam_ap (float): Diameter of the larger aperture.
        diam_ob (float): Diameter of the central obstruction.
        strut_width (float): Width of the struts.

    Returns:
        numpy.ndarray: Cassegrain telescope aperture mask.

    """
    # not so cute but more direct way to make our cassegrain aperture
    center = int(grid_dim[0]/2)
    r_ap = int(diam_ap/2) #radius of larger aperture
    grid = create_circular_aperture(grid_dim,diam_ap)
    for i in range((center-r_ap),(center+r_ap)): # dont want to go over entire grid too slow
        for j in range((center-r_ap),(center+r_ap)):
            if (np.abs(i-j)<strut_width/2): #are i and j close to eachother so we get diagonal struts
                grid[i,j] = 0
                grid[-i,j] = 0 #symmetric for hubble but this doesnt have to be
            if (i-center)**2+(j-center)**2 < int(diam_ob/2)**2:
                grid[i,j] = 0 #center is blocked
    return grid


