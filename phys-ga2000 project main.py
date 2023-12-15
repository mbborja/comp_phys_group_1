from telescope_methods import telescope

# Declare all variables
resolution = 2001
wavelength = 150
focal_length = 100
aperture_size = resolution/20
grid_dim = (resolution,resolution)

# Create test class (Feel free to change the variables to change the telescope)
telescope_test = telescope(resolution, wavelength,focal_length,aperture_size)

print(telescope_test.first_zero_diameter)

telescope_test.plot_airy_disk()

telescope_test.plot_casse_disk()

telescope_test.plot_normalized_fft()

telescope_test.plot_residuals()

telescope_test.plot_residual_slice()

telescope_test.plot_final()

telescope_test.plot_aperture_size_comparison()