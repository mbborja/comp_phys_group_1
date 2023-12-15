import unittest
import numpy as np

def create_circular_aperture(grid_dim, diameter):
    aperture = np.zeros(grid_dim)
    center_x, center_y = grid_dim[0] // 2, grid_dim[1] // 2
    y, x = np.ogrid[:grid_dim[0], :grid_dim[1]]
    mask = (x - center_x)**2 + (y - center_y)**2 <= (diameter / 2)**2
    aperture[mask] = 1
    return aperture

class TestCreateCircularAperture(unittest.TestCase):
    def test_aperture_size(self):
        # Test if the aperture has the correct dimensions
        grid_dim = (100, 100)
        diameter = 30
        aperture = create_circular_aperture(grid_dim, diameter)
        self.assertEqual(aperture.shape, grid_dim)

    def test_center_value(self):
        # Test if the center of the aperture is set to 1
        grid_dim = (100, 100)
        diameter = 30
        aperture = create_circular_aperture(grid_dim, diameter)
        center_x, center_y = grid_dim[0] // 2, grid_dim[1] // 2
        self.assertEqual(aperture[center_y, center_x], 1)

    def test_outer_value(self):
        # Test if values outside the circular aperture are set to 0
        grid_dim = (100, 100)
        diameter = 30
        aperture = create_circular_aperture(grid_dim, diameter)
        center_x, center_y = grid_dim[0] // 2, grid_dim[1] // 2
        distance = int(diameter / 2) + 1  # distance from the center
        outside_value = aperture[center_y + distance, center_x]
        self.assertEqual(outside_value, 0)

if __name__ == '__main__':
    unittest.main()
