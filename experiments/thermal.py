# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:29:34 2023

@author: hao9
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the size of the thermal map (number of rows and columns)
num_rows = 100
num_cols = 100

# Define the temperature values for each zone center point and wall center point
zone_temperatures = {
    (25, 25): 20.0,
    (50, 50): 25.0,
    (75, 75): 22.0
}

wall_temperatures = {
    ((25, 25), (50, 50)): 22.0,
    ((50, 50), (75, 75)): 24.0,
    ((75, 75), (25, 25)): 21.0
}

# Define the areas for each zone
zone_areas = {
    1: (0, 50, 0, 50),   # Zone 1 area (top-left quadrant)
    2: (0, 50, 50, 100),  # Zone 2 area (bottom-left quadrant)
    3: (50, 100, 50, 100) # Zone 3 area (bottom-right quadrant)
}

# Create the thermal map matrix
thermal_map = np.zeros((num_rows, num_cols))

# Iterate over each pixel in the thermal map
for i in range(num_rows):
    for j in range(num_cols):
        x = i + 0.5  # X-coordinate of the pixel center
        y = j + 0.5  # Y-coordinate of the pixel center
        
        # Determine which zone the pixel belongs to based on its coordinates
        zone = None
        for zone_id, (x1, x2, y1, y2) in zone_areas.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                zone = zone_id
                break
        
        # Check if the pixel belongs to a zone
        if zone is not None:
            # Get the temperature at the zone center point
            zone_temperature = zone_temperatures.get((x1 + x2) // 2, 0.0)
            
            # Assign the temperature to the pixel
            thermal_map[i, j] = zone_temperature

# Create the thermal map plot
plt.imshow(thermal_map, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Temperature (°C)')
plt.xlabel('Column')
plt.ylabel('Row')
plt.title('Thermal Map')
plt.show()
















import numpy as np
import matplotlib.pyplot as plt

# Define the size of the image (number of rows and columns)
num_rows = 100
num_cols = 100

# Define the colors and temperatures for each zone
zone_temperatures = {
    1: 0.0,
    2: 10.5,
    3: 25.0,
    4: 27.5,
    5: 30.0,
    6: 40.5
}

# Create an empty image with the specified dimensions
image = np.zeros((num_rows, num_cols), dtype=np.uint8)

# Assign colors and temperatures to each pixel based on the zone
for i in range(num_rows):
    for j in range(num_cols):
        x = i + 0.5  # X-coordinate of the pixel center
        y = j + 0.5  # Y-coordinate of the pixel center
        
        # Determine which zone the pixel belongs to based on its coordinates
        if x < 33.33 and y < 50:
            zone = 1
        elif x < 66.67 and y < 50:
            zone = 2
        elif y < 50:
            zone = 3
        elif x < 33.33:
            zone = 4
        elif x < 66.67:
            zone = 5
        else:
            zone = 6
        
        # Assign the color and temperature to the pixel
        image[i, j] = zone_temperatures[zone]
        temperature = zone_temperatures[zone]
        # image[i, j, 0] = int(temperature * 255 / 32.5)  # Red channel represents temperature

# Display the image
plt.imshow(image, cmap='hot', vmin=0, vmax=40.5)
plt.colorbar(label='Temperature')
plt.axis('off')
plt.title('Floor Plan with Temperature Zones')
plt.show()











import numpy as np
import matplotlib.pyplot as plt

# Define the number of rows and columns for the zone matrix
num_rows = 100
num_cols = 100

# Create a meshgrid
x, y = np.meshgrid(np.arange(num_cols), np.arange(num_rows))

# Define the boundaries of each zone
zone_boundaries = [0, 33, 50, 67, 100]

# Create an empty matrix to store the zone values
zone_matrix = np.zeros((num_rows, num_cols))

# Assign values to the center of each zone
for zone in range(1, 5):
    xmin, xmax = zone_boundaries[zone - 1], zone_boundaries[zone]
    ymin, ymax = 0, num_rows if zone <= 3 else 50
    
    zone_value = zone * 10  # Example value for the zone
    
    # Find the indices of the center of the zone
    center_x = int((xmin + xmax) / 2)
    center_y = int((ymin + ymax) / 2)
    
    # Assign the value to the center of the zone
    zone_matrix[center_y, center_x] = zone_value

# Plot the zone matrix with a colorbar
plt.imshow(zone_matrix, cmap='rainbow')
plt.colorbar(label='Zone Value')
plt.title('Zone Matrix')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()


















import numpy as np
import matplotlib.pyplot as plt

# Define the size of the image (number of rows and columns)
num_rows = 100
num_cols = 100

# Define the thermal properties
thermal_conductivity = 0.5  # Thermal conductivity coefficient
heat_generation = 1.0  # Heat generation rate

# Define the boundary conditions
outside_temperature = 20.0  # Temperature of the outside environment
room_temperatures = {
    1: 25.0,
    2: 22.5,
    3: 27.5,
    4: 20.0,
    5: 30.0,
    6: 32.5
}

# Create an empty image with the specified dimensions
image = np.zeros((num_rows, num_cols))

# Define the grid points
x = np.linspace(0, num_cols - 1, num_cols)
y = np.linspace(0, num_rows - 1, num_rows)
X, Y = np.meshgrid(x, y)

# Calculate the thermal map using interpolation and thermal equations
for i in range(num_rows):
    for j in range(num_cols):
        x = j
        y = i
        
        # Determine the zone based on the location of the pixel
        if x < 33.33 and y < 50:
            zone = 1
        elif x < 66.67 and y < 50:
            zone = 2
        elif y < 50:
            zone = 3
        elif x < 33.33:
            zone = 4
        elif x < 66.67:
            zone = 5
        else:
            zone = 6
        
        # Get the temperature of the current zone
        zone_temperature = room_temperatures[zone]
        
        # Calculate the thermal map using the thermal equation
        image[i, j] = zone_temperature + (outside_temperature - zone_temperature) * np.exp(-thermal_conductivity * Y[i, j])

# Display the thermal map
plt.imshow(image, cmap='hot', vmin=outside_temperature, vmax=max(room_temperatures.values()))
plt.colorbar(label='Temperature (°C)')
plt.title('Thermal Map')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()















import numpy as np
import matplotlib.pyplot as plt

# Define the thermal properties and constants
thermal_conductivity = 1.0  # Thermal conductivity coefficient
num_iterations = 100  # Number of iterations

# Define the dimensions of the image
num_rows = 100
num_cols = 100

# Create an empty image array
image = np.zeros((num_rows, num_cols))

# Assign the initial temperature values based on known temperatures or boundary conditions
for i in range(num_rows):
    for j in range(num_cols):
        x = i + 0.5  # X-coordinate of the pixel center
        y = j + 0.5  # Y-coordinate of the pixel center
        
        # Determine which zone the pixel belongs to based on its coordinates
        if x < 33.33 and y < 50:
            zone = 1
        elif x < 66.67 and y < 50:
            zone = 2
        elif y < 50:
            zone = 3
        elif x < 33.33:
            zone = 4
        elif x < 66.67:
            zone = 5
        else:
            zone = 6
        
        # Assign the color and temperature to the pixel
        image[i, j] = zone_temperatures[zone]

# Iterate over each pixel in the image
for iteration in range(num_iterations):
    # Create a copy of the image to store the updated temperatures for the current iteration
    updated_image = np.copy(image)
    
    # Iterate over each pixel in the image
    for i in range(num_rows):
        for j in range(num_cols):
            # Determine the neighboring pixels of the current pixel
            neighbors = []
            if i > 0:
                neighbors.append(image[i - 1, j])  # Top neighbor
            if i < num_rows - 1:
                neighbors.append(image[i + 1, j])  # Bottom neighbor
            if j > 0:
                neighbors.append(image[i, j - 1])  # Left neighbor
            if j < num_cols - 1:
                neighbors.append(image[i, j + 1])  # Right neighbor
            
            # Calculate the average temperature of the neighboring pixels
            average_temperature = np.mean(neighbors)
            
            # Assign the calculated average temperature to the current pixel in the updated image
            updated_image[i, j] = average_temperature
    
    # Update the image with the temperatures from the current iteration
    image = updated_image

# Display the resulting thermal map
plt.imshow(image, cmap='hot', vmin=np.min(image), vmax=np.max(image))
plt.colorbar(label='Temperature')
plt.title('Thermal Map')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()





