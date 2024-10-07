import numpy as np

# Function to convert heading angles to slope
def heading_to_slope(heading_angle):
    return np.tan(np.radians(heading_angle))

# Function to compute the intersection point
def find_intersection(x_h, y_h, heading_h, x_t, y_t, heading_t):
    # Convert heading angles to slopes
    m_h = heading_to_slope(heading_h)
    m_t = heading_to_slope(heading_t)

    # Line equations in slope-intercept form (y = mx + b)
    # For vehicle H: y_h = m_h * x_h + b_h => b_h = y_h - m_h * x_h
    b_h = y_h - m_h * x_h

    # For vehicle T: y_t = m_t * x_t + b_t => b_t = y_t - m_t * x_t
    b_t = y_t - m_t * x_t

    # Solving for intersection point C(x, y):
    # m_h * x + b_h = m_t * x + b_t
    # Rearranging to find x:
    x_c = (b_t - b_h) / (m_h - m_t)

    # Substitute x_c into one of the line equations to get y_c
    y_c = m_h * x_c + b_h

    return x_c, y_c

# Example usage:
# Coordinates and headings for vehicles H and T
x_h, y_h = 0, 0      # Starting point of vehicle H
x_t, y_t = 10, 10    # Starting point of vehicle T
heading_h = 45       # Heading angle of vehicle H in degrees
heading_t = 135      # Heading angle of vehicle T in degrees

# Find the intersection point
intersection_point = find_intersection(x_h, y_h, heading_h, x_t, y_t, heading_t)
print(f"Intersection Point C: {intersection_point}")
