import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

# Length of the bicycle
L = 1.0  # Length of the vehicle (bicycle)

class EKF:
    def __init__(self, dt, state_dim, measurement_dim):
        self.dt = dt
        self.x = np.zeros((state_dim, 1))  # State: [x, y, v, theta]
        self.P = np.eye(state_dim) * 0.1  # Covariance
        self.Q = np.diag([0.1, 0.1, 0.1, 0.1])  # Process noise
        self.R = np.diag([0.5, 0.5, 0.1])  # Measurement noise (GPS)
        self.I = np.eye(state_dim)
        self.H = np.zeros((measurement_dim, state_dim))  # Measurement matrix
        self.H[0, 0] = 1  # GPS x
        self.H[1, 1] = 1  # GPS y
        self.H[2, 3] = 1  # GPS heading

    def predict(self, u):
        x, y, v, theta = self.x.flatten()

        acc = u[0]  # Linear acceleration
        delta = u[1]  # Steering angle

        # Update state using bicycle model
        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        v_new = v + acc * self.dt
        theta_new = theta + (v / L) * np.tan(delta) * self.dt

        self.x = np.array([x_new, y_new, v_new, theta_new]).reshape(-1, 1)

        # State transition matrix
        F = np.eye(4)
        F[0, 2] = np.cos(theta) * self.dt
        F[1, 2] = np.sin(theta) * self.dt
        F[3, 2] = (np.tan(delta) * self.dt) / L

        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        y = z.reshape(-1, 1) - (self.H @ self.x)  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

def generate_simulation_data(time_steps, dt):
    X_real = []
    Y_real = []
    V_real = []
    Heading_real = []

    x = 0.0
    y = 0.0
    v = 5.0  # Start with a higher constant speed
    theta = 0.0

    for t in range(time_steps):
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        Heading_real.append(theta)

        X_real.append(x)
        Y_real.append(y)
        V_real.append(v)

        if 50 < t < 100:
            delta = np.pi / 6  # Sharp right turn
            v = 5.0  # Maintain speed
        elif 100 < t < 150:
            delta = -np.pi / 6  # Sharp left turn
            v = 5.0  # Maintain speed
        elif 200 < t < 300:
            delta = np.pi / 4  # Left turn at intersection
            v = 3.0  # Slow down for the turn
        elif 400 < t < 600:
            delta = np.pi / 3  # Sharp right turn
            v = 5.0  # Speed up during the turn
        elif 601 < t < 700:
            delta = np.pi / -30  # Sharp right turn
            v = 1.0  # Speed up during the turn
        elif 701 < t < 900:
            delta = np.pi / 4  # Sharp right turn
            v = 1.0  # Speed up during the turn
        else:
            delta = 0.0  # Straight

        theta += (v / L) * np.tan(delta) * dt

    return np.array(X_real), np.array(Y_real), np.array(V_real), np.array(Heading_real)


def plot_comparison(X_real, Y_real, X_estimated, Y_estimated, Heading_real, Heading_estimated):
    plt.figure(figsize=(10, 8))

    # Calculate deviations
    deviations = np.sqrt((X_real - X_estimated) ** 2 + (Y_real - Y_estimated) ** 2)

    # Calculate real and estimated velocities
    V_real = np.sqrt(np.diff(X_real) ** 2 + np.diff(Y_real) ** 2)  # Real velocity
    V_estimated = np.sqrt(np.diff(X_estimated) ** 2 + np.diff(Y_estimated) ** 2)  # Estimated velocity

    # Add an extra element at the beginning for plotting purposes
    V_real = np.insert(V_real, 0, 0)
    V_estimated = np.insert(V_estimated, 0, 0)

    # Plot real and estimated positions
    plt.plot(X_real, Y_real, 'b-', label='Real Position', markersize=6)
    plt.plot(X_estimated, Y_estimated, 'r--', label='Estimated Position', markersize=6)

    # Plot within 0.5 meter deviation points in green
    below_half_meter = deviations < 0.5
    plt.plot(X_estimated[below_half_meter], Y_estimated[below_half_meter], 'go', label='Within 0.5m Deviation',
             markersize=4)

    # Plot velocities as arrows
    for i in range(0, len(X_real) - 1, 5):  # Use len(X_real)-1 to prevent indexing issues
        plt.arrow(X_real[i], Y_real[i], np.cos(Heading_real[i]) * 0.2, np.sin(Heading_real[i]) * 0.2,
                  head_width=0.05, color='blue', alpha=0.5)
        plt.arrow(X_estimated[i], Y_estimated[i], np.cos(Heading_estimated[i]) * 0.2,
                  np.sin(Heading_estimated[i]) * 0.2,
                  head_width=0.05, color='red', alpha=0.5)

    # Plotting the velocity arrows
    for i in range(len(X_real) - 1):
        # Calculate velocity direction
        dir_real = np.array([X_real[i + 1] - X_real[i], Y_real[i + 1] - Y_real[i]])
        dir_estimated = np.array([X_estimated[i + 1] - X_estimated[i], Y_estimated[i + 1] - Y_estimated[i]])

        # Normalize the direction vectors
        if np.linalg.norm(dir_real) > 0:
            dir_real /= np.linalg.norm(dir_real)
        if np.linalg.norm(dir_estimated) > 0:
            dir_estimated /= np.linalg.norm(dir_estimated)

        # Scale by velocity for visualization
        plt.arrow(X_real[i], Y_real[i], dir_real[0] * V_real[i], dir_real[1] * V_real[i],
                  head_width=0.1, color='blue', alpha=0.5, length_includes_head=True)
        plt.arrow(X_estimated[i], Y_estimated[i], dir_estimated[0] * V_estimated[i], dir_estimated[1] * V_estimated[i],
                  head_width=0.1, color='red', alpha=0.5, length_includes_head=True)

    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Real vs Estimated Trajectory with Deviation and Velocity Arrows')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


# Main execution
if __name__ == "__main__":
    dt = 0.1  # Time step (100 ms)
    time_steps = 1000  # Simulate for 100 seconds
    state_dim = 4  # State dimensions: [x, y, v, theta]
    measurement_dim = 3  # GPS measurements

    X_real, Y_real, V_real, Heading_real = generate_simulation_data(time_steps, dt)

    ekf = EKF(dt, state_dim, measurement_dim)

    X_estimated = []
    Y_estimated = []
    Heading_estimated = []

    for t in range(time_steps):
        acc = 0.0  # Constant acceleration
        steering_angle = 0.0  # No steering input
        if 50 < t < 100:
            steering_angle = np.pi / 6  # Sharp right turn
        elif 100 < t < 150:
            steering_angle = -np.pi / 6  # Sharp left turn
        elif 200 < t < 300:
            steering_angle = np.pi / 4  # Left turn at intersection
        elif 400 < t < 600:
            steering_angle = np.pi / 3  # Sharp right turn

        imu_data = np.array([acc, steering_angle])

        ekf.predict(imu_data)

        X_estimated.append(ekf.x[0, 0])
        Y_estimated.append(ekf.x[1, 0])
        Heading_estimated.append(ekf.x[3, 0])

        if t % 10 == 5:
            gps_measurement = np.array([X_real[t], Y_real[t], Heading_real[t]])
            ekf.update(gps_measurement)

    X_estimated = np.array(X_estimated)
    Y_estimated = np.array(Y_estimated)
    Heading_estimated = np.array(Heading_estimated)

    plot_comparison(X_real, Y_real, X_estimated, Y_estimated, Heading_real, Heading_estimated)
