import numpy as np
import matplotlib.pyplot as plt
kps = np.load(r"C:\Users\Neurotech\Downloads\world_points.npy", allow_pickle=True)
print(kps.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kps[:, 0], kps[:, 1], kps[:, 2])

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()