import numpy as np
import matplotlib.pyplot as plt
kps = np.load(r"C:\Users\Neurotech\Downloads\world_points.npy", allow_pickle=True)
print(kps.shape)
# print(kps)

for i in range(kps.shape[2]):

    print(kps[:,:,i].shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(kps[:,0 , i], kps[:, 1, i], kps[:, 2, i])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
