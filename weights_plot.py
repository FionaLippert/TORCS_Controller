import numpy as np
import matplotlib.pyplot as plt

weights = np.load('./trained_nn/pca2.npy')

plt.imshow(weights)
plt.title('Weights of the PCA', size=18)
plt.xlabel('Output Neuron', size=18)
plt.ylabel('Input Sensor', size=18)
plt.yticks(range(weights.shape[0]))
plt.xticks(range(weights.shape[1]))
plt.colorbar()
# plt.savefig('pca_weights.png')
plt.show()
