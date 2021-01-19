import matplotlib.pyplot as plt
import numpy as np

def read_img():
    binaryFile = open("rawdata/rawdata/1223","rb")
    data = np.fromfile(binaryFile, dtype='uint8') # reads the whole file
    binaryFile.close()

    print(data.shape)
    data_reshaped = data.reshape(128,128)
    plt.imshow(data_reshaped)
    plt.show()


if __name__ == "__main__":
    read_img()