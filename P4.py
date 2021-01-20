import matplotlib.pyplot as plt
import numpy as np

def iter_folder():
    """
    Se controla la lectura de tantas imágenes como se desee
    """
    la_lista = []
    for i in range(1223,1223+4000):
        try:
            path = 'rawdata/'+ str(i)
            algo = read_img(path)
            # print(algo.shape)
            if algo.shape != (262144,):
                la_lista.append(algo)
        except:
            pass
    # print(la_lista)
    # print(len(la_lista))
    mean(la_lista)


def mean(data):
    """
    Se obtiene la media de las imágenes y se muestra la imagen resultante
    """
    data_mean = np.mean(data, axis=0)
    data_reshaped = data_mean.reshape(128,128)
    plt.imshow(data_reshaped)
    plt.show()


def read_img(file_path):
    """
    Lee el archivo de imagen en formato binario
    """
    binaryFile = open(file_path,"rb")
    data = np.fromfile(binaryFile, dtype='uint8')
    
    binaryFile.close()
    # print(data.shape)
    return data

    # data_reshaped = data.reshape(128,128)
    # return data_reshaped.shape
    # plt.imshow(data_reshaped)
    # plt.show()


if __name__ == "__main__":
    iter_folder()