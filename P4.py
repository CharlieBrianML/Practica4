import matplotlib.pyplot as plt
import numpy as np
# from os import listdir
# from os.path import isfile, join
# import cv2


def iter_folder():
    """
    Se controla la lectura de tantas imágenes como se desee
    """
    orgn_matrix = []

    for i in range(1223, 1223+20):
        try:
            path = 'rawdata/' + str(i)
            algo = read_file(path)
            if algo.shape != (262144,):
                orgn_matrix.append(algo)
        except:
            pass
    return orgn_matrix


def svd(data):
    U, Sigma, VT = np.linalg.svd(data, full_matrices=False)
    # Sanity check on dimensions
    print("data:", data.shape)
    print("U:", U.shape)
    print("Sigma:", Sigma.shape)
    print("V^T:", VT.shape)


def img_mean(data):
    """
    Se obtiene la media de las imágenes y se muestra la imagen resultante
    """
    data_mean = np.mean(data, axis=0)
    # data_reshaped = data_mean.reshape(128,128)
    # plt.imshow(data_reshaped)
    # plt.show()
    return data_mean


def mean_subtract(orgn_matrix, img_mean):
    """
    Se resta la media resultante de cada una de las imágenes contenidas en orgn_matrix
    """
    mean_matrix = []
    for i in range(len(orgn_matrix)):
        mean_matrix.append(orgn_matrix[i] - img_mean[i])
    # plt.imshow(result.reshape(128,128))
    # plt.show()
    return mean_matrix


def read_file(file_path):
    """
    Lee el archivo de imagen en formato binario
    """
    binaryFile = open(file_path, "rb")
    data = np.fromfile(binaryFile, dtype='uint8')

    binaryFile.close()
    # print(data.shape)
    return data

    # data_reshaped = data.reshape(128,128)
    # return data_reshaped.shape
    # plt.imshow(data_reshaped)
    # plt.show()


if __name__ == "__main__":
    orgn_matrix = iter_folder()
    img_mean_v = img_mean(orgn_matrix)  # img_mean_v.shape = (16384,)
    mean_matrix = mean_subtract(orgn_matrix, img_mean_v)
    svd(np.matrix(mean_matrix))  # np.matrix(mean_matrix).shape = (18, 16384)

    # forma de Charlie
    # save_imges(ls('rawdata'))
    # images = vector_img(ls('data'))
    # img_mean = mean(images)
    # print(img_mean.shape)
    # cv2.imshow('img_mean',img_mean)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# def read_img(fileName):
# 	binaryFile = open('rawdata/'+fileName) # Lecura del archivo
# 	data = np.fromfile(binaryFile, dtype='uint8') # Conversion a enteros de 8 bits sin signo
# 	binaryFile.close()
# 	tam = int(np.sqrt(len(data))) # Se optiene el tamaño del la fila y columna
# 	data_reshaped = data.reshape(tam,tam) # Se cambia la dimension del vector
# 	if(tam>128):
# 		data_reshaped = cv2.resize(data_reshaped, (128,128), fx=0.75, fy=0.75) # Reescala la imagen a 128x128
# 	return data_reshaped

# def ls(ruta):
#     return [arch for arch in listdir(ruta) if isfile(join(ruta, arch))] #Lista los archivos del directorio

# def vector_img(listFiles):
# 	images = np.zeros((len(listFiles),128,128))
# 	for i in range(len(listFiles)):
# 		images[i,:,:] = cv2.imread('data/'+listFiles[i],0)
# 	return images

# def save_imges(listFiles):
# 	for i in range(len(listFiles)):
# 		img = read_img(listFiles[i])
# 		plt.imsave('data/'+listFiles[i]+'.png',img, cmap = 'gray') # Guarda como imagen .png

# def mean(matrix):
# 	matrix_mean = np.zeros((128, 128))
# 	for i in range(3993):
# 		matrix_mean = matrix_mean + matrix[i,:,:]
# 	print(np.max(matrix_mean))
# 	for j in range(128):
# 		for k in range(128):
# 			matrix_mean[j,k] = int(matrix_mean[j,k]/3993)
# 	#matrix_mean = matrix_mean/3993
# 	print(np.max(matrix_mean))
# 	return matrix_mean
