import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
# from os import listdir
# from os.path import isfile, join
import cv2


def iter_folder():
    """
    Se controla la lectura de tantas imágenes como se desee
    """
    orgn_matrix = []

    for i in range(1223, 1223+500):
        try:
            path = 'rawdata/' + str(i)
            algo = read_file(path)
            if algo.shape != (262144,):
                orgn_matrix.append(algo)
        except:
            pass
    dim = read_img('gato.jpg').reshape(16384,)
    orgn_matrix.append(dim)

    # plt.imshow(orgn_matrix[1].reshape(128, 128), cmap='gray')
    # plt.show()
    return orgn_matrix


def svd(data):
    # print("data:", data.shape)
    # print("U:", U.shape)
    # print("Sigma:", Sigma.shape)
    # print("V^T:", VT.shape)
    return np.linalg.svd(data, full_matrices=False)


def img_mean(data):
    """
    Se obtiene la media de las imágenes y se muestra la imagen resultante
    """
    data_mean = np.mean(data, axis=0)
    # data_reshaped = data_mean.reshape(128, 128)
    # plt.imshow(data_reshaped, cmap='gray')
    # plt.show()
    return data_mean


def mean_subtract(orgn_matrix, img_mean):
    """
    Se resta la media resultante de cada una de las imágenes contenidas en orgn_matrix
    """
    sub_img_mn = []
    for i in range(len(orgn_matrix)):
        sub_img_mn.append(subtract_ind(orgn_matrix[i], img_mean))
        # plt.imshow(sub_img_mn[i].reshape(128,128),cmap='gray')
        # plt.show()
    # plt.imshow(sub_img_mn[0].reshape(128, 128), cmap='gray')
    # plt.show()
    return np.matrix(sub_img_mn)


def subtract_ind(orgn_data, mean):
    return orgn_data - mean


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


def generic_name(data, VT, num_components):
    return np.matmul(data, VT[:num_components, :].T)


def reconstruction(Y, C, M, h, w, image_index):
    # n_samples, n_features = Y.shape
    weights = np.dot(Y, C.transpose())
    centered_vector = np.dot(weights[image_index, :], C)
    recovered_image = (M+centered_vector).reshape(h, w)
    return recovered_image


def save_imges(listFiles):
    for i in range(len(listFiles)):
        # Guarda como imagen .png
        plt.imsave('data_gato_10/'+str(i)+'.png', listFiles[i], cmap='gray')


def read_img(path):
    return cv2.imread(path, 0)


if __name__ == "__main__":
    orgn_matrix = iter_folder()  # 498
    img_mean_v = img_mean(orgn_matrix)  # img_mean_v.shape = (16384,)
    sub_img_mn = mean_subtract(orgn_matrix, img_mean_v)
    U, Sigma, VT = svd(sub_img_mn)  # np.matrix(sub_img_mn).shape = (18, 16384)
    num_components = 100
    components = VT[:num_components]
    Y = generic_name(orgn_matrix, VT, num_components)

    recovered_images = [reconstruction(
        sub_img_mn, components, img_mean_v, 128, 128, i) for i in range(len(orgn_matrix))]

    plt.imshow(recovered_images[1],cmap='gray')
    plt.show()
    # recov_img = reconstruction(sub_img_mn, components, img_mean_v, 128, 128, 1)
    # plt.imshow(components[1].reshape(128,128),cmap='gray')
    # plt.show()
    # save_imges(recovered_images)
    plt.scatter(Y[:, 0].tolist(), Y[:, 1].tolist())
    plt.show()

    kmed = KMeans(3)
    kmed.fit(Y)
    labels = kmed.predict(Y)

    arr = []
    for i in range(499):
        if labels[i] == 0:
            arr.append('b')
        elif labels[i] == 1:
            arr.append('g')
        elif labels[i] == 2:
            arr.append('r')
    plt.scatter(Y[:, 0].tolist(), Y[:, 1].tolist(), s=30, c=arr)
    plt.show()

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
