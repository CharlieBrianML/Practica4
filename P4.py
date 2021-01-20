import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import cv2

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
    img_mean(la_lista)


def img_mean(data):
    """
    Se obtiene la media de las imágenes y se muestra la imagen resultante
    """
    data_mean = np.mean(data, axis=0)
    data_reshaped = data_mean.reshape(128,128)
    plt.imshow(data_reshaped)
    plt.show()


def read_img(fileName):
	binaryFile = open('rawdata/'+fileName) # Lecura del archivo
	data = np.fromfile(binaryFile, dtype='uint8') # Conversion a enteros de 8 bits sin signo
	binaryFile.close()
	tam = int(np.sqrt(len(data))) # Se optiene el tamaño del la fila y columna
	data_reshaped = data.reshape(tam,tam) # Se cambia la dimension del vector
	if(tam>128):
		data_reshaped = cv2.resize(data_reshaped, (128,128), fx=0.75, fy=0.75) # Reescala la imagen a 128x128
	return data_reshaped
	
def ls(ruta):
    return [arch for arch in listdir(ruta) if isfile(join(ruta, arch))] #Lista los archivos del directorio	
	
def vector_img(listFiles):
	images = np.zeros((len(listFiles),128,128))
	for i in range(len(listFiles)):
		images[i,:,:] = cv2.imread('data/'+listFiles[i],0)
	return images
	
def save_imges(listFiles):
	for i in range(len(listFiles)):
		img = read_img(listFiles[i])
		plt.imsave('data/'+listFiles[i]+'.png',img, cmap = 'gray') # Guarda como imagen .png
		
def mean(matrix):
	matrix_mean = np.zeros((128, 128))
	for i in range(3993):
		matrix_mean = matrix_mean + matrix[i,:,:]
	print(np.max(matrix_mean))
	for j in range(128):
		for k in range(128):
			matrix_mean[j,k] = int(matrix_mean[j,k]/3993)
	#matrix_mean = matrix_mean/3993
	print(np.max(matrix_mean))
	return matrix_mean

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
    # iter_folder() # para hacerlo como Enrique

    # forma de Charlie
	#save_imges(ls('rawdata'))
	images = vector_img(ls('data'))
	img_mean = mean(images)
	print(img_mean.shape)
	cv2.imshow('img_mean',img_mean)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
