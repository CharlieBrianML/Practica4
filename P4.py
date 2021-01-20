import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join

def read_img(fileName):
	binaryFile = open('rawdata/'+fileName) # Lecura del archivo
	data = np.fromfile(binaryFile, dtype='uint8') # Conversion a enteros de 8 bits sin signo
	binaryFile.close()
	tam = int(np.sqrt(len(data))) # Se optiene el tamaño del la fila y columna
	data_reshaped = data.reshape(tam,tam) # Se cambia la dimension del vector
	plt.imsave('data/'+fileName+'.png',data_reshaped, cmap = 'gray') #
	
def ls(ruta = 'rawdata'):
    return [arch for arch in listdir(ruta) if isfile(join(ruta, arch))] #Lista los archivos del directorio	

if __name__ == "__main__":
	listFiles = ls()
	for i in range(len(listFiles)):
		read_img(listFiles[i])