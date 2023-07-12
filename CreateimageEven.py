# import sys
# from PIL import Image

# image_fullpath=sys.argv[1]
# image_name=sys.argv[2]

# img= Image.open(str(image_fullpath))

# image_save_path=image_fullpath.replace(image_name,"temp.png")
# img.rotate(90).convert("LA").save(image_save_path)

# print("/media/temp.png")

import sys
from PIL import Image
import pants
import numpy as np
import cv2
import os
import uuid
from itertools import permutations
from Function import TabuSearch
from Function import RandomSearch
from Function import GreedySearch
from Function import ACO
from Function import SkorPertama
from ProcessImage import ScaleImage
from ProcessImage import ProcesImage
from ProcessImage import SeparateImage
from ProcessImage import ConvertRGB
from ProcessImage import ConvertArrayImage
from ProcessImage import ConvertLiditoArray
from ProcessImage import CreateImage

def SkorACO(a, b):
    temp1 = Array_data[a[0]]
    temp2 = Array_data[a[1]]
    temp3 = Array_data[b[0]]
    temp4 = Array_data[b[1]]

    SkorArray1 = SkorPertama(temp1, temp2)
    SkorArray2 = SkorPertama(temp3, temp4)
    SkorArray3 = SkorPertama(temp2, temp3)

    SkorArray = 1/(1 + SkorArray1 + SkorArray2 + SkorArray3)
    return SkorArray

image_fullpath=sys.argv[1]
image_name=sys.argv[2]

jmlBaris = sys.argv[3]
jmlBaris = int(jmlBaris)
jmlBaris = int(jmlBaris/2)

Baris = sys.argv[4]
Baris = int(Baris)

ModeGenerate = sys.argv[5]

folderUser = sys.argv[6]
makeFolder = f"media/{folderUser}"
# Make folder User
if(not os.path.exists(makeFolder)):
    os.mkdir(makeFolder)

ModeGenerate = int(ModeGenerate)

unique_file_name = uuid.uuid4().hex
unique = f"{folderUser}/{unique_file_name}.png"
image_save_path=image_fullpath.replace(image_name, unique)

#inisialisasi nama direktori
namaDirektori = f"Image/{folderUser}"
Direktori = str(namaDirektori)

if(not os.path.exists(Direktori)):
    os.mkdir(Direktori)

#Pemisahan Baris
img = cv2.imread(str(image_fullpath), 1)
SeparateImage(img, Direktori)

height, width, channels = img.shape
img = []
Lidi = []

# image to RGBA
img, Lidi = ConvertRGB(img, Lidi, height, Direktori)

# convert binary
Array_data = []
Tabu_List = []

Array_data = ConvertArrayImage(img, Array_data)

comb = list(permutations(Lidi, 2))

SkorArray = []
for i in range(0, int(len(Lidi))):
    for j in range(0, int(len(Lidi)-1)):
        temp1 = Array_data[i].copy()
        temp2 = Array_data[j].copy()

        SkorArray.append(SkorPertama(temp1, temp2))

#World Ant
world = pants.World(comb, SkorACO)
solver = pants.Solver()

# Bagaimana menyimpan perbandingan setiap baris?
comb = np.array_split(comb, height)
PanjangLidi = int(len(Lidi))-1


if(ModeGenerate == 1):
    # print("Tabu Serach Result \n")
    Tabu_List,Best_Solution = TabuSearch(PanjangLidi, Array_data, Baris, jmlBaris, Tabu_List)
    a = Best_Solution[0]
    # print(a)
elif(ModeGenerate == 2):
    # print("Greedy Serach Result \n")
    a = GreedySearch(PanjangLidi, comb,Baris, jmlBaris)
    # print(a)
elif(ModeGenerate == 3):
    # print("Random Search Result \n")
    a = RandomSearch(PanjangLidi, jmlBaris)
    # print(a)
elif(ModeGenerate == 4):
    # print("ACO Result \n")
    a = ACO(solver, world, jmlBaris)
    # print(a)
# else:
    # print("Format yang anda masukkan salah ex : 1")


#Inisialisasi Lidi ke dalam Array

img = cv2.imread(str(image_fullpath), 1)

height, width, channels = img.shape
img = []

img = ConvertLiditoArray(img, height, Direktori)

#Preparation Lidi

c = a.copy()
c = c[::-1]
a.extend(c)
b = a.copy()
b = [x+1 for x in b]
#Create Image 

img = CreateImage(a, img)
Image.fromarray(img).save(f"{namaDirektori}/Hasil1.jpg")

img= Image.open(f"{namaDirektori}/Hasil1.jpg")
img = ProcesImage(img)
img = ScaleImage(img)
img.save(image_save_path)


print(f"/media/{folderUser}/{unique_file_name}.png{b}")
