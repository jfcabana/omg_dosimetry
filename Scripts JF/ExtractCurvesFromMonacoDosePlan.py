#-------------------------------------------------------------------------------
# Name:        Lecture DosePlan de Monaco
# Purpose:
#
# Author:      Jonathan Morin
#
# Created:     22-06-2018
# Copyright:   (c) mojo1259 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import string
root = Tk()

def main():
    name = askopenfilename(initialdir="P:\\Projets\\CRIC\\Physique_Medicale\\VersaHD\\Modelisation Monaco\\Modèle CRIC\\Électrons", filetypes = (("ALL Files","*.*"),), title = "Choisissez le fichier à analyser") #Fonction qui ouvre l'explorateur windows pour choisir un fichier à analyser.
    with open(name, "r") as MONACO:
        for i in range(0,28,1):        #On boucle sur 28 items, car il semblerait que le saut de ligne soit lu comme une ligne à part entière par la fonction readline. Ainsi, chaque ligne est en réalité 2 lignes selon cette fonction.
            buffer = MONACO.readline().replace("\n", "")
        print(buffer)
        buffer = MONACO.readline()
        NbPtsX = int(buffer.split(",")[1])   #Les informations sont séparées par une virgule, donc sur cette ligne, on en veut pas le début qui est "DosePtsxy". Les informations qui suivent sont le nombre de points en x et en y dans le fichier.
        NbPtsY = int(buffer.split(",")[2])
        for i in range(0,3,1):
            buffer = MONACO.readline()
        DosePlan = [[0 for i in range(NbPtsX)] for j in range(NbPtsY)]  #On crée une matrice de dose 2D ayant les dimensions y par x.
        RepertoireComplet = name.split("/")                       #On crée la matrice RepertoireComplet dont chaque élément est ce qui se trouve entre chaque /. Ainsi, le dernier élément est le nom du fichier.
        NomFichier = RepertoireComplet[len(RepertoireComplet)-1]  #À l'aide de la longueur de la liste RepertoireComplet, on est capable d'extraire le nom du fichier seulement en indiquant -1, car l'indice de position commence à 0.
       # print(NomFichier.split(".")[2])
        for y in range(NbPtsY):         #On boucle sur chaque ligne de la matrice de dose.
            buffer = MONACO.readline()  #On insère toute la ligne y dans buffer
            for x in range(NbPtsX):     #On boucle sur le nombre de points en x de la matrice de dose.
                DosePlan[y][x] =  float(buffer.split(",")[x])  #Pour chaque séparation avec une virgule, on a une valeur de dose. On stock la matrice de dose du plan dans le fichier dans un array 2D DosePlan[y][x]. Les coordonnées x et y sont échangées pour lire ligne par ligne.
            buffer = MONACO.readline()  #Comme mentionné plus haut, chaque saut de ligne est perçu comme une ligne par la fonction readline. On exécute donc un deuxième buffer avant d'incrémenter y.
        IsoX = int((NbPtsX + 1)/2 - 1)  #Les fichiers exporter par Monaco sont remplit de 0 pour faire une matrice de dose symétrique par rapport à l'isocentre. Le point central de la matrice est donc toujours l'isocentre.
        IsoY = int((NbPtsY + 1)/2 - 1)
        print(DosePlan[IsoY][IsoX])
        #DSS = float(input("Quelle est la DSS du champs?"))
        DSS = 100
        Profile_type = 1
        while Profile_type == 1:
            Type = input("Quel type de profil voulez-vous, PDD, PY ou PX?")
            if (Type == "PDD" or Type == "PX" or Type == "PY"):
                Profile_type = 0
            else:
                pass
        Sens = NomFichier.split(".")[2] #Le fichier exporter par Monaco contient toujours le plan utilisé comme troisième élément séparé par un point dans le nom du fichier lui-même.
        if Type == "PDD":
#            x_startrel = x_endrel = float(input("À quelle position x voulez-vous votre PDD par rapport à l'isocentre"))
#            y_startrel = y_endrel = float(input("À quelle position y voulez-vous votre PDD par rapport à l'isocentre"))
            #z_startrel = float(input("Quelle est la profondeur de départ (cm):"))
            #z_endrel = float(input("Quelle est la profondeur de fin (cm):"))
            x_startrel = x_endrel = 0
            y_startrel = y_endrel = 0
            z_startrel = 0
            z_endrel = 10
        elif Type == "PX":
#            x_startrel = float(input("Quelle est la position x de départ par rapport à l'isocentre (cm):"))
#            x_endrel = float(input("Quelle est la position x de fin par rapport à l'isocentre (cm):"))
#            y_startrel = y_endrel = float(input("À quelle position y voulez-vous votre PX par rapport à l'isocentre"))
#            z_startrel = z_endrel = float(input("À quelle profondeur voulez-vous votre profile de dose:"))
            x_startrel = -24
            x_endrel = 24
            y_startrel = y_endrel = 0
            z_startrel = z_endrel = 10
        else:
#            x_startrel = x_endrel = float(input("À quelle position x voulez-vous votre PY par rapport à l'isocentre"))
#            y_startrel = float(input("Quelle est la position y de départ par rapport à l'isocentre (cm):"))
#            y_endrel = float(input("Quelle est la position y de fin par rapport à l'isocentre (cm):"))
            #z_startrel = z_endrel = float(input("À quelle profondeur voulez-vous votre profile de dose:"))
            x_startrel = x_endrel = 0
            y_startrel = -24
            y_endrel = 24
            z_startrel = z_endrel = 10
        if  Sens == "Transverse":
            x_start = int(x_startrel/0.1) + IsoX
            x_end = int(x_endrel/0.1) + IsoX
            y_start = y_end = 0
            z_start = int(IsoY - (100 - DSS)/0.1 + z_startrel/0.1)
            z_end = int(IsoY - (100-DSS)/0.1 + z_endrel/0.1)
            if Type == "PX":
                Points = abs(x_start - x_end)+1
                Coordonnees = [0 for i in range(Points)]  #On crée une matrice 1D nommé coordonnees qui va contenir la position par rapport à l'isocentre. Les DosePlans extraites de Monaco sont toujours au 1mm, ainsi, on boucle sur la position de départ jusqu'au nombre de points total dans le profil par incrément de 0.1cm.
                for i in range(0, Points, 1):
                    Coordonnees[i] = round(x_startrel + i*0.1,1)    #J'ai ajouté un round avec une seule décimale, car le script sort une dizaine de chiffre après le point et lui donne une valeur parfois, par exemple 1.0000000001 au lieu de 1.0
                filename = str(Type) + ".txt"
                dose_file = open('P:\\Projets\\CRIC\\Physique_Medicale\\VersaHD\\Modelisation Monaco\\Modèle CRIC\\Électrons\\Courbes\\%s' % filename, "w")
                for i in range(x_start, x_start+Points, 1):
                    dose_file.write(str(Coordonnees[i-x_start]).replace(".", ",") + "  " + str(DosePlan[z_start][i]).replace(".", ",")+"\n")   #On stock la position par rapport à l'isocentre, un espace et la dose à ce point.
                dose_file.close()
            else:   #Si PDD
                Points = abs(z_end-z_start)+1
                Coordonnees = [0 for i in range(Points)]
                for i in range(0, Points, 1):
                    Coordonnees[i] = round(z_startrel + i*0.1,1)
                filename = str(Type) + ".txt"
                dose_file = open('P:\\Projets\\CRIC\\Physique_Medicale\\VersaHD\\Modelisation Monaco\\Modèle CRIC\\Électrons\\Courbes\\%s' % filename, "w")
                for i in range(z_start, z_start+Points, 1):
                    dose_file.write(str(Coordonnees[i-z_start]).replace(".", ",") + "  " + str(DosePlan[i][x_start]).replace(".", ",")+"\n")   #On stock la position par rapport à l'isocentre, un espace et la dose à ce point.
                dose_file.close()
        elif Sens == "Coronal":
            x_start = int(x_startrel/0.1) + IsoX
            x_end = int(x_endrel/0.1) + IsoX
            y_start = int(y_startrel/0.1) + IsoY
            y_end = int(y_endrel/0.1) + IsoY
            z_start = z_end = 0
            if Type == "PX":
                Points = abs(x_start - x_end)+1
                Coordonnees = [0 for i in range(Points)]
                for i in range(0, Points, 1):
                    Coordonnees[i] = round(x_startrel + i*0.1,1)
                filename = str(Type) + ".txt"
                dose_file = open('P:\\Projets\\CRIC\\Physique_Medicale\\VersaHD\\Modelisation Monaco\\Modèle CRIC\\Électrons\\Courbes\\%s' % filename, "w")
                for i in range(x_start, x_start+Points, 1):
                    dose_file.write(str(Coordonnees[i-x_start]).replace(".", ",") + "  " + str(DosePlan[y_start][i]).replace(".", ",")+"\n")   #On stock la position par rapport à l'isocentre, un espace et la dose à ce point.
                dose_file.close()
            else:    #Si on veut extraire PY.
                Points = abs(y_start - y_end)+1
                Coordonnees = [0 for i in range(Points)]
                for i in range(0, Points, 1):
                    Coordonnees[i] = round(y_startrel + i*0.1,1)
                filename = str(Type) + ".txt"
                dose_file = open('P:\\Projets\\CRIC\\Physique_Medicale\\VersaHD\\Modelisation Monaco\\Modèle CRIC\\Électrons\\Courbes\\%s' % filename, "w")
                for i in range(y_start, y_start+Points, 1):
                    dose_file.write(str(Coordonnees[i-y_start]).replace(".", ",") + "  " + str(DosePlan[i][x_start]).replace(".", ",")+"\n")   #On stock la position par rapport à l'isocentre, un espace et la dose à ce point.
                dose_file.close()
        else:         #Si le plan utilisé est sagittal. Le comportement de la fonction d'exportation DosePlan est différent. Il fait une rotation par rapport à ce qu'on voit dans Monaco et un réflexion en x, donc le rendement en profondeur suit l'axe y négatif et l'axe y suit l'axe x négatif.
            x_start = x_end =0
            y_start = IsoX - int(y_startrel/0.1)
            y_end = IsoX - int(y_endrel/0.1)
            z_start = int(IsoY - (100 - DSS)/0.1 + z_startrel/0.1)
            z_end = int(IsoY - (100-DSS)/0.1 + z_endrel/0.1)
            if Type == "PY":
                Points = abs(y_start - y_end)+1
                Coordonnees = [0 for i in range(Points)]
                for i in range(0, Points, 1):
                    Coordonnees[i] = round(y_startrel + i*0.1,1)
                filename = str(Type) + ".txt"
                dose_file = open('P:\\Projets\\CRIC\\Physique_Medicale\\VersaHD\\Modelisation Monaco\\Modèle CRIC\\Électrons\\Courbes\\%s' % filename, "w")
                for i in range(y_start, y_end-1, -1):
                    dose_file.write(str(Coordonnees[y_start-i]).replace(".", ",") + "  " + str(DosePlan[z_start][i]).replace(".", ",")+"\n")   #On stock la position par rapport à l'isocentre, un espace et la dose à ce point.
                dose_file.close()
            else:  #Si PDD
                Points = abs(z_start - z_end)+1
                Coordonnees = [0 for i in range(Points)]
                for i in range(0, Points, 1):
                    Coordonnees[i] = round(z_startrel + i*0.1,1)
                filename = str(Type) + ".txt"
                dose_file = open('P:\\Projets\\CRIC\\Physique_Medicale\\VersaHD\\Modelisation Monaco\\Modèle CRIC\\Électrons\\Courbes\\%s' % filename, "w")
                for i in range(z_start, z_start+Points, 1):
                    dose_file.write(str(Coordonnees[i-z_start]).replace(".", ",") + "  " + str(DosePlan[i][y_start]).replace(".", ",")+"\n")   #On stock la position par rapport à l'isocentre, un espace et la dose à ce point.
                dose_file.close()
if __name__ == '__main__':
    main()
