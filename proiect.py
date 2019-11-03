# coding: utf-8
# pip install numpy scikit-learn

import numpy as np
import re
import os
from collections import defaultdict
from sklearn import svm


PRIMELE_N_CUVINTE = 1000


def accuracy(y, p):
    return 100 * (y==p).astype('int').mean()

def files_in_folder(mypath):
    fisiere = []
    for f in os.listdir(mypath):
        if os.path.isfile(os.path.join(mypath, f)):
            fisiere.append(os.path.join(mypath, f))
    return sorted(fisiere)

def extrage_fisier_fara_extensie(cale_catre_fisier):
    nume_fisier = os.path.basename(cale_catre_fisier)
    nume_fisier_fara_extensie = nume_fisier.replace('.txt', '')
    return nume_fisier_fara_extensie

def citeste_texte_din_director(cale):
    date_text = []
    iduri_text = []
    for fis in files_in_folder(cale):
        id_fis = extrage_fisier_fara_extensie(fis)
        iduri_text.append(id_fis)
        with open(fis, 'r', encoding='utf-8') as fin:
            text = fin.read()

        # incercati cu si fara punctuatie sau cu lowercase
        text_fara_punct = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text)
        cuvinte_text = text_fara_punct.split()
        date_text.append(cuvinte_text)
    return (iduri_text, date_text)

### citim datele ###
dir_path = './date_proiect/'
labels = np.loadtxt(os.path.join(dir_path, 'labels_train.txt'))

train_data_path = os.path.join(dir_path, 'train')
iduri_train, data = citeste_texte_din_director(train_data_path)


print(data[0][:10])
### citim datele ###


### numaram cuvintele din toate documentele ###
contor_cuvinte = defaultdict(int)
for doc in data:
    for word in doc:
        contor_cuvinte[word] += 1

# transformam dictionarul in lista de tupluri ['cuvant1', frecventa1, 'cuvant2': frecventa2]
perechi_cuvinte_frecventa = list(contor_cuvinte.items())

# sortam descrescator lista de tupluri dupa frecventa
perechi_cuvinte_frecventa = sorted(perechi_cuvinte_frecventa, key=lambda kv: kv[1], reverse=True)

# extragem primele 1000 cele mai frecvente cuvinte din toate textele
perechi_cuvinte_frecventa = perechi_cuvinte_frecventa[0:PRIMELE_N_CUVINTE]

print ("Primele 10 cele mai frecvente cuvinte ", perechi_cuvinte_frecventa[0:10])


list_of_selected_words = []
for cuvant, frecventa in perechi_cuvinte_frecventa:
    list_of_selected_words.append(cuvant)
### numaram cuvintele din toate documentele ###


def get_bow(text, lista_de_cuvinte):
    '''
    returneaza BoW corespunzator unui text impartit in cuvinte
    in functie de lista de cuvinte selectate
    '''
    contor = dict()
    cuvinte = set(lista_de_cuvinte)
    for cuvant in cuvinte:
        contor[cuvant] = 0
    for cuvant in text:
        if cuvant in cuvinte:
            contor[cuvant] += 1
    return contor

def get_bow_pe_corpus(corpus, lista):
    '''
    returneaza BoW normalizat
    corespunzator pentru un intreg set de texte
    sub forma de matrice np.array
    '''
    bow = np.zeros((len(corpus), len(lista)))
    for idx, doc in enumerate(corpus):
        bow_dict = get_bow(doc, lista)
        ''' 
            bow e dictionar.
            bow.values() e un obiect de tipul dict_values 
            care contine valorile dictionarului
            trebuie convertit in lista apoi in numpy.array
        '''
        v = np.array(list(bow_dict.values()))
        # incercati si alte tipuri de normalizari
        # v = v / np.sqrt(np.sum(v ** 2))
        bow[idx] = v
    return bow

data_bow = get_bow_pe_corpus(data, list_of_selected_words)
print ("Data bow are shape: ", data_bow.shape)

nr_exemple_train = 2000
nr_exemple_valid = 500
nr_exemple_test = len(data) - (nr_exemple_train + nr_exemple_valid)

indici_train = np.arange(0, nr_exemple_train)
indici_valid = np.arange(nr_exemple_train, nr_exemple_train + nr_exemple_valid)
indici_test = np.arange(nr_exemple_train + nr_exemple_valid, len(data))

print ("Histograma cu clasele din train: ", np.histogram(labels[indici_train])[0])
print ("Histograma cu clasele din validation: ", np.histogram(labels[indici_valid])[0])
print ("Histograma cu clasele din test: ", np.histogram(labels[indici_test])[0])
# clasele sunt balansate in cazul asta pentru train, valid si nr_exemple_test


# cu cat creste C, cu atat clasificatorul este mai predispus sa faca overfit
# https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    clasificator = svm.SVC(C = C, kernel = 'linear')
    clasificator.fit(data_bow[indici_train, :], labels[indici_train])
    predictii = clasificator.predict(data_bow[indici_valid, :])
    print ("Acuratete pe validare cu C =", C, ": ", accuracy(predictii, labels[indici_valid]))


# concatenam indicii de train si validare
# incercati diferite valori pentru C si testati pe datele de test
indici_train_valid = np.concatenate([indici_train, indici_valid])
clasificator = svm.SVC(C = 10, kernel = 'linear')
clasificator.fit(data_bow[indici_train_valid, :], labels[indici_train_valid])
predictii = clasificator.predict(data_bow[indici_test])
print ("Acuratete pe test cu C = 10: ", accuracy(predictii, labels[indici_test]))


def scrie_fisier_submission(nume_fisier, predictii, iduri):
    with open(nume_fisier, 'w') as fout:
        fout.write("Id,Prediction\n")
        for id_text, pred in zip(iduri, predictii):
            fout.write(str(id_text) + ',' + str(int(pred)) + '\n')



scrie_fisier_submission("submisie_C10.csv", predictii, indici_test)

'''
TODO pentru a face un submission pe kaggle:
    1) citesc datele de test si creez data_test_bow dupa modelul de mai sus
    2) antrenez un clasificator pe toate datele 
    3) generez predictii pe data_test_bow
    4) incerc cel putin doua clasificatoare (knn, naive bayes)
    5) fac matrice de confuzie pentru fiecare clasificator
    6) fac cross-validare: https://www.youtube.com/watch?v=sFO2ff-gTh0
    7) citesc cu atentie cerinta proiectului (!!! trebuie si un pdf cu detalii)
'''


