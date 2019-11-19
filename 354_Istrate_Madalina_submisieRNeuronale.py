# coding: utf-8
# pip install numpy scikit-learn

import numpy as np
import os
import time
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

total_time = time.time()

PRIMELE_N_CUVINTE = 10000


def accuracy(y, p):
    return 100 * (y == p).astype('int').mean()


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
        cuvinte_text = text.split()
        date_text.append(cuvinte_text)
    return iduri_text, date_text


dir_path = './date_proiect/'
labels = np.loadtxt(os.path.join(dir_path, 'labels_train.txt'))

train_data_path = os.path.join(dir_path, 'train')
iduri_train, train_data = citeste_texte_din_director(train_data_path)
# print(train_data[0][:10])  # primele 10 cuvinte din primul text

test_data_path = os.path.join(dir_path, "test")
iduri_test, test_data = citeste_texte_din_director(test_data_path)

contor_cuvinte = defaultdict(int)
for doc in train_data:
    for word in doc:
        contor_cuvinte[word] += 1

# transformam dictionarul in lista de tupluri ['cuvant1', frecventa1, 'cuvant2': frecventa2]
perechi_cuvinte_frecventa = list(contor_cuvinte.items())  # [('cuvant1', 150), ('cuvant2', 500), ('cuvant3', 10)]

# sortam descrescator lista de tupluri dupa frecventa
perechi_cuvinte_frecventa = sorted(perechi_cuvinte_frecventa, key=lambda kv: kv[1], reverse=True)  # [('cuvant2', 500), ('cuvant1', 150), ('cuvant3', 10)]

# extragem primele 1000 cele mai frecvente cuvinte din toate textele (+ frecventele lor)
perechi_cuvinte_frecventa = perechi_cuvinte_frecventa[0:PRIMELE_N_CUVINTE]

# print("Primele 10 cele mai frecvente cuvinte ", perechi_cuvinte_frecventa[0:10])


list_of_selected_words = []
for cuvant, frecventa in perechi_cuvinte_frecventa:
    list_of_selected_words.append(cuvant)   # luam doar cuvintele, nu si frecventele


def get_bow(text, lista_de_cuvinte):  # face bag of words peste un singur text
    '''
    returneaza BoW corespunzator unui text impartit in cuvinte
    in functie de lista de cuvinte selectate
    '''
    contor = dict()
    cuvinte = set(lista_de_cuvinte)  # luam cuvintele unice
    for cuvant in cuvinte:
        contor[cuvant] = 0  # pentru fiecare cuvant unic din cele 1000, punem initial frecventa 0
    for cuvant in text:
        if cuvant in cuvinte:
            contor[cuvant] += 1  # crestem frecventele cand e cazul
    return contor  # returnam un dictionar cu cuvintele unice din cele 1000 si frecventele lor


def get_bow_pe_corpus(corpus, lista):  # face bag of words pentru tot setul de date
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
        v = np.array(list(bow_dict.values()))  # in v avem frecventele
        bow[idx] = v
    return bow  # va returna o matrice cu 4480 de linii si 10.000 de coloane.
    # pe fiecare linie avem frecventele corespunzatoare celor 1000 de cuvinte, din cele 4480 de texte


def confusion_matrix(predictii, labels):
    M = np.zeros((11, 11))
    for pred, adevar in zip(np.array(predictii).astype(int), np.array(labels).astype(int)):
        M[adevar, pred] += 1
    return M


def scrie_fisier_submission(nume_fisier, predictii, iduri):
    with open(nume_fisier, 'w') as fout:
        fout.write("Id,Prediction\n")
        for id_text, pred in zip(iduri, predictii):
            fout.write(str(id_text) + ',' + str(int(pred)) + '\n')


def k_fold_cross_validation_mlp(data, k=10):
    kfold = KFold(k, True, 1)
    accuracies = []
    final_confusion_matrix = np.zeros((11, 11))
    for train, test in kfold.split(data):
        train_fold = np.array(data)[train]
        test_fold = np.array(data)[test]
        data_bow_train_fold = get_bow_pe_corpus(train_fold, list_of_selected_words)
        data_bow_test_fold = get_bow_pe_corpus(test_fold, list_of_selected_words)
        clasificator_mlp = MLPClassifier(hidden_layer_sizes = (249,), activation = 'relu', solver='adam', alpha=0.001, max_iter=10000)
        kfold_train_time = time.time()
        clasificator_mlp.fit(data_bow_train_fold, labels[train])  # antrenarea modelului
        predictions = clasificator_mlp.predict(data_bow_test_fold)  # predictiile modelului
        print("Timpul de antrenare pe un fold: %.2f minute" % ((time.time() - kfold_train_time)/ 60))
        acc = accuracy(predictions, labels[test])
        accuracies.append(acc)
        print("Acuratete pe test fold cu MLP", acc)
        M = confusion_matrix(predictions, labels[test])
        # print("Matricea de confuzie: ")
        # print(M)
        final_confusion_matrix += M
    return np.average(accuracies), final_confusion_matrix


mean_accuracy, conf_matrix = k_fold_cross_validation_mlp(train_data)

print('Acuratetea medie in urma k-fold cross validation pe MLP este ', mean_accuracy)
print('Matricea de confuzie este: \n', conf_matrix)

data_bow_train = get_bow_pe_corpus(train_data, list_of_selected_words)
data_bow_test = get_bow_pe_corpus(test_data, list_of_selected_words)

clasificator_mlp = MLPClassifier(hidden_layer_sizes = (249,), activation = 'relu', solver='adam', alpha=0.001, max_iter=10000)
final_train_time = time.time()
clasificator_mlp.fit(data_bow_train, labels)  # antrenarea modelului
predictii_finale = clasificator_mlp.predict(data_bow_test)
print("Timpul de antrenare pe testul final: %.2f minute" % ((time.time() - final_train_time) / 60))

indici_test_final = np.arange(2984, 4481)
scrie_fisier_submission("submisie_rn.csv", predictii_finale, indici_test_final)


print("Timpul de rulare a programului: %.2f minute" % ((time.time() - total_time) / 60))