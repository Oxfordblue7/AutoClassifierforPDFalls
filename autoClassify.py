import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from nltk.stem.porter import *
from nltk.corpus import stopwords
from multiprocessing import Pool, Process, Queue, Manager
import matplotlib.pyplot as plt
from pymetamap import MetaMap

st = stopwords.words('english')
stemmer = PorterStemmer()


def loadDataAsDataFrame(f_path):
    '''
        Given a path, loads a data set and puts it into a dataframe
        - simplified mechanism
    '''
    df = pd.read_csv(f_path, dtype=str)
    return df

def preprocess_text(raw_text):
    '''
        Preprocessing function
        PROGRAMMING TIP: Always a good idea to have a *master* preprocessing function that reads in a string and returns the
        preprocessed string after applying a series of functions.
    '''
    #stemming and lowercasing (no stopword removal)
    words = [stemmer.stem(w) for w in raw_text.lower().split()]
    return (" ".join(words))


def vectorize_addFeatures(texts_preprocessed_train, texts_preprocessed_test,
                          locations_preprocessed_train, locations_preprocessed_test, fold_idx):
    vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=10000)
    # n-grams
    training_data_vectors = vectorizer.fit_transform(texts_preprocessed_train).toarray()
    test_data_vectors = vectorizer.transform(texts_preprocessed_test).toarray()

    # feature: locations
    vectorizer_loc = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=10000)
    locations_train_vectors = vectorizer_loc.fit_transform(locations_preprocessed_train).toarray()
    locations_test_vectors = vectorizer_loc.transform(locations_preprocessed_test).toarray()
    features_vectors[fold_idx]['locations_train'] = locations_train_vectors
    features_vectors[fold_idx]['locations_test'] = locations_test_vectors

    training_data_vectors = np.concatenate((training_data_vectors, locations_train_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, locations_test_vectors), axis=1)

    # features: cuis, semtypes, preferred_name
    mm = MetaMap.get_instance('/Users/hx/Downloads/public_mm/bin/metamap18')
    cuis_train = []
    semtypes_train = []
    preferredNames_train = []
    for tx in texts_preprocessed_train:
        concepts, errors = mm.extract_concepts([tx])
        cuis = []
        semtypes = []
        preferredNames = []
        for c in concepts:
            # print(c.score, c.preferred_name, c.cui, c.semtypes)
            if float(c.score) > 9.0:
                cuis.append(c.cui)
                semtypes += [s for s in c.semtypes.strip('[').strip(']').split(',')]
                preferredNames.append(c.preferred_name)
            else:
                break
        if len(cuis) == 0:
            cuis.append(concepts[0].cui)
            semtypes += [s for s in concepts[0].semtypes.strip('[').strip(']').split(',')]
            preferredNames.append(concepts[0].preferred_name)

        cuis_train.append(" ".join(cuis))
        semtypes_train.append(" ".join(semtypes))
        preferredNames_train.append(" ".join(preferredNames))
        # concepts_forFolds[fold_idx] = {'cuis_train': cuis_train, 'semtypes_train': semtypes_train, 'preferredNames_train': preferredNames_train}

    cuis_test = []
    semtypes_test = []
    preferredNames_test = []
    for tx in texts_preprocessed_test:
        concepts, errors = mm.extract_concepts([tx])
        cuis = []
        semtypes = []
        preferredNames = []
        for c in concepts:
            # print(c.score, c.preferred_name, c.cui, c.semtypes)
            if float(c.score) > 9.0:
                cuis.append(c.cui)
                semtypes += [s for s in c.semtypes.strip('[').strip(']').split(',')]
                preferredNames.append(c.preferred_name)
            else:
                break
        if len(cuis) == 0:
            cuis.append(concepts[0].cui)
            semtypes += [s for s in concepts[0].semtypes.strip('[').strip(']').split(',')]
            preferredNames.append(concepts[0].preferred_name)

        cuis_test.append(" ".join(cuis))
        semtypes_test.append(" ".join(semtypes))
        preferredNames_test.append(" ".join(preferredNames))
        # concepts_forFolds[fold_idx] = {'cuis_test': cuis_test, 'semtypes_test': semtypes_test, 'preferredNames_test': preferredNames_test}

    vectorizer_cui = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=10000)
    cuis_train_vectors = vectorizer_cui.fit_transform(cuis_train).toarray()
    cuis_test_vectors = vectorizer_cui.transform(cuis_test).toarray()
    features_vectors[fold_idx]['cuis_train'] = cuis_train_vectors
    features_vectors[fold_idx]['cuis_test'] = cuis_test_vectors
    training_data_vectors = np.concatenate((training_data_vectors, cuis_train_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, cuis_test_vectors), axis=1)

    vectorizer_semtype = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=10000)
    semtypes_train_vectors = vectorizer_semtype.fit_transform(semtypes_train).toarray()
    semtypes_test_vectors = vectorizer_semtype.transform(semtypes_test).toarray()
    features_vectors[fold_idx]['semtypes_train'] = semtypes_train_vectors
    features_vectors[fold_idx]['semtypes_test'] = semtypes_test_vectors
    training_data_vectors = np.concatenate((training_data_vectors, semtypes_train_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, semtypes_test_vectors), axis=1)

    vectorizer_prefName = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=10000)
    preferredNames_train_vectors = vectorizer_prefName.fit_transform(preferredNames_train).toarray()
    preferredNames_test_vectors = vectorizer_prefName.transform(preferredNames_test).toarray()
    features_vectors[fold_idx]['preferredNames_train'] = preferredNames_train_vectors
    features_vectors[fold_idx]['preferredNames_test'] = preferredNames_test_vectors
    training_data_vectors = np.concatenate((training_data_vectors, preferredNames_train_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, preferredNames_test_vectors), axis=1)

    return training_data_vectors, test_data_vectors


def train_gnb(training_data_vectors, val_data_vectors, ttp_train, ttp_val):
    gnb_classifier = GaussianNB()
    gnb_classifier = gnb_classifier.fit(training_data_vectors, ttp_train)
    predictions = gnb_classifier.predict(val_data_vectors)

    accs_gnb.append(accuracy_score(predictions, ttp_val))
    f1micro_gnb.append(f1_score(predictions, ttp_val, average='micro'))
    f1macro_gnb.append(f1_score(predictions, ttp_val, average='macro'))

def train_ridge(training_data_vectors, val_data_vectors, ttp_train, ttp_val):
    for alpha in [0.1, 0.5, 1, 10, 50]:
        ridge_classifier = RidgeClassifier(alpha=alpha)
        ridge_classifier = ridge_classifier.fit(training_data_vectors, ttp_train)
        predictions = ridge_classifier.predict(val_data_vectors)

        accs_ridge[alpha].append(accuracy_score(predictions, ttp_val))
        f1micro_ridge[alpha].append(f1_score(predictions, ttp_val, average='micro'))
        f1macro_ridge[alpha].append(f1_score(predictions, ttp_val, average='macro'))

def train_randForest(training_data_vectors, val_data_vectors, ttp_train, ttp_val):
    for n in [10, 30, 50, 70, 100, 120]:
        rf_classifier = RandomForestClassifier(n_estimators=n, n_jobs=-1)
        rf_classifier = rf_classifier.fit(training_data_vectors, ttp_train)
        predictions = rf_classifier.predict(val_data_vectors)

        accs_rf[n].append(accuracy_score(predictions, ttp_val))
        f1micro_rf[n].append(f1_score(predictions, ttp_val, average='micro'))
        f1macro_rf[n].append(f1_score(predictions, ttp_val, average='macro'))

def train_knn(training_data_vectors, val_data_vectors, ttp_train, ttp_val):
    for k in range(1, 10):
        knn_classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn_classifier = knn_classifier.fit(training_data_vectors, ttp_train)
        predictions = knn_classifier.predict(val_data_vectors)

        accs_knn[k].append(accuracy_score(predictions, ttp_val))
        f1micro_knn[k].append(f1_score(predictions, ttp_val, average='micro'))
        f1macro_knn[k].append(f1_score(predictions, ttp_val, average='macro'))

def train_svm(training_data_vectors, val_data_vectors, ttp_train, ttp_val):
    for kernel in ['linear', 'rbf']:
        for c in [0.5, 1, 2, 4, 8, 16, 32, 64, 128]:
            svm_classifier = svm.SVC(kernel=kernel, C=c)
            svm_classifier = svm_classifier.fit(training_data_vectors, ttp_train)
            predictions = svm_classifier.predict(val_data_vectors)

            accs_svm[(kernel, c)].append(accuracy_score(predictions, ttp_val))
            f1micro_svm[(kernel, c)].append(f1_score(predictions, ttp_val, average='micro'))
            f1macro_svm[(kernel, c)].append(f1_score(predictions, ttp_val, average='macro'))

def train_mlp(training_data_vectors, val_data_vectors, ttp_train, ttp_val):
    for ls in [(20,), (50,), (100,), (50, 50), (100, 100)]:
        mlp_classifier = MLPClassifier(hidden_layer_sizes=ls)
        mlp_classifier = mlp_classifier.fit(training_data_vectors, ttp_train)
        predictions = mlp_classifier.predict(val_data_vectors)

        accs_mlp[ls].append(accuracy_score(predictions, ttp_val))
        f1micro_mlp[ls].append(f1_score(predictions, ttp_val, average='micro'))
        f1macro_mlp[ls].append(f1_score(predictions, ttp_val, average='macro'))


def find_bestHyperParas(score_dict):
    best_score = 0
    for para, scores in score_dict.items():
        s = np.mean(scores)
        if s > best_score:
            best_score = s
            best_para = para

    print("best hyperparameter: {}; best f1-micro score: {}".format(best_para, best_score))
    return best_para, best_score


def initialize_bestClassifier(best_clf_name, best_clf_para, num_folds, num_keys):
    if num_keys > 0:
        if best_clf_name == 'Gaussian NB classifier':
            clfs = [[GaussianNB() for i in range(num_folds)] for n in range(num_keys)]
        elif best_clf_name == 'Ridge classifier':
            clfs = [[RidgeClassifier() for i in range(num_folds)] for n in range(num_keys)]
        elif best_clf_name == 'Random Forest classifier':
            clfs = [[RandomForestClassifier(n_estimators=best_clf_para, n_jobs=-1) for i in range(num_folds)] for n in range(num_keys)]
        elif best_clf_name == 'K-Nearest Neighbor classifier':
            clfs = [[KNeighborsClassifier(n_neighbors=best_clf_para, n_jobs=-1) for i in range(num_folds)] for n in range(num_keys)]
        elif best_clf_name == 'SVM classifier':
            clfs = [[svm.SVC(kernel=best_clf_para[0], C=best_clf_para[1]) for i in range(num_folds)] for n in range(num_keys)]
        elif best_clf_name == 'MLP classifier':
            clfs = [[MLPClassifier(hidden_layer_sizes=best_clf_para) for i in range(num_folds)] for n in range(num_keys)]

        return clfs
    else:
        if best_clf_name == 'Gaussian NB classifier':
            clfs = [GaussianNB() for i in range(num_folds)]
        elif best_clf_name == 'Ridge classifier':
            clfs = [RidgeClassifier() for i in range(num_folds)]
        elif best_clf_name == 'Random Forest classifier':
            clfs = [RandomForestClassifier(n_estimators=best_clf_para, n_jobs=-1) for i in range(num_folds)]
        elif best_clf_name == 'K-Nearest Neighbor classifier':
            clfs = [KNeighborsClassifier(n_neighbors=best_clf_para, n_jobs=-1) for i in range(num_folds)]
        elif best_clf_name == 'SVM classifier':
            clfs = [svm.SVC(kernel=best_clf_para[0], C=best_clf_para[1]) for i in range(num_folds)]
        elif best_clf_name == 'MLP classifier':
            clfs = [MLPClassifier(hidden_layer_sizes=best_clf_para) for i in range(num_folds)]

        return clfs


def ablation_features(allFolds, best_clf_name, best_clf_para):
    accs, f1micros, f1macros = defaultdict(list), defaultdict(list), defaultdict(list)
    df = pd.DataFrame()
    num_folds = len(allFolds)
    clfs_dict = {}
    clfs = initialize_bestClassifier(best_clf_name, best_clf_para, num_folds, 4)
    clfs_dict['locations_removed'] = clfs[0]
    clfs_dict['cuis_removed'] = clfs[1]
    clfs_dict['semtypes_removed'] = clfs[2]
    clfs_dict['preferredNames_removed'] = clfs[3]

    for abl in ['locations_removed', 'cuis_removed', 'semtypes_removed', 'preferredNames_removed']:
        for i in range(num_folds):
            fold_idx = i+1
            texts_train_preprocessed_train, texts_train_preprocessed_val, locations_train_preprocessed_train, \
            locations_train_preprocessed_val, ttp_train, ttp_test = allFolds[i]

            vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, preprocessor=None,
                                         max_features=10000)
            # n-grams
            training_data_vectors = vectorizer.fit_transform(texts_train_preprocessed_train).toarray()
            test_data_vectors = vectorizer.transform(texts_train_preprocessed_val).toarray()

            # feature: locations
            if abl != 'locations_removed':
                training_data_vectors = np.concatenate((training_data_vectors, features_vectors[fold_idx]['locations_train']), axis=1)
                test_data_vectors = np.concatenate((test_data_vectors, features_vectors[fold_idx]['locations_test']), axis=1)

            # feature: cuis
            if abl != 'cuis_removed':
                training_data_vectors = np.concatenate((training_data_vectors, features_vectors[fold_idx]['cuis_train']), axis=1)
                test_data_vectors = np.concatenate((test_data_vectors, features_vectors[fold_idx]['cuis_test']), axis=1)

            if abl != 'semtypes_removed':
                training_data_vectors = np.concatenate((training_data_vectors, features_vectors[fold_idx]['semtypes_train']), axis=1)
                test_data_vectors = np.concatenate((test_data_vectors, features_vectors[fold_idx]['semtypes_test']), axis=1)

            if abl != 'preferredNames_removed':
                training_data_vectors = np.concatenate((training_data_vectors, features_vectors[fold_idx]['preferredNames_train']), axis=1)
                test_data_vectors = np.concatenate((test_data_vectors, features_vectors[fold_idx]['preferredNames_test']), axis=1)

            clfs_dict[abl][i].fit(training_data_vectors, ttp_train)
            predictions = clfs_dict[abl][i].predict(test_data_vectors)

            accs[abl].append(accuracy_score(predictions, ttp_test))
            f1micros[abl].append(f1_score(predictions, ttp_test, average='micro'))
            f1macros[abl].append(f1_score(predictions, ttp_test, average='macro'))

        df.loc[abl, 'accuracy'] = np.mean(accs[abl])
        df.loc[abl, 'f1-micro'] = np.mean(f1micros[abl])
        df.loc[abl, 'f1-macro'] = np.mean(f1macros[abl])

    print("MLP - accuracy: ", accs, '\n', "MLP - f1-micro: ", f1micros, '\n', "MLP - f1-macro: ", f1macros)
    df.to_csv("outputs/output_scores_ablateFeatures", sep='\t')


def foreach_trainSize(s):
    scale = round(s * totalTrainLen)
    # trainSizes.append(scale)
    training_texts_subset = texts_train_preprocessed[:scale]
    training_locations_subset = locations_train_preprocessed[:scale]
    training_classes_subset = classes_train_preprocessed[:scale]

    vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=10000)
    training_data_vectors = vectorizer.fit_transform(training_texts_subset).toarray()
    test_data_vectors = vectorizer.transform(texts_test_preprocessed).toarray()

    vectorizer_loc = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, preprocessor=None,
                                     max_features=10000)
    locations_train_vectors = vectorizer_loc.fit_transform(training_locations_subset).toarray()
    locations_test_vectors = vectorizer_loc.transform(locations_test_preprocessed).toarray()
    training_data_vectors = np.concatenate((training_data_vectors, locations_train_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, locations_test_vectors), axis=1)

    mm = MetaMap.get_instance('/Users/hx/Downloads/public_mm/bin/metamap18')
    cuis_train = []
    semtypes_train = []
    preferredNames_train = []
    for tx in training_texts_subset:
        concepts, errors = mm.extract_concepts([tx])
        cuis = []
        semtypes = []
        preferredNames = []
        for c in concepts:
            # print(c.score, c.preferred_name, c.cui, c.semtypes)
            if float(c.score) > 9.0:
                cuis.append(c.cui)
                semtypes += [s for s in c.semtypes.strip('[').strip(']').split(',')]
                preferredNames.append(c.preferred_name)
            else:
                break
        if len(cuis) == 0:
            cuis.append(concepts[0].cui)
            semtypes += [s for s in concepts[0].semtypes.strip('[').strip(']').split(',')]
            preferredNames.append(concepts[0].preferred_name)

        cuis_train.append(" ".join(cuis))
        semtypes_train.append(" ".join(semtypes))
        preferredNames_train.append(" ".join(preferredNames))

    cuis_test = []
    semtypes_test = []
    preferredNames_test = []
    for tx in texts_test_preprocessed:
        concepts, errors = mm.extract_concepts([tx])
        cuis = []
        semtypes = []
        preferredNames = []
        for c in concepts:
            # print(c.score, c.preferred_name, c.cui, c.semtypes)
            if float(c.score) > 9.0:
                cuis.append(c.cui)
                semtypes += [s for s in c.semtypes.strip('[').strip(']').split(',')]
                preferredNames.append(c.preferred_name)
            else:
                break
        if len(cuis) == 0:
            cuis.append(concepts[0].cui)
            semtypes += [s for s in concepts[0].semtypes.strip('[').strip(']').split(',')]
            preferredNames.append(concepts[0].preferred_name)

        cuis_test.append(" ".join(cuis))
        semtypes_test.append(" ".join(semtypes))
        preferredNames_test.append(" ".join(preferredNames))

    vectorizer_cui = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=10000)
    cuis_train_vectors = vectorizer_cui.fit_transform(cuis_train).toarray()
    cuis_test_vectors = vectorizer_cui.transform(cuis_test).toarray()
    training_data_vectors = np.concatenate((training_data_vectors, cuis_train_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, cuis_test_vectors), axis=1)

    vectorizer_semtype = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=10000)
    semtypes_train_vectors = vectorizer_semtype.fit_transform(semtypes_train).toarray()
    semtypes_test_vectors = vectorizer_semtype.transform(semtypes_test).toarray()
    training_data_vectors = np.concatenate((training_data_vectors, semtypes_train_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, semtypes_test_vectors), axis=1)

    vectorizer_prefName = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=10000)
    preferredNames_train_vectors = vectorizer_prefName.fit_transform(preferredNames_train).toarray()
    preferredNames_test_vectors = vectorizer_prefName.transform(preferredNames_test).toarray()
    training_data_vectors = np.concatenate((training_data_vectors, preferredNames_train_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, preferredNames_test_vectors), axis=1)

    clfs = initialize_bestClassifier(best_classifier_name, best_hyperparas, num_folds, 0)
    clf = MLPClassifier(hidden_layer_sizes=(20,))
    clf.fit(training_data_vectors, training_classes_subset)
    predictions = clf.predict(test_data_vectors)

    print("scale: {}; train size: {} -- accuracy = {}; f1-micro = {}; f1-macro = {}".format(
        s, scale, accuracy_score(predictions, classes_test_preprocessed), f1_score(predictions, classes_test_preprocessed, average='micro'),
        f1_score(predictions, classes_test_preprocessed, average='macro')))


if __name__ == '__main__':
    # Load the data
    print("Preparing data ... ")
    f_path = './pdfalls.csv'
    data_all = loadDataAsDataFrame(f_path)

    # SPLIT THE DATA
    data_train, data_test = train_test_split(data_all, test_size=0.2)
    # print(data_train, data_test)

    ids_train = data_train['record_id']
    texts_train = data_train['fall_description']
    classes_train = data_train['fall_class']
    ages_train = data_train['age']
    # genders_train = data_train['female']
    locations_train = data_train['fall_location']

    texts_test = data_test['fall_description']
    classes_test = data_test['fall_class']
    ages_test = data_test['age']
    # genders_test = data_test['female']
    locations_test = data_test['fall_location']

    # PREPROCESS THE DATA
    texts_train_preprocessed = [preprocess_text(tr) for tr in texts_train]
    texts_test_preprocessed = [preprocess_text(te) for te in texts_test]
    locations_train_preprocessed = [preprocess_text(tr) for tr in locations_train]
    locations_test_preprocessed = [preprocess_text(te) for te in locations_test]

    classes_train_preprocessed = ['CoM' if x == 'CoM' else 'Other' for x in classes_train]
    classes_test_preprocessed = ['CoM' if x == 'CoM' else 'Other' for x in classes_test]

    # print(texts_train_preprocessed, '\n', locations_train_preprocessed, '\n', classes_train_preprocessed)
    # print(texts_test_preprocessed, '\n', locations_test_preprocessed, '\n', classes_test_preprocessed)

    # Evaluate CLASSIFIERS (CROSS VALIDATION)
    accs_gnb, f1micro_gnb, f1macro_gnb = [], [], []
    accs_ridge, f1micro_ridge, f1macro_ridge = defaultdict(list), defaultdict(list), defaultdict(list)
    accs_rf, f1micro_rf, f1macro_rf = defaultdict(list), defaultdict(list), defaultdict(list)
    accs_knn, f1micro_knn, f1macro_knn = defaultdict(list), defaultdict(list), defaultdict(list)
    accs_svm, f1micro_svm, f1macro_svm = defaultdict(list), defaultdict(list), defaultdict(list)
    accs_mlp, f1micro_mlp, f1macro_mlp = defaultdict(list), defaultdict(list), defaultdict(list)
    # split data
    num_folds = 10
    allFolds = []
    allFolds_afterVects = []
    # concepts_forFolds = {}
    features_vectors = defaultdict(dict)
    skf = StratifiedKFold(n_splits=num_folds)
    fold_i = 1
    for train_index, test_index in skf.split(texts_train_preprocessed, classes_train_preprocessed):
        # texts
        texts_train_preprocessed_train = np.array(texts_train_preprocessed)[train_index]
        texts_train_preprocessed_val = np.array(texts_train_preprocessed)[test_index]
        # locations
        locations_train_preprocessed_train = np.array(locations_train_preprocessed)[train_index]
        locations_train_preprocessed_val = np.array(locations_train_preprocessed)[test_index]
        # classes
        ttp_train, ttp_val = np.array(classes_train_preprocessed)[train_index], np.array(classes_train_preprocessed)[test_index]

        allFolds.append((texts_train_preprocessed_train, texts_train_preprocessed_val,
                         locations_train_preprocessed_train, locations_train_preprocessed_val,
                         ttp_train, ttp_val))
        # vectorize and add features
        training_data_vectors, val_data_vectors = vectorize_addFeatures(texts_train_preprocessed_train,
                                                                        texts_train_preprocessed_val,
                                                                        locations_train_preprocessed_train,
                                                                        locations_train_preprocessed_val, fold_i)
        fold_i += 1
        allFolds_afterVects.append((training_data_vectors, val_data_vectors, ttp_train, ttp_val))

    print("Evaluating ...")
    for training_data_vectors, val_data_vectors, ttp_train, ttp_val in allFolds_afterVects:
        # Baseline: Gaussian Naive Bayes Classifier
        train_gnb(training_data_vectors, val_data_vectors, ttp_train, ttp_val)

        # Classifier using Ridge Regression
        # for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        train_ridge(training_data_vectors, val_data_vectors, ttp_train, ttp_val)

        # Random Forest Classifier
        # hyper-para: n_estimators = [10, 50, 70, 100, 120]
        train_randForest(training_data_vectors, val_data_vectors, ttp_train, ttp_val)

        # K Nearest Neighbor Classifier
        # hyper-para: k = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        train_knn(training_data_vectors, val_data_vectors, ttp_train, ttp_val)

        # SVM Classifier
        # hyper-paras: c = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]; kernel = ['linear', 'rbf']
        train_svm(training_data_vectors, val_data_vectors, ttp_train, ttp_val)

        # # MLP classifier
        # # hyper-para: hidden_layer_sizes = [(20,), (50,), (100,), (200,), (100, 100)]
        train_mlp(training_data_vectors, val_data_vectors, ttp_train, ttp_val)

    print("GNB - accuracy: ", accs_gnb, '\n', "GNB - f1-micro: ", f1micro_gnb, '\n', "GNB - f1-macro: ", f1macro_gnb)
    print("Ridge - accuracy: ", accs_ridge, '\n', "Ridge - f1-micro: ", f1micro_ridge, '\n', "Ridge - f1-macro: ", f1macro_ridge)
    print("RandomForest - accuracy: ", accs_rf, '\n', "RandomForest - f1-micro: ", f1micro_rf, '\n', "RandomForest - f1-macro: ", f1macro_rf)
    print("KNN - accuracy: ", accs_knn, '\n', "KNN - f1-micro: ", f1micro_knn, '\n', "KNN - f1-macro: ", f1macro_knn)
    print("SVM - accuracy: ", accs_svm, '\n', "SVM - f1-micro: ", f1micro_svm, '\n', "SVM - f1-macro: ", f1macro_svm)
    print("MLP - accuracy: ", accs_mlp, '\n', "MLP - f1-micro: ", f1micro_mlp, '\n', "MLP - f1-macro: ", f1macro_mlp)

    # Identity the best classifier (based on f1-micro)
    print("Identify the best classifier ... ")
    df_scores = pd.DataFrame()
    best_f1micro = 0
    # GNB
    gnb_best_score = np.mean(f1micro_gnb)
    df_scores.loc['GNB', 'accuracy'] = np.mean(accs_gnb)
    df_scores.loc['GNB', 'f1-micro'] = gnb_best_score
    df_scores.loc['GNB', 'f1-macro'] = np.mean(f1macro_gnb)
    if gnb_best_score > best_f1micro:
        best_f1micro = gnb_best_score
        best_classifier_name = 'Gaussian NB classifier'
        best_classifier = GaussianNB()
    # Ridge
    ridge_best_para, ridge_best_score = find_bestHyperParas(f1micro_ridge)
    df_scores.loc['Ridge', 'accuracy'] = np.mean(accs_ridge[ridge_best_para])
    df_scores.loc['Ridge', 'f1-micro'] = ridge_best_score
    df_scores.loc['Ridge', 'f1-macro'] = np.mean(f1macro_ridge[ridge_best_para])
    if ridge_best_score > best_f1micro:
        best_f1micro = ridge_best_score
        best_classifier_name = 'Ridge classifier'
        best_classifier = RidgeClassifier()
    # RandomForest
    print("** RandomForest")
    rf_best_para, rf_best_score = find_bestHyperParas(f1micro_rf)
    df_scores.loc['RandomForest', 'accuracy'] = np.mean(accs_rf[rf_best_para])
    df_scores.loc['RandomForest', 'f1-micro'] = rf_best_score
    df_scores.loc['RandomForest', 'f1-macro'] = np.mean(f1macro_rf[rf_best_para])
    if rf_best_score > best_f1micro:
        best_f1micro = rf_best_score
        best_hyperparas = rf_best_para
        best_classifier_name = 'Random Forest classifier'
        best_classifier = RandomForestClassifier(n_estimators=rf_best_para, n_jobs=-1)
    # KNN
    print("** KNN")
    knn_best_para, knn_best_score = find_bestHyperParas(f1micro_knn)
    df_scores.loc['KNN', 'accuracy'] = np.mean(accs_knn[knn_best_para])
    df_scores.loc['KNN', 'f1-micro'] = knn_best_score
    df_scores.loc['KNN', 'f1-macro'] = np.mean(f1macro_knn[knn_best_para])
    if knn_best_score > best_f1micro:
        best_f1micro = knn_best_score
        best_hyperparas = knn_best_para
        best_classifier_name = 'K-Nearest Neighbor classifier'
        best_classifier = KNeighborsClassifier(n_neighbors=knn_best_para, n_jobs=-1)
    # SVM
    print("** SVM")
    svm_best_para, svm_best_score = find_bestHyperParas(f1micro_svm)
    df_scores.loc['SVM', 'accuracy'] = np.mean(accs_svm[svm_best_para])
    df_scores.loc['SVM', 'f1-micro'] = svm_best_score
    df_scores.loc['SVM', 'f1-macro'] = np.mean(f1macro_svm[svm_best_para])
    if svm_best_score > best_f1micro:
        best_f1micro = svm_best_score
        best_hyperparas = svm_best_para
        best_classifier_name = 'SVM classifier'
        best_classifier = svm.SVC(kernel=svm_best_para[0], C=svm_best_para[1])
    # MLP
    print("** MLP")
    mlp_best_para, mlp_best_score = find_bestHyperParas(f1micro_mlp)
    df_scores.loc['MLP', 'accuracy'] = np.mean(accs_mlp[mlp_best_para])
    df_scores.loc['MLP', 'f1-micro'] = mlp_best_score
    df_scores.loc['MLP', 'f1-macro'] = np.mean(f1macro_mlp[mlp_best_para])
    if mlp_best_score > best_f1micro:
        best_f1micro = mlp_best_score
        best_hyperparas = mlp_best_para
        best_classifier_name = 'MLP classifier'
        best_classifier = MLPClassifier(hidden_layer_sizes=mlp_best_para)

    print("The best single classifier is: {}".format(best_classifier_name))

    # Voting ensemble classifier with GNB, optimized RandomForest, and SVM
    accs_vote = []
    f1micro_vote = []
    f1macro_vote = []
    for training_data_vectors, val_data_vectors, ttp_train, ttp_val in allFolds_afterVects:
        vote_classifier = VotingClassifier(estimators=[('gnb', GaussianNB()), ('svm', svm.SVC(kernel=svm_best_para[0], C=svm_best_para[1])),
                                                       ('rf', RandomForestClassifier(n_estimators=rf_best_para, n_jobs=-1))])

        vote_classifier.fit(training_data_vectors, ttp_train)
        predictions = vote_classifier.predict(val_data_vectors)

        accs_vote.append(accuracy_score(predictions, ttp_val))
        f1micro_vote.append(f1_score(predictions, ttp_val, average='micro'))
        f1macro_vote.append(f1_score(predictions, ttp_val, average='macro'))

    print("Voting - accuracy: ", accs_vote, '\n', "Voting - f1-micro: ", f1micro_vote, '\n', "Voting - f1-macro: ", f1macro_vote)

    df_scores.loc['Voting', 'accuracy'] = np.mean(accs_vote)
    vote_best_score = np.mean(f1micro_vote)
    df_scores.loc['Voting', 'f1-micro'] = vote_best_score
    df_scores.loc['Voting', 'f1-macro'] = np.mean(f1macro_vote)

    df_scores.to_csv("outputs/output_scores_allClassifiers", sep='\t')

    # compare voting classifier with the best single classifier
    if vote_best_score > best_f1micro:
        best_f1micro = vote_best_score
        best_classifier_name = "Voting ensemble classifier"
        best_classifier = vote_classifier = VotingClassifier(estimators=[('gnb', GaussianNB()), ('svm', svm.SVC(kernel=svm_best_para[0], C=svm_best_para[1])),
                                                       ('rf', RandomForestClassifier(n_estimators=rf_best_para, n_jobs=-1))])
        print("The voting ensemble classifier outperformed all single classifier.")
    else:
        print("The best single classifier {} outperformed the voting ensemble classifier.".format(best_classifier_name))


    # FURTHER EVALUATING WITH THE BEST CLASSIFIER
    # use the best classifier to evaluate the feature set combination
    print("Ablation study ...")
    ablation_features(allFolds, best_classifier_name, best_hyperparas)


    # train size vs performance
    # 20% test dataset
    print("Training size versus Performance")
    scales = [0.2, 0.4, 0.6, 0.8, 1.0]
    totalTrainLen = len(texts_train_preprocessed)
    # trainSizes = []
    with Pool(10) as p:
        p.map(foreach_trainSize, scales)




