# coding=utf-8
import numpy as np
from numpy import *
from numpy import linalg as la
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,AdaBoostClassifier
from xgboost.sklearn import XGBClassifier


def read_fasta_file(fasta_file):
    seq_dict = {}
    fp = open(fasta_file, 'r')
    name = ''
    # pdb.set_trace()
    for line in fp:
        # let's discard the newline at the end (if any)
        line = line.rstrip()
        # distinguish header from sequence
        if line[0] == '>':  # or line.startswith('>')
            # it is the header
            name = line[1:].upper()  # discarding the initial >
            seq_dict[name] = ''
        else:
            # it is sequence
            seq_dict[name] = seq_dict[name] + line
    fp.close()

    return seq_dict


def TransDict_from_list(groups):
    transDict = dict()
    tar_list = ['0', '1', '2', '3', '4', '5', '6']
    result = {}
    index = 0
    for group in groups:
        g_members = sorted(group)  # Alphabetically sorted list
        for c in g_members:
            # print('c' + str(c))
            # print('g_members[0]' + str(g_members[0]))
            result[c] = str(tar_list[index])  # K:V map, use group's first letter as represent.
        index = index + 1
    return result


def get_3_protein_trids():
    nucle_com = []
    chars = ['0', '1', '2', '3', '4', '5', '6']
    base = len(chars)
    end = len(chars) ** 3
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n / base
        ch1 = chars[int(n % base)]
        n = n / base
        ch2 = chars[int(n % base)]
        nucle_com.append(ch0 + ch1 + ch2)
    return nucle_com


def get_4_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 4
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n / base
        ch1 = chars[int(n % base)]
        n = n / base
        ch2 = chars[int(n % base)]
        n = n / base
        ch3 = chars[int(n % base)]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    return nucle_com


def translate_sequence(seq, TranslationDict):
    '''
    Given (seq) - a string/sequence to translate,
    Translates into a reduced alphabet, using a translation dict provided
    by the TransDict_from_list() method.
    Returns the string/sequence in the new, reduced alphabet.
    Remember - in Python string are immutable..

    '''
    import string
    from_list = []
    to_list = []
    for k, v in TranslationDict.items():
        from_list.append(k)
        to_list.append(v)
    # TRANS_seq = seq.translate(str.maketrans(zip(from_list,to_list)))
    TRANS_seq = seq.translate(str.maketrans(str(from_list), str(to_list)))
    # TRANS_seq = maketrans( TranslationDict, seq)
    return TRANS_seq


def get_RNA_seq_concolutional_array(seq, motif_len=4):
    # data = {}
    alpha = 'ACGT'
    # for seq in seqs:
    # for key, seq in seqs.iteritems():
    row = (len(seq) + 2 * motif_len - 2)
    new_array = np.zeros((row, 4))
    for i, val in enumerate(seq):
        if val not in 'ACGTN':
            new_array[i] = np.array([0.25] * 4)
            continue
        if val == 'N' or i < motif_len or i > len(seq) - motif_len:
            new_array[i] = np.array([0.25] * 4)
        else:
            index = alpha.index(val)
            new_array[i][index] = 1
        # data[key] = new_array
    return new_array


def get_4_nucleotide_composition(tris, seq, pythoncount=True):
    seq_len = len(seq)
    tri_feature = [0] * len(tris)
    k = len(tris[0])
    note_feature = [[0 for cols in range(len(seq) - k + 1)] for rows in range(len(tris))]
    if pythoncount:
        for val in tris:
            num = seq.count(val)
            tri_feature.append(float(num) / seq_len)
    else:
        # tmp_fea = [0] * len(tris)
        for x in range(len(seq) + 1 - k):
            kmer = seq[x:x + k]
            if kmer in tris:
                ind = tris.index(kmer)
                # tmp_fea[ind] = tmp_fea[ind] + 1
                note_feature[ind][x] = note_feature[ind][x] + 1
        # tri_feature = [float(val)/seq_len for val in tmp_fea]    #tri_feature type:list len:256
        u, s, v = la.svd(note_feature)
        for i in range(len(s)):
            tri_feature = tri_feature + u[i] * s[i] / seq_len
        # print tri_feature
        # pdb.set_trace()

    return tri_feature


def prepare_RPI488_feature(extract_only_posi=False,
                           pseaac_file=None, deepmind=False, seperate=False, chem_fea=True):
    print ('RPI488 dataset')
    dbName = '488.mat'
    interaction_pair = {}
    RNA_seq_dict = {}
    protein_seq_dict = {}
    with open('ncRNA-protein/lncRNA-protein-488.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                label = values[1]
                name = values[0].split('_')
                protein = name[0] + '-' + name[1]
                RNA = name[0] + '-' + name[2]
                if label == 'interactive':
                    interaction_pair[(protein, RNA)] = 1
                else:
                    interaction_pair[(protein, RNA)] = 0
                index = 0
            else:
                seq = line[:-1]
                if index == 0:
                    protein_seq_dict[protein] = seq
                else:
                    RNA_seq_dict[RNA] = seq
                index = index + 1
    # name_list = read_name_from_lncRNA_fasta('ncRNA-protein/lncRNA_RNA.fa')
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    tris = get_4_trids()
    # tris3 = get_3_trids()
    train = []
    label = []
    chem_fea = []
    proteinFea = []
    # get protein feature
    tempStruct = scipy.io.loadmat(dbName)
    tempfea = tempStruct['fea']
    proteinFea = tempfea[0, 0]

    protein_index = 0
    for key, val in interaction_pair.items():
        protein, RNA = key[0], key[1]
        # pdb.set_trace()
        if RNA in RNA_seq_dict and protein in protein_seq_dict:  # and protein_fea_dict.has_key(protein) and RNA_fea_dict.has_key(RNA):
            label.append(val)
            RNA_seq = RNA_seq_dict[RNA]
            protein_seq = translate_sequence(protein_seq_dict[protein], group_dict)
            seqName = 'F' + protein
            seqName = seqName.replace("-", "_")
            # proteinFea[protein_index][0]= proteinFea[protein_index][0]/len(protein_seq)
            if deepmind:
                RNA_tri_fea = get_RNA_seq_concolutional_array(RNA_seq)
                protein_tri_fea = get_RNA_seq_concolutional_array(protein_seq)
                train.append((RNA_tri_fea, protein_tri_fea))
            else:
                # pdb.set_trace()
                RNA_tri_fea = get_4_nucleotide_composition(tris, RNA_seq, pythoncount=False)
                # protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq, pythoncount =False)
                protein_tri_fea = proteinFea[seqName][0]
                # RNA_tri3_fea = get_4_nucleotide_composition(tris3, RNA_seq, pythoncount =False)
                # RNA_fea = [RNA_fea_dict[RNA][ind] for ind in fea_imp]
                # tmp_fea = protein_fea_dict[protein] + tri_fea #+ RNA_fea_dict[RNA]
                if seperate:
                    tmp_fea = (protein_tri_fea, RNA_tri_fea)
                    # chem_tmp_fea = (protein_fea_dict[protein], RNA_fea_dict[RNA])
                else:
                    tmp_fea = protein_tri_fea + RNA_tri_fea
                    # chem_tmp_fea = protein_fea_dict[protein] + RNA_fea_dict[RNA]
                train.append(tmp_fea)
                protein_index = protein_index + 1
                # chem_fea.append(chem_tmp_fea)
        else:
            print (RNA, protein)

    return np.array(train), label


def prepare_RPI1807_feature(graph=False, deepmind=False, seperate=False, chem_fea=True, dataset=''):
    print('RPI-Pred data')
    # name_list = read_name_from_fasta('ncRNA-protein/RNA_seq.fa')
    seq_dict = read_fasta_file('ncRNA-protein/RPI1807_RNA_seq.fa')
    protein_seq_dict = read_fasta_file('ncRNA-protein/RPI1807_protein_seq.fa')
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    tris = get_4_trids()
    # pdb.set_trace()
    train = []
    label = []
    chem_fea = []
    tempStruct = scipy.io.loadmat(dataset)
    tempfea = tempStruct['fea']
    proteinFea = tempfea[0, 0]
    # pdb.set_trace()
    with open('ncRNA-protein/RPI1807_PositivePairs.csv', 'r') as fp:
        for line in fp:
            if 'Protein ID' in line:
                continue
            pro1, pro2 = line.rstrip().split('\t')
            # pro1 = pro1.upper()
            pro2 = pro2.upper()
            if pro2 in seq_dict and  pro1 in protein_seq_dict:  # and protein_fea_dict.has_key(pro1) and RNA_fea_dict.has_key(pro2):
                label.append(1)
                RNA_seq = seq_dict[pro2]
                protein_seq = translate_sequence(protein_seq_dict[pro1], group_dict)
                seqName = 'F' + pro1
                seqName = seqName.replace("-", "_")
                if deepmind:
                    RNA_tri_fea = get_RNA_seq_concolutional_array(RNA_seq)
                    protein_tri_fea = get_RNA_seq_concolutional_array(protein_seq)
                    train.append((RNA_tri_fea, protein_tri_fea))
                else:
                    RNA_tri_fea = get_4_nucleotide_composition(tris, RNA_seq, pythoncount=False)
                    protein_tri_fea = proteinFea[seqName][0]
                    # protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq, pythoncount =False)
                    # RNA_fea = [RNA_fea_dict[RNA][ind] for ind in fea_imp]
                    # tmp_fea = protein_fea_dict[protein] + tri_fea #+ RNA_fea_dict[RNA]
                    if seperate:
                        tmp_fea = (protein_tri_fea, RNA_tri_fea)
                        # chem_tmp_fea = (protein_fea_dict[pro1], RNA_fea_dict[pro2])
                    else:
                        tmp_fea = protein_tri_fea + RNA_tri_fea
                        # chem_tmp_fea = protein_fea_dict[pro1] + RNA_fea_dict[pro2]
                    train.append(tmp_fea)
                    # chem_fea.append(chem_tmp_fea)
            else:
                print (pro1, pro2)
    with open('ncRNA-protein/RPI1807_NegativePairs.csv', 'r') as fp:
        for line in fp:
            if 'Protein ID' in line:
                continue
            pro1, pro2 = line.rstrip().split('\t')
            # pro1 = pro1.upper()
            pro2 = pro2.upper()
            if pro2 in seq_dict and  pro1 in protein_seq_dict:  # and protein_fea_dict.has_key(pro1) and RNA_fea_dict.has_key(pro2):
                label.append(0)
                RNA_seq = seq_dict[pro2]
                protein_seq = translate_sequence(protein_seq_dict[pro1], group_dict)
                seqName = 'F' + pro1
                seqName = seqName.replace("-", "_")
                if deepmind:
                    RNA_tri_fea = get_RNA_seq_concolutional_array(RNA_seq)
                    protein_tri_fea = get_RNA_seq_concolutional_array(protein_seq)
                    train.append((RNA_tri_fea, protein_tri_fea))
                else:
                    RNA_tri_fea = get_4_nucleotide_composition(tris, RNA_seq, pythoncount=False)
                    protein_tri_fea = proteinFea[seqName][0]
                    # protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq, pythoncount =False)
                    # RNA_fea = [RNA_fea_dict[RNA][ind] for ind in fea_imp]
                    # tmp_fea = protein_fea_dict[protein] + tri_fea #+ RNA_fea_dict[RNA]
                    if seperate:
                        tmp_fea = (protein_tri_fea, RNA_tri_fea)
                        # chem_tmp_fea = (protein_fea_dict[pro1], RNA_fea_dict[pro2])
                    else:
                        tmp_fea = protein_tri_fea + RNA_tri_fea
                        # chem_tmp_fea = protein_fea_dict[pro1] + RNA_fea_dict[pro2]
                    train.append(tmp_fea)
                    # chem_fea.append(chem_tmp_fea)
            else:
                print(pro1, pro2)
    return np.array(train), label


def prepare_RPI2241_369_feature(rna_fasta_file, data_file, protein_fasta_file, extract_only_posi=False,
                                graph=False, deepmind=False, seperate=False, chem_fea=True, dataset=''):
    seq_dict = read_fasta_file(rna_fasta_file)
    protein_seq_dict = read_fasta_file(protein_fasta_file)
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    tris = get_4_trids()
    train = []
    label = []
    chem_fea = []

    # posi_set = set()
    # pro_set = set()
    tempStruct = scipy.io.loadmat(dataset)
    tempfea = tempStruct['fea']
    proteinFea = tempfea[0, 0]

    with open(data_file, 'r') as fp:
        for line in fp:
            if line[0] == '#':
                continue
            protein, RNA, tmplabel = line.rstrip('\r\n').split('\t')
            if RNA in seq_dict and  protein in protein_seq_dict:
                label.append(int(tmplabel))
                RNA_seq = seq_dict[RNA]
                protein_seq = translate_sequence(protein_seq_dict[protein], group_dict)
                seqName = 'a' + protein
                seqName = seqName.replace("-", "_")
                if deepmind:
                    RNA_tri_fea = get_RNA_seq_concolutional_array(RNA_seq)
                    protein_tri_fea = get_RNA_seq_concolutional_array(protein_seq)
                    train.append((RNA_tri_fea, protein_tri_fea))
                else:
                    RNA_tri_fea = get_4_nucleotide_composition(tris, RNA_seq, pythoncount=False)
                    # protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq, pythoncount =False)
                    protein_tri_fea = proteinFea[seqName][0]
                    if seperate:
                        tmp_fea = (protein_tri_fea, RNA_tri_fea)
                    else:
                        tmp_fea = protein_tri_fea + RNA_tri_fea
                    train.append(tmp_fea)
            else:
                print (RNA, protein)

    return np.array(train), label


def get_data_deepmind(dataset, deepmind=False, seperate=True, chem_fea=False, extract_only_posi=False,
                      indep_test=False):
    if dataset == 'RPI2241':
        X, labels = prepare_RPI2241_369_feature('ncRNA-protein/RPI2241_rna.fa', 'ncRNA-protein/RPI2241_all.txt',
                                                'ncRNA-protein/RPI2241_protein.fa', graph=False, deepmind=deepmind,
                                                seperate=seperate, chem_fea=chem_fea, dataset='2241.mat')
    elif dataset == 'RPI369':
        X, labels = prepare_RPI2241_369_feature('ncRNA-protein/RPI369_rna.fa', 'ncRNA-protein/RPI369_all.txt',
                                                'ncRNA-protein/RPI369_protein.fa', graph=False, deepmind=deepmind,
                                                seperate=seperate, chem_fea=chem_fea, dataset='369.mat')
    elif dataset == 'RPI488':
        X, labels = prepare_RPI488_feature(deepmind=deepmind, seperate=seperate, chem_fea=chem_fea)
    elif dataset == 'RPI1807':
        X, labels = prepare_RPI1807_feature(graph=False, deepmind=deepmind, seperate=seperate, chem_fea=chem_fea,
                                            dataset='1807.mat')
    return X, labels


def plot_roc_curve(labels, probality, legend_text, auc_tag=True):
    # fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality)  # probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text + ' (AUC=%6.3f) ' % roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text)


def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return acc, precision, sensitivity, specificity, MCC


def transfer_array_format(data):
    formated_matrix1 = []
    formated_matrix2 = []
    for val in data:
        formated_matrix1.append(val[0])
        formated_matrix2.append(val[1])
    return np.array(formated_matrix1), np.array(formated_matrix2)


def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label


def preprocess_data(X, scaler=None, stand=True):
    if not scaler:
        if stand:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X


def get_blend_data(j, clf, skf, X_test, X_dev, Y_dev, blend_train, blend_test):
    print ('Training classifier [%s]' % (j))
    blend_test_j = np.zeros((X_test.shape[0], len(
        skf)))  # Number of testing data x Number of folds , we will take the mean of the predictions later
    for i, (train_index, cv_index) in enumerate(skf):
        print ('Fold [%s]' % (i))

        # This is the training and validation set
        X_train = X_dev[train_index]
        Y_train = Y_dev[train_index]
        X_cv = X_dev[cv_index]
        Y_cv = Y_dev[cv_index]

        clf.fit(X_train, Y_train)

        # This output will be the basis for our blended classifier to train against,
        # which is also the output of our classifiers
        # blend_train[cv_index, j] = clf.predict(X_cv)
        # blend_test_j[:, i] = clf.predict(X_test)
        blend_train[cv_index, j] = clf.predict_proba(X_cv)[:, 1]
        blend_test_j[:, i] = clf.predict_proba(X_test)[:, 1]
    # Take the mean of the predictions of the cross validation set
    blend_test[:, j] = blend_test_j.mean(1)
    print('Y_dev.shape = %s' % (Y_dev.shape))


def RPI_SE(dataset):

    X, labels = get_data_deepmind(dataset, seperate=True)
    X_data1, X_data2 = transfer_array_format(X)
    print (X_data1.shape, X_data2.shape)
    X_data1 = preprocess_data(X_data1)
    X_data2 = preprocess_data(X_data2)
    # y,encoder = preprocess_labels(labels)
    X_data = np.concatenate((X_data1, X_data2), axis=1)
    y = np.array(labels, dtype=int)

    num_cross_val = 5  # 5-fold
    all_performance_xgb = []
    all_performance_lgbm = []
    all_performance_nb = []
    all_performance_knn = []
    all_performance_stack = []
    all_performance_lstm = []
    all_labels = []
    all_prob = {}
    num_classifier = 4
    all_prob[0] = []
    all_prob[1] = []
    all_prob[2] = []
    all_prob[3] = []
    all_prob[4] = []
    all_prob[5] = []
    all_average = []
    print(X_data.shape,X_data.shape[0],X_data.shape[1])

    for fold in range(num_cross_val):
        train = np.array([x for i, x in enumerate(X_data) if i % num_cross_val != fold])
        test = np.array([x for i, x in enumerate(X_data) if i % num_cross_val == fold])
        # train1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val != fold])
        # test1 = np.array([x for i, x in enumerate(X_data1) if i % num_cross_val == fold])
        # train2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val != fold])
        # test2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
        real_labels = []
        for val in test_label:
            if val == 1:
                real_labels.append(1)
            else:
                real_labels.append(0)

        train_label_new = []
        for val in train_label:
            if val == 1:
                train_label_new.append(1)
            else:
                train_label_new.append(0)

        blend_train = np.zeros((train.shape[0], num_classifier))  # Number of training data x Number of classifiers
        blend_test = np.zeros((test.shape[0], num_classifier))  # Number of testing data x Number of classifiers
        skf = list(StratifiedKFold(train_label_new, num_classifier))
        #train, test, prefilter_train_bef, prefilter_test_bef = autoencoder_two_subnetwork_fine_tuning(train1, train2, train_label, test1, test2, test_label)

        all_labels = all_labels + real_labels
        tmp_aver = [0] * len(real_labels)
        class_index = 0

        print ('SVM')
        svm = svm.SVC(kernel='rbf',probability=True)
        svm.fit(train, train_label)
        svm_proba = svm.predict_proba(test)[:, 1]
        all_prob[0] = all_prob[0] + [val for val in svm_proba]
        tmp_aver = [val1 + val2 / 4 for val1, val2 in zip(svm_proba, tmp_aver)]
        y_pred_lgbm = transfer_label_from_prob(svm_proba)
        # print proba
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred_lgbm, real_labels)
        print (acc, precision, sensitivity, specificity, MCC)
        all_performance_lgbm.append([acc, precision, sensitivity, specificity, MCC])
        get_blend_data(class_index, svm, skf, test, train, np.array(train_label_new), blend_train, blend_test)
        print ('---' * 50)

        print ('XGB')
        class_index = class_index + 1
        xgb1 = XGBClassifier(max_depth=6,booster='gblinear')#learning_rate=0.1,max_depth=6, booster='gbtree'
        xgb1.fit(train, train_label)
        xgb_proba = xgb1.predict_proba(test)[:, 1]
        all_prob[1] = all_prob[1] + [val for val in xgb_proba]
        tmp_aver = [val1 + val2 / 4 for val1, val2 in zip(xgb_proba, tmp_aver)]
        y_pred_xgb = transfer_label_from_prob(xgb_proba)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred_xgb, real_labels)
        print (acc, precision, sensitivity, specificity, MCC)
        all_performance_xgb.append([acc, precision, sensitivity, specificity, MCC])
        get_blend_data(class_index, xgb1, skf, test, train, np.array(train_label_new), blend_train, blend_test)
        print ('---' * 50)

        print ('ExtraTrees')
        class_index = class_index + 1
        etree = ExtraTreesClassifier()
        etree.fit(train, train_label)
        etree_proba = etree.predict_proba(test)[:, 1]
        all_prob[2] = all_prob[2] + [val for val in etree_proba]
        tmp_aver = [val1 + val2 / 4 for val1, val2 in zip(etree_proba, tmp_aver)]
        y_pred_knn = transfer_label_from_prob(etree_proba)
        # y_pred_stack = blcf.predict(test)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred_knn, real_labels)
        print (acc, precision, sensitivity, specificity, MCC)
        all_performance_knn.append([acc, precision, sensitivity, specificity, MCC])
        get_blend_data(class_index, etree, skf, test, train, np.array(train_label_new), blend_train, blend_test)
        print ('---' * 50)

        print ('AdaBoost')
        class_index = class_index + 1
        Ada = AdaBoostClassifier()
        Ada.fit(train, train_label)
        proba = Ada.predict_proba(test)[:, 1]
        all_prob[3] = all_prob[3] + [val for val in proba]
        tmp_aver = [val1 + val2 / 4 for val1, val2 in zip(proba, tmp_aver)]
        y_pred_gnb = transfer_label_from_prob(proba)
        # y_pred_stack = blcf.predict(test)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred_gnb, real_labels)
        print (acc, precision, sensitivity, specificity, MCC)
        all_performance_nb.append([acc, precision, sensitivity, specificity, MCC])
        get_blend_data(class_index, Ada, skf, test, train, np.array(train_label_new), blend_train, blend_test)
        print ('---' * 50)

        all_average = all_average + tmp_aver

        print ("Stacked Ensemble")
        #bclf = VotingClassifier(voting='soft',estimators=[('svm',gbm),('XGB',xgb1),('ET',knn),('adaB',gnb)],weights=[2,2,1,1])
        bclf = LogisticRegression()
        bclf.fit(blend_train, train_label_new)
        stack_proba = bclf.predict_proba(blend_test)[:, 1]
        all_prob[4] = all_prob[4] + [val for val in stack_proba]
        y_pred_stack = transfer_label_from_prob(stack_proba)
        # y_pred_stack = blcf.predict(test)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred_stack,
                                                                             real_labels)
        print (acc, precision, sensitivity, specificity, MCC)
        all_performance_stack.append([acc, precision, sensitivity, specificity, MCC])
        print ('---' * 50)
        '''
        print "Average Ensemble"
        y_pred_average = transfer_label_from_prob(tmp_aver)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred_average, real_labels)
        print acc, precision, sensitivity, specificity, MCC
        all_performance_aver.append([acc, precision, sensitivity, specificity, MCC])
        print '---' * 50
        '''
    print('mean performance of XGB')
    print(np.mean(np.array(all_performance_xgb), axis=0))
    print('---' * 50)
    print('mean performance of SVM')
    print(np.mean(np.array(all_performance_lgbm), axis=0))
    print('---' * 50)
    print('mean performance of AdaBoost')
    print(np, mean(np.array(all_performance_nb), axis=0))
    print('---' * 50)

    print('mean performance of ExtraTrees')
    print(np, mean(np.array(all_performance_knn), axis=0))
    print('---' * 50)
    # print('mean performance of lstm')
    # print(np, mean(np.array(all_performance_lstm), axis=0))
    # print('---' * 50)
    print('mean performance of Stacked ensembling')
    print(np.mean(np.array(all_performance_stack), axis=0))
    print('---' * 50)

    Figure = plt.figure()
    plot_roc_curve(all_labels, all_prob[0], 'Kernal-SVM')
    plot_roc_curve(all_labels, all_prob[1], 'XGBoost')
    plot_roc_curve(all_labels, all_prob[2], 'ExtraTrees')
    plot_roc_curve(all_labels, all_prob[3], 'AdaBoost')
    plot_roc_curve(all_labels, all_prob[4], 'Stacked ensembling')
    #plot_roc_curve(all_labels, all_prob[5], 'LSTM')
    plot_roc_curve(all_labels, all_average, 'Average ensembling')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    # plt.savefig(save_fig_dir + selected + '_' + class_type + '.png')
    plt.show()


parser = argparse.ArgumentParser(description="""RPI-SE""")

parser.add_argument('-dataset',
                    type=str, help='which dataset you want to do 5-fold cross-validation')

parser.add_argument('-r',
                    type=str, help='RNA fasta file to store RNAs')

parser.add_argument('-p',
                    type=str, help='protein fasta file to store proteins')
args = parser.parse_args()
dataset = args.dataset
if dataset is not None:
    RPI_SE(dataset)
else:
    RNA_file = args.r
    protein_file = args.p
    if RNA_file is None or protein_file is None:
        print('you must input RNA and protein fasta file')

'''
/**                                                                          
 *          .,:,,,                                        .::,,,::.          
 *        .::::,,;;,                                  .,;;:,,....:i:         
 *        :i,.::::,;i:.      ....,,:::::::::,....   .;i:,.  ......;i.        
 *        :;..:::;::::i;,,:::;:,,,,,,,,,,..,.,,:::iri:. .,:irsr:,.;i.        
 *        ;;..,::::;;;;ri,,,.                    ..,,:;s1s1ssrr;,.;r,        
 *        :;. ,::;ii;:,     . ...................     .;iirri;;;,,;i,        
 *        ,i. .;ri:.   ... ............................  .,,:;:,,,;i:        
 *        :s,.;r:... ....................................... .::;::s;        
 *        ,1r::. .............,,,.,,:,,........................,;iir;        
 *        ,s;...........     ..::.,;:,,.          ...............,;1s        
 *       :i,..,.              .,:,,::,.          .......... .......;1,       
 *      ir,....:rrssr;:,       ,,.,::.     .r5S9989398G95hr;. ....,.:s,      
 *     ;r,..,s9855513XHAG3i   .,,,,,,,.  ,S931,.,,.;s;s&BHHA8s.,..,..:r:     
 *    :r;..rGGh,  :SAG;;G@BS:.,,,,,,,,,.r83:      hHH1sXMBHHHM3..,,,,.ir.    
 *   ,si,.1GS,   sBMAAX&MBMB5,,,,,,:,,.:&8       3@HXHBMBHBBH#X,.,,,,,,rr    
 *   ;1:,,SH:   .A@&&B#&8H#BS,,,,,,,,,.,5XS,     3@MHABM&59M#As..,,,,:,is,   
 *  .rr,,,;9&1   hBHHBB&8AMGr,,,,,,,,,,,:h&&9s;   r9&BMHBHMB9:  . .,,,,;ri.  
 *  :1:....:5&XSi;r8BMBHHA9r:,......,,,,:ii19GG88899XHHH&GSr.      ...,:rs.  
 *  ;s.     .:sS8G8GG889hi.        ....,,:;:,.:irssrriii:,.        ...,,i1,  
 *  ;1,         ..,....,,isssi;,        .,,.                      ....,.i1,  
 *  ;h:               i9HHBMBBHAX9:         .                     ...,,,rs,  
 *  ,1i..            :A#MBBBBMHB##s                             ....,,,;si.  
 *  .r1,..        ,..;3BMBBBHBB#Bh.     ..                    ....,,,,,i1;   
 *   :h;..       .,..;,1XBMMMMBXs,.,, .. :: ,.               ....,,,,,,ss.   
 *    ih: ..    .;;;, ;;:s58A3i,..    ,. ,.:,,.             ...,,,,,:,s1,    
 *    .s1,....   .,;sh,  ,iSAXs;.    ,.  ,,.i85            ...,,,,,,:i1;     
 *     .rh: ...     rXG9XBBM#M#MHAX3hss13&&HHXr         .....,,,,,,,ih;      
 *      .s5: .....    i598X&&A&AAAAAA&XG851r:       ........,,,,:,,sh;       
 *      . ihr, ...  .         ..                    ........,,,,,;11:.       
 *         ,s1i. ...  ..,,,..,,,.,,.,,.,..       ........,,.,,.;s5i.         
 *          .:s1r,......................       ..............;shs,           
 *          . .:shr:.  ....                 ..............,ishs.             
 *              .,issr;,... ...........................,is1s;.               
 *                 .,is1si;:,....................,:;ir1sr;,                  
 *                    ..:isssssrrii;::::::;;iirsssssr;:..                    
 *                         .,::iiirsssssssssrri;;:.                      
 */
'''