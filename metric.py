# coding=utf-8
import random
import numpy as np
from pyNTCIREVAL import Labeler
from pyNTCIREVAL.metrics import nERR, QMeasure, MSnDCG, nDCG

def get_query_label(label_file_path):
    label_dict = {}
    f = open(label_file_path, 'r')
    f.readline()
    while True:
        line = f.readline()
        if not line:
            break
        (qid, doc_id, relevance) = line.split('\t')
        qid = int(qid)
        if label_dict.get(qid) is None:
            label_dict[qid] = {}
        relevance = int(relevance)
        label_dict[qid][doc_id] = relevance
    f.close()
    return label_dict

def data_process(y_pred, y_true):
    qrels = {}
    ranked_list = []
    c = list(zip(y_pred, y_true))
    random.shuffle(c)
    c = sorted(c, key=lambda x:x[0], reverse=True)
    for i in range(len(c)):
        qrels[i] = c[i][1]
        ranked_list.append(i)
    grades = range(1, label_range+1)

    labeler = Labeler(qrels)
    labeled_ranked_list = labeler.label(ranked_list)
    rel_level_num = len(grades)
    xrelnum = labeler.compute_per_level_doc_num(rel_level_num)
    return xrelnum, grades, labeled_ranked_list

def n_dcg(y_pred, y_true, k):
    xrelnum, grades, labeled_ranked_list = data_process(y_pred, y_true)
    metric = nDCG(xrelnum, grades, cutoff=k, logb=2)
    result = metric.compute(labeled_ranked_list)
    return result


def q_measure(y_pred, y_true, k, beta=1.0):
    xrelnum, grades, labeled_ranked_list = data_process(y_pred, y_true)
    metric = QMeasure(xrelnum, grades, beta, cutoff=k)
    result = metric.compute(labeled_ranked_list)
    return result


def n_err(y_pred, y_true, k):
    xrelnum, grades, labeled_ranked_list = data_process(y_pred, y_true)
    metric = nERR(xrelnum, grades, cutoff=k)
    result = metric.compute(labeled_ranked_list)
    return result

def compute_metrics(result_file_path, query_label, cutoff):
    with open(result_file_path, 'r') as f:
        lines = f.readlines()
    score_dict = {}
    for line in lines:
        (qid, _, doc_id, _, predict_score, _) = line.split()
        qid = int(qid)
        predict_score = float(predict_score)
        if score_dict.get(qid) is None:
            score_dict[qid] = {}
        score_dict[qid][doc_id] = predict_score
    print('Metrics with cutoff@%d:' % cutoff)
    n_dcg_scores = []
    q_measure_scores = []
    n_err_scores = []
    for qid, q2s in score_dict.items():
        q2r = query_label[qid]
        y_pred = []
        y_true = []
        for doc_id, score in q2s.items():
            y_pred.append(score)
            if q2r.get(doc_id) is None:
                y_true.append(0)
            else:
                y_true.append(q2r[doc_id])
        if max(y_true) == 0:
            continue
        n_dcg_score = n_dcg(y_pred, y_true, k=cutoff)
        n_dcg_scores.append(n_dcg_score)
        q_measure_score = q_measure(y_pred, y_true, k=cutoff)
        q_measure_scores.append(q_measure_score)
        n_err_score = n_err(y_pred, y_true, k=cutoff)
        n_err_scores.append(n_err_score)
        print('qid: %d, n_dcg: %f, q_measure: %f, n_err: %f' % (qid, n_dcg_score, q_measure_score, n_err_score))
    print('===================')
    print('avg ndcg: %f, avg q-measure: %f, avg n-err: %f' % (np.mean(n_dcg_scores), np.mean(q_measure_scores), np.mean(n_err_scores)))

if __name__ == '__main__':
    label_range = 4  # 4级相关性标注（0，1，2，3）
    query_label = get_query_label('ntcir14_test_label.txt')
    for cutoff in [5, 10, 20]:
        compute_metrics('result.txt', query_label, cutoff)
