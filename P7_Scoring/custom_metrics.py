from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, fbeta_score  # for classification
from sklearn.metrics import confusion_matrix


def eval_classif_metrics(actual, pred, pred_proba, threshold=None):
    '''
    Calcule les métriques de classification composé de :
        auc,
        accuracy,
        precision,
        recall,
        business_score : F_score(beta=10)

    Input :
        actual,
        pred_proba : predict probability
        threshold : threshold to determine predict class based on predict probability
                    If filled, 'pred' is not used but replaced by np.where('pred_proba' >= threshold, 1, 0)
    '''
    if pred_proba is None: pred_proba = pred
    if threshold: pred = np.where(pred_proba >= threshold, 1, 0)

    auc = round(roc_auc_score(actual, pred_proba) ,2)
    accuracy = round(accuracy_score(actual, pred) ,2)
    precision = round(precision_score(actual, pred) ,2)
    recall = round(recall_score(actual, pred) ,2)

    return auc, accuracy, precision, recall


def eval_business_metrics(actual, pred, pred_proba=None, threshold=None):
    '''
    Calcule les métriques business de classification composé de :
        Business_Stat : stat = F_score(beta=10)
        Business_Cost : cost = 1*FP + 20*FN
        Business_Income : income = 1*TN - 1*FP - 20*FN

    Input :
        actual,
        pred_proba : predict probability
        threshold : threshold to determine predict class based on predict probability
                    If filled, 'pred' is not used but replaced by np.where('pred_proba' >= threshold, 1, 0)
    '''
    if pred_proba is None: pred_proba = pred
    if threshold: pred = np.where(pred_proba >= threshold, 1, 0)

    tn, fp, fn, tp = confusion_matrix(actual, pred).ravel()  # /len(actual)
    business_stat = round(fbeta_score(actual, pred, beta=10), 2)
    business_cost = norm_business_metrics(tn, fp, fn, tp,
                                          tn_value=0, fp_value=-1, fn_value=-20, tp_value=0)
    business_income = norm_business_metrics(tn, fp, fn, tp,
                                            tn_value=1, fp_value=-1, fn_value=-20, tp_value=0)

    return business_stat, business_cost, business_income, (
    tn / len(actual), fp / len(actual), fn / len(actual), tp / len(actual))


def norm_business_metrics(tn, fp, fn, tp,
                          tn_value, fp_value, fn_value, tp_value):
    J = tp * tp_value + tn * tn_value + fp * fp_value + fn * fn_value
    max_J = (fp + tn) * tn_value + (fn + tp) * tp_value
    min_J = (fp + tn) * fp_value + (fn + tp) * fn_value

    # Gain normalisé entre 0 et 1
    J_normalized = (J - min_J) / (max_J - min_J)
    return round(J_normalized, 2)

# Créer les fonctions de score business
from sklearn.metrics import make_scorer

business_stat_scorer = make_scorer(fbeta_score, beta=10)

def business_cost(actual, pred, pred_proba=None, threshold=None) :
    return eval_business_metrics(actual, pred, pred_proba=pred_proba, threshold=threshold)[1]

business_cost_scorer = make_scorer(business_cost, greater_is_better=True)

def business_income(actual, pred, pred_proba=None, threshold=None) :
    return eval_business_metrics(actual, pred, pred_proba, threshold=threshold)[2]

business_income_scorer = make_scorer(business_income, greater_is_better=True)