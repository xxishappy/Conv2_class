import numpy as np
from sklearn.metrics import accuracy_score,cohen_kappa_score,confusion_matrix,f1_score,matthews_corrcoef,precision_score,recall_score,roc_auc_score

def macro_specificity(y_ture,y_pred,num_classes):
    '''
    TN/(TN+FP)
    '''
    labels=list(range(num_classes))
    matrix=confusion_matrix(y_ture,y_pred,labels=labels)
    total=matrix.sum()
    per_class=[]

    for class_idx in range(num_classes):
        TN=(total-matrix[class_idx,:].sum()-matrix[:,class_idx].sum()+matrix[class_idx,class_idx])
        FP=matrix[:,class_idx].sum()-matrix[class_idx,class_idx]
        denominator=TN+FP
        per_class.append(TN/denominator)
    return float(np.mean(per_class)*100.0)

def cal_metrics(y_true, y_pred, num_classes, avg_inference_ms=None, y_prob=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))

    metrics = {
        "ACC": float(accuracy_score(y_true, y_pred) )*100,
        "F1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ) *100,
        "Precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ) *100,
        "Sensitivity": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) )*100,
        "Specificity": macro_specificity(y_true, y_pred, num_classes) ,
        "MCC": float(matthews_corrcoef(y_true, y_pred)) *100,
        "Kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)) *100,
        "AUC":float(roc_auc_score(y_true,np.asarray(y_prob),labels=labels,multi_class="ovr",average="macro")) *100,
        "Time":float(avg_inference_ms)
    }
    return metrics
