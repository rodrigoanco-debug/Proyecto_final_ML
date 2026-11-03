"""Métricas de clasificación con un umbral explícito y sin ajustar modelos."""
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)


def calcular_metricas(y, probabilidades, umbral=0.5):
    pred = (probabilidades >= umbral).astype(int)
    return {'accuracy': accuracy_score(y, pred),
            'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0),
            'f1': f1_score(y, pred, zero_division=0),
            'roc_auc': roc_auc_score(y, probabilidades),
            'average_precision': average_precision_score(y, probabilidades)}
