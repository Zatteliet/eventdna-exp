from itertools import chain

from sklearn.metrics import classification_report


def score_micro_average(golds, predictions):
    """Score predictions against golds.

    `golds` and `preds` are parallel lists of IOB sequences, one sequence per example.
    """

    # Flatten each list into a flat IOB sequence.
    golds = list(chain.from_iterable(golds))
    preds = list(chain.from_iterable(predictions))
    
    return classification_report(golds, preds, output_dict=True)
