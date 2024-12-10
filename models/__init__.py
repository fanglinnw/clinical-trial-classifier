from .bert_classifier import BERTClassifier
from .baseline_classifiers import BaselineClassifiers

# For backward compatibility
class BioBERTClassifier(BERTClassifier):
    def __init__(self, model_path: str = "./trained_models/biobert", max_length: int = 8000):
        super().__init__(model_type='biobert', model_path=model_path, max_length=max_length)

class ClinicalBERTClassifier(BERTClassifier):
    def __init__(self, model_path: str = "./trained_models/clinicalbert", max_length: int = 8000):
        super().__init__(model_type='clinicalbert', model_path=model_path, max_length=max_length)

class PubMedBERTClassifier(BERTClassifier):
    def __init__(self, model_path: str = "./trained_models/pubmedbert", max_length: int = 8000):
        super().__init__(model_type='pubmedbert', model_path=model_path, max_length=max_length)

__all__ = ['BERTClassifier', 'BioBERTClassifier', 'ClinicalBERTClassifier', 
           'PubMedBERTClassifier', 'BaselineClassifiers']