from .voc import VOCDataset
from .registry import DATASETS

@DATASETS.register_module
class person_dataset(VOCDataset):
    CLASSES=("person",)


