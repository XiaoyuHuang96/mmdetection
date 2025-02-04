from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class VOCDataset(XMLDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        self.year=2012
#        if 'VOC2007' in self.img_prefix:
#            self.year = 2007
#        elif 'VOC2012' in self.img_prefix:
#            self.year = 2012
#        else:
#            raise ValueError('Cannot infer dataset year from img_prefix')
