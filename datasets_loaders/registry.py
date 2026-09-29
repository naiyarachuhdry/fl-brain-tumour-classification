from .brain_tumor_dataset import BrainTumorLoader
from .titanic_dataset import TitanicLoader
#from .kaggle_mri_dataset import KaggleMRILoader


DATASET_REGISTRY = {
    "brain_tumor": BrainTumorLoader,
    "titanic": TitanicLoader,
#    "kaggle_mri": KaggleMRILoader,
}