from pathlib import Path
import tensorflow_datasets as tfds
import tensorflow as tf
import tifffile as tiff
import csv

class savana_scenes(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Brazilian Cerrado-Savanna Scenes Dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return tfds.core.DatasetInfo(
        builder=self,	
	# Description and homepage used for documentation
      description="""
      Brazildam dataset consists of multispectral images of ore tailings dams throughout Brazil
      """,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Tensor(shape=(64, 64, 3), dtype=tf.uint8),
            'label': tfds.features.ClassLabel(names=['FOR', 'AGR','HRB','SHR']),
        }),
      supervised_keys=('image', 'label'),
      homepage='http://www.patreo.dcc.ufmg.br/2017/11/12/brazilian-cerrado-savanna-scenes-dataset/',
      # Bibtex citation for the dataset
      citation=r"""
	@inproceedings{nogueira2016towards,
	title={Towards vegetation species discrimination by using data-driven descriptors},
	author={Nogueira, Keiller and Dos Santos, Jefersson A and Fornazari, Tamires and Silva, Thiago Sanna Freire and Morellato, Leonor Patricia and Torres, Ricardo da S},
	booktitle={2016 9th IAPR Workshop on Pattern Recogniton in Remote Sensing (PRRS)},
	pages={1--6},
	year={2016},
	organization={Ieee}
	}
  
      """,
    )

  def _generate_examples(self, path):
      """Generator of examples for each split."""
      with open(str(path),mode='r') as reader:
          rd=csv.reader(reader,delimiter='.')
          for r in rd:
              x_name=r[1]+".tif"
              yield x_name, {'image': tiff.imread(Path(path).parents[1] / "images/" / x_name),'label': r[0],}

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    extracted_path = dl_manager.download_and_extract("https://homepages.dcc.ufmg.br/~keiller.nogueira/datasets/brazilian_cerrado_dataset.zip")
#   extracted_path =Path('/home/ami-m-017/Documents/MsComputerScience/research')
    # dl_manager returns pathlib-like objects with `path.read_text()`,
    # `path.iterdir()`,...
    return {
        'fold1': self._generate_examples(path=extracted_path / 'Brazilian_Cerrado_Savana_Scenes_Dataset/folds/fold1'),
        'fold2': self._generate_examples(path=extracted_path / 'Brazilian_Cerrado_Savana_Scenes_Dataset/folds/fold2'),
        'fold3': self._generate_examples(path=extracted_path / 'Brazilian_Cerrado_Savana_Scenes_Dataset/folds/fold3'),
        'fold4': self._generate_examples(path=extracted_path / 'Brazilian_Cerrado_Savana_Scenes_Dataset/folds/fold4'),
        'fold5': self._generate_examples(path=extracted_path / 'Brazilian_Cerrado_Savana_Scenes_Dataset/folds/fold5'),
    }

