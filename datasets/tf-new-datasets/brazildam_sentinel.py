from pathlib import Path
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.data import Iterator
import tifffile as tiff


class brazildam(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for brazildam dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Go to https://drive.google.com/drive/folders/1v1F4faAD8zCm_vocGxILiIUncaRz1pZB and download 'sentinel_compressed.7z' to get the data. Place the
  extracted file in the `manual_dir/` (~tensorflow_datasets/downloads/manual)
  """

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
            'image': tfds.features.Tensor(shape=(384, 384, 13), dtype=tf.float32),
            'label': tfds.features.ClassLabel(names=['barragem', 'nao_barragem']),
        }),
      supervised_keys=('image', 'label'),
      homepage='http://www.patreo.dcc.ufmg.br/2020/01/27/brazildam-dataset/',
      # Bibtex citation for the dataset
      citation=r"""
      @inproceedings{
  ferreira2020brazildam,
  title={BrazilDAM: A Benchmark dataset for Tailings Dam Detection},
  author={Ferreira, E., Brito, M., Alvim, M. S., Balaniuk, R., dos Santos, J.},
  booktitle={Latin American GRSS & ISPRS Remote Sensing Conference 2020, Santigo, Chile},
  year={2020},
  organization={IEEE}
}
      """,
    )

  def _generate_examples(self, path): #-> Iterator[Tuple[Key, Example]]:
      """Generator of examples for each split."""
      for sub in path.iterdir():
          new_dir=path / sub
          for img_path in new_dir.glob('*.tif'):
              # Yields (key, example)
              if (tiff.imread(img_path)).shape==(384,384,13):
                  yield img_path.name, {'image': tiff.imread(img_path),'label': sub.name,}

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    # data_path is a pathlib-like `Path('<manual_dir>/data.zip')`
    archive_path = dl_manager.manual_dir / 'sentinel_compressed'
    # Extract the manually downloaded `data.zip`
    extracted_path = dl_manager.extract(archive_path)
    print( extracted_path)
    # dl_manager returns pathlib-like objects with `path.read_text()`,
    # `path.iterdir()`,...
    return {
        'train_2016': self._generate_examples(path=extracted_path / '2016'),
        'train_2017': self._generate_examples(path=extracted_path / '2017'),
        'train_2018': self._generate_examples(path=extracted_path / '2018'),
        'train_2019': self._generate_examples(path=extracted_path / '2019'),
    }

