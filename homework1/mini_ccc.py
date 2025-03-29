"""Custom Dataset Builder for Common Crawl."""
import datasets
from homework import html_to_text, clean_text, heuristic_quality_filter
from utils import read_warc_file
from typing import List, Iterator, Tuple, Dict, Any

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = "cmu large language model homework1 dataset"
_DATA_URL = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2018-17/segments/1524125937193.1/warc/CC-MAIN-20180420081400-20180420101400-00000.warc.gz" 
 
class MiniCleanedCommonCrawl(datasets.GeneratorBasedBuilder):
    def _info(self) -> datasets.DatasetInfo:
        """
        Should return a DatasetInfo object describing <string> type values for a url and it's corresponding text.
        """
        return datasets.DatasetInfo(
          description = _DESCRIPTION,
          features=datasets.Features(
                {
                    "url": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            ),
          supervised_keys=None,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """
        Should return a SplitGenerator object which downloads your data and creates train and validation splits.
        """
        download_file = dl_manager.download_and_extract(_DATA_URL)
        return [
          datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={"file_path": download_file}
          ),
        ]
    
    def _generate_examples(self, file_path) -> Iterator[Tuple[Any, Dict[str, str]]]:
        """
        Streams raw data from the downloaded file and yields tuples consisting of a unique ID and the url/cleaned text.
        Should call the functions you defined in homework.py. 
        """
        example_id = 0
        for url, html in read_warc_file(file_path):
          try:
            text = html_to_text(html)
            if heuristic_quality_filter(text):
              yield example_id, {'url': url, 'text': clean_text(text)}
              example_id += 1
          except Exception as e:
            logger.warning(f'Error processing URL {url} : {e}')
 
if __name__ == "__main__":   
    # Note: Calling load_dataset caches the processed dataset locally.
    # The default cache directory is ~/.cache/huggingface/datasets.
    # To force the dataset to be recreated, you should pass in the
    # additional argument download_mode=datasets.DownloadMode.REUSE_CACHE_IF_EXISTS
    dataset = datasets.load_dataset(
        "mini_ccc.py",
        "MiniCleanedCommonCrawl",
        trust_remote_code=True,
        split=datasets.Split.TRAIN)
    
    # Iterate over the first 100 examples.
    for ex in dataset.take(100):
        print(ex['url'])
        print(ex['text'])