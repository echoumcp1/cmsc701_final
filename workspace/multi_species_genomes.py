# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script
# contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script for the multi-species genomes dataset. This dataset contains the genomes
from 850 different species."""

from typing import List
import datasets
import pandas as pd
from Bio import SeqIO


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{o2016reference,
  title={Reference sequence (RefSeq) database at NCBI: current status, taxonomic expansion, and functional annotation},
  author={O'Leary, Nuala A and Wright, Mathew W and Brister, J Rodney and Ciufo, Stacy and Haddad, Diana and McVeigh, Rich and Rajput, Bhanu and Robbertse, Barbara and Smith-White, Brian and Ako-Adjei, Danso and others},
  journal={Nucleic acids research},
  volume={44},
  number={D1},
  pages={D733--D745},
  year={2016},
  publisher={Oxford University Press}
}
"""

# You can copy an official description
_DESCRIPTION = """\
Dataset made of diverse genomes available on NCBI and coming from ~850 different species. 
Test and validation are made of 50 species each. The rest of the genomes are used for training.
Default configuration "6kbp" yields chunks of 6.2kbp (100bp overlap on each side). Similarly,
the "12kbp"configuration yields chunks of 12.2kbp. The chunks of DNA are cleaned and processed so that
they can only contain the letters A, T, C, G and N.
"""

_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/"

_LICENSE = "https://www.ncbi.nlm.nih.gov/home/about/policies/"

_CHUNK_LENGTHS = [6000, 12000]


def filter_fn(char: str) -> str:
    """
    Transforms any letter different from a base nucleotide into an 'N'.
    """
    if char in {'A', 'T', 'C', 'G'}:
        return char
    else:
        return 'N'


def clean_sequence(seq: str) -> str:
    """
    Process a chunk of DNA to have all letters in upper and restricted to
    A, T, C, G and N.
    """
    seq = seq.upper()
    seq = map(filter_fn, seq)
    seq = ''.join(list(seq))
    return seq


class MultiSpeciesGenomesConfig(datasets.BuilderConfig):
    """BuilderConfig for The Human Reference Genome."""

    def __init__(self, *args, chunk_length: int, overlap: int = 100, **kwargs):
        """BuilderConfig for the multi species genomes.
        Args:
            chunk_length (:obj:`int`): Chunk length.
            overlap: (:obj:`int`): Overlap in base pairs for two consecutive chunks (defaults to 100).
            **kwargs: keyword arguments forwarded to super.
        """
        num_kbp = int(chunk_length/1000)
        super().__init__(
            *args,
            name=f'{num_kbp}kbp',
            **kwargs,
        )
        self.chunk_length = chunk_length
        self.overlap = overlap


class MultiSpeciesGenomes(datasets.GeneratorBasedBuilder):
    """Genomes from 850 species, filtered and split into chunks of consecutive
    nucleotides. 50 genomes are taken for test, 50 for validation and 800
    for training."""

    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIG_CLASS = MultiSpeciesGenomesConfig
    BUILDER_CONFIGS = [MultiSpeciesGenomesConfig(chunk_length=chunk_length) for chunk_length in _CHUNK_LENGTHS]
    DEFAULT_CONFIG_NAME = "6kbp"

    def _info(self):

        features = datasets.Features(
            {
                "sequence": datasets.Value("string"),
                "description": datasets.Value("string"),
                "start_pos": datasets.Value("int32"),
                "end_pos": datasets.Value("int32"),
                "fasta_url": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:

        urls_filepath = dl_manager.download_and_extract('urls.txt')
        with open(urls_filepath) as urls_file:
            urls = [line.rstrip() for line in urls_file]
        
        test_urls = urls[-50:]   # 50 genomes for test set
        validation_urls = urls[-100:-50]  # 50 genomes for validation set
        train_urls = urls[:-100]  # 800 genomes for training

        train_downloaded_files = dl_manager.download_and_extract(train_urls)
        test_downloaded_files = dl_manager.download_and_extract(test_urls)
        validation_downloaded_files = dl_manager.download_and_extract(validation_urls)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": train_downloaded_files, "chunk_length": self.config.chunk_length}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"files": validation_downloaded_files, "chunk_length": self.config.chunk_length}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"files": test_downloaded_files, "chunk_length": self.config.chunk_length}),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, files, chunk_length):
        key = 0
        for file in files:
            with open(file, 'rt') as f:
                fasta_sequences = SeqIO.parse(f, 'fasta')

                for record in fasta_sequences:

                    # parse descriptions in the fasta file
                    sequence, description = str(record.seq), record.description

                    # clean chromosome sequence
                    sequence = clean_sequence(sequence)
                    seq_length = len(sequence)

                    # split into chunks
                    num_chunks = (seq_length - 2 * self.config.overlap) // chunk_length

                    if num_chunks < 1:
                        continue

                    sequence = sequence[:(chunk_length * num_chunks + 2 * self.config.overlap)]
                    seq_length = len(sequence)

                    for i in range(num_chunks):
                        # get chunk
                        start_pos = i * chunk_length
                        end_pos = min(seq_length, (i+1) * chunk_length + 2 * self.config.overlap)
                        chunk_sequence = sequence[start_pos:end_pos]

                        # yield chunk
                        yield key, {
                            'sequence': chunk_sequence,
                            'description': description,
                            'start_pos': start_pos,
                            'end_pos': end_pos,
                            'fasta_url': file.split('::')[-1]
                        }
                        key += 1
