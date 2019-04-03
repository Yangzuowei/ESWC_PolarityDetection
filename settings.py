import os

BASE_DIR = ''
EMBEDDINGS_DIR = os.path.join(BASE_DIR, '../WordVectors')
ESWC2018_EMBEDDINGS = os.path.join(EMBEDDINGS_DIR, 'EmbeddingsESWC2018')
SHRUNK_EMBEDDINGS = os.path.join(ESWC2018_EMBEDDINGS, 'shrunk')
DATA_PATH = '../data/en/ESWC2018Challenge/dranziera_protocol/'
TEST_PATH = '../data/en/ESWC2018Challenge/test/task1_testset.xml'
GOOGLE_NEWS_EMBEDDINGS = os.path.join(EMBEDDINGS_DIR, 'GoogleNews-vectors-negative300.bin.gz')