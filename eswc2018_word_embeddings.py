import word_embeddings as we
import settings
import os


def load(size, epochs, shrunk=True):
    dir_name = settings.SHRUNK_EMBEDDINGS if shrunk else settings.ESWC2018_EMBEDDINGS
    filename = 'embeddings_snap_s%d_e%d.txt' % (size, epochs)
    path = os.path.join(dir_name, filename)
    return we.load(path)
