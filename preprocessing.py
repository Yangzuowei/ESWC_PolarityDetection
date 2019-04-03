import eswc2018_dataset_handler as data_handler
import word_embeddings as we
import os


def shrink_all_embeddings(embeddings_path, data_path):
    data = data_handler.read(data_path)

    print('Building vocabulary...')
    vocabulary = data_handler.vocabulary(data)

    for filename in os.listdir(embeddings_path):
        path = os.path.join(embeddings_path, filename)
        we.shrink_to_vocabulary(path, vocabulary)


def polarity_to_int(polarity):
    return {
        'positive': 0,
        'negative': 1
    }[polarity]


def int_to_polarity(value):
    return {
        0: 'positive',
        1: 'negative'
    }[value]


def extract_text(record):
    summary = str(record['summary']).lower()
    text = str(record['text']).lower()
    return summary + ' ' + text + ' ' + summary
