from __future__ import print_function
import os
import xmltodict
import nlp
import settings


def read(data_path=settings.DATA_PATH):
    file_names = []

    for root, _, filenames in os.walk(data_path):
        for name in filenames:
            if name.endswith('.xml'):
                full_name = os.path.join(root, name)
                file_names.append(full_name)
    data = []
    for name in file_names:
        data += read_xml(name)

    print(len(data), 'records read successfully!!')

    return data


def read_xml(name):
    with open(name) as fd:
        print('Reading', name)
        doc = xmltodict.parse(fd.read())
        return doc['Sentences']['sentence']


def vocabulary(data):
    result = set()
    for record in data:
        text = str(record['text']) + ' ' + str(record['summary'])
        tokens = nlp.tokenize(text)
        for token in tokens:
            result.add(token.lower())
    return result
