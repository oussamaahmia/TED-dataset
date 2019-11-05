import json
import gzip
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

version = "v1"
language = "EN" #ES LV ET MT FI BG NL CS FR PL DA GA PT DE HR RO EL HU SK IT SL EN LT SV
labels_language = "EN"
hierarchy = 2
cpv_labels = json.load(open('fd-TED/cpv.json'))[labels_language]


def cpv_hierarchy(cpv, hierarchy=2):
    return [i[:hierarchy] + "0"*(8 - hierarchy) for i in cpv]


def get_documents(path):
    data = []
    for i, doc in enumerate(gzip.open(path)):
        l = []
        doc_dict = json.loads(doc.decode())
        l.append(doc_dict['objet'])
        cpv = doc_dict['cpv']
        if 'additional_information' in doc_dict:
            l.extend(doc_dict['additional_information'])
        if 'lots' in doc_dict:
            if doc_dict['lots']:
                for lot in doc_dict['lots']:
                    if 'cpv' in lot:
                        cpv.extend(lot['cpv'])
                    l.append(lot['subject'])
                    if 'desc' in lot:
                        l.extend(lot['desc'])
        yield ("\n".join(l), cpv_hierarchy(cpv))


txt, y_ = zip(*get_documents('fd-TED/filtered/ted-%s-%s.jsons.gz' % (language, version)))

count = CountVectorizer().fit(txt)
binarizer = MultiLabelBinarizer().fit(y_)
X = count.transform(txt)
y = binarizer.transform(y_)

sgd = OneVsRestClassifier(SGDClassifier())
sgd.fit(X, y)

pred = sgd.predict(X)
print(classification_report(y, pred, target_names=[
      cpv_labels[i] for i in binarizer.classes_]))
