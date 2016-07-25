# ASSIGNMENT 7
# Your Name

import numpy as np
import csv
import sklearn.cluster as skcluster
from sklearn.decomposition.pca import PCA
from elasticsearch import Elasticsearch
from time import time
import itertools
import cPickle as pickle
import glob
import requests
import cv2
import os
import errno
from multiprocessing import Pool

# Import ORB as SIFT to avoid confusion.
try:
    from cv2 import ORB as SIFT
except ImportError:
    try:
        from cv2 import SIFT

        cv2.ocl.setUseOpenCL(False)
    except ImportError:
        try:
            SIFT = cv2.ORB_create
        except:
            raise AttributeError("Version of OpenCV(%s) does not have SIFT / ORB."
                                 % cv2.__version__)


def mkdir_p(path):
    try:
        os.makedirs(path, mode=0o744)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def fetch_artwork(count=1000, min_height=500, min_width=500):
    headers = {'Ocp-Apim-Subscription-Key': os.environ.get('AZURE_COGNITIVE_KEY')}
    images = {}
    current = 0
    while len(images) < count:
        req = 'https://api.cognitive.microsoft.com/bing/v5.0/images/search?q=famous artwork&count={}&offset={}'. \
            format(150, current)
        res = requests.get(req, headers=headers).json()

        if 'value' not in res or len(res['value']) == 0:
            print("res didn't contain value. Perhaps we are out of images with {} added.".format(len(images)))
            print(res)
            return images.values()

        for image in res['value']:
            if image['height'] > min_height and image['width'] > min_width and image['imageId'] not in images:
                images[image['imageId']] = image

        print("Image Count: {}, Current offset: {}, Number returned: {}".format(len(images), current, len(res['value'])))
        current += len(res['value'])

    return images.values()[:count]


def store_artwork(artworks=None, path='images/artwork', processes=20):
    if artworks is None:
        artworks = []
    mkdir_p(path)
    pool = Pool(processes=processes)
    results = pool.map(write_art_image, artworks)
    pool.close()
    pool.join()
    for result in results:
        print(result)
    return results


def write_art_image(art, chunk_size=512 * 1024, base_path='images/artwork'):
    image_id = art['imageId'].encode('utf-8').strip()
    image_format = art['encodingFormat'].encode('utf-8').strip()
    image_name = art['name'].encode('utf-8').strip()
    file_name = os.path.join(base_path, "{}.{}".format(image_id, image_format))

    if os.path.exists(file_name):
        return "File already exists. Skipping."
    else:
        print("Writing: '{}' via {} to {}".format(image_name, art['contentUrl'], file_name))

        try:
            res = requests.get(art['contentUrl'], stream=True)
            with open(file_name, 'wb') as fd:
                for chunk in res.iter_content(chunk_size):
                    fd.write(chunk)
            return "Wrote: '{}' via {} to {}".format(image_name, art['contentUrl'], file_name)
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            return "Failed because of: {}".format(e.message)


def surf_data(artwork_path='images/source'):
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_files = []
    for extension in extensions:
        image_files.extend(glob.glob(os.path.join(artwork_path, extension)))

    surf = cv2.xfeatures2d.SURF_create(500)
    for path in image_files:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = surf.detectAndCompute(img, None)
        yield (os.path.basename(os.path.splitext(path)[0]), keypoints, descriptors)


def store_surf_data(pickle_data, surf_path='images/surf'):
    mkdir_p(surf_path)
    pickle.dump(pickle_data, open(os.path.join(surf_path, "artwork_surf.p"), "wb"))


def store_descriptors_csv(descriptors, surf_path='images/surf'):
    with open(os.path.join(surf_path, 'surf.csv'), 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for descriptor in descriptors:
            writer.writerow(list(descriptor))


def load_surf_data(surf_path='images/surf'):
    file_path = os.path.join(surf_path, "artwork_surf.p")
    if os.path.exists(file_path):
        return pickle.load(open(file_path, "rb"))
    else:
        return []


def build_kmeans_model(descriptors, num_clusters=10000):
    print("starting clustering")
    kmeans = skcluster.KMeans(num_clusters, max_iter=10000)
    start = time()
    idxs = kmeans.fit_predict(descriptors)
    print("done in %0.3fs" % (time() - start))
    return idxs, kmeans


def store_surf_model(model, model_path="images/surf/model.sav"):
    pickle.dump(model, open(model_path, 'wb'))


def load_surf_model(model_path="images/surf/model.sav"):
    return pickle.load(open(model_path, 'rb'))


def pickle_keypoints(file_id, keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp_array.append(
            (point.pt, point.size, point.angle, point.response, point.octave, point.class_id, descriptors[i], file_id))
        i += 1
    return temp_array


def unpickle_keypoints(array):
    file_ids = []
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                                    _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        file_ids = point[7]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return file_ids, keypoints, np.array(descriptors)


def index_source_image(es, file_id, terms):
    # by default we connect to localhost:9200
    es.indices.create(index='comp-photo-idx', ignore=400)
    es.index(index="comp-photo-idx", doc_type="artwork", id=file_id, body={"tags": list(set(terms))})


def query_source_images(es, terms):
    return es.search(index="comp-photo-idx", doc_type="artwork",
                     body={"query": {"terms": {"tags": list(set(terms))}}})


# fetch famous art images from bing
# bing_art = fetch_artwork(count=440)

# write those images to file system
# store_artwork(bing_art)

# store_surf_data([pickle_keypoints(file_id, kps, descs) for file_id, kps, descs in surf_data()])

# surf_slice = itertools.islice(surf_data(), 30)
data = list(surf_data())
descriptors = list(itertools.chain.from_iterable([descs for file_id, kps, descs in data]))

# store_descriptors_csv(descriptors)
#
# descriptors = np.array(descriptors)

# print "starting PCA from 64 down to 32"
# decomp = PCA(32)
# reduced = decomp.fit_transform(descriptors)

# idxs, model = build_kmeans_model(descriptors)
# store_surf_model(model)

model = load_surf_model()

print("Loaded model and initializing ElasticSearch")

# Had to bump up the ElasticSearch max clause count to the following. index.query.bool.max_clause_count: 4096
es = Elasticsearch()
for file_id, _, descs in data:
    index_source_image(es, file_id, [int(i) for i in model.predict(descs)])

print("ElasticSearch initialized with source image data and model terms.")

print("Querying for images based on terms sets.")

sample_surf_data = surf_data(artwork_path="images/test")
for file_id, _, descs in sample_surf_data:
    outer = set([int(i) for i in model.predict(descs)])
    for f, _, d in data:
        inner = set([int(i) for i in model.predict(d)])
        print("File: {} has {} intersecting integers; outer: {}, inner {}".format(f, len(
            set(outer).intersection(inner)), len(outer), len(inner)))
    print("Hits for file: {}".format(file_id))
    for hit in query_source_images(es, [int(i) for i in model.predict(descs)])["hits"]["hits"]:
        print("Score: {}, File: {}".format(hit["_score"], hit["_id"]))

    print("")
