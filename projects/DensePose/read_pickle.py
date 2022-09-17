import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path to pickle file")
args = parser.parse_args()


f = open(args.path, 'rb')
data = pickle.load(f)


#densepose0 = data[0]['pred_densepose'][0]
#labels = densepose0.labels # integers
#uvs = densepose0.uv
#nonzero_label_indices = labels.nonzero(as_tuple=True)
#print(labels)

