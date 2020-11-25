from datasets.datasets import create_dataset
import autokeras as ak

#run default ak classifier
X1,X2,y1,y2=create_dataset('eurosat_rgb')
clf = ak.ImageClassifier( overwrite=True,multi_label=True,max_trials=10)
clf.fit(X1, y1)
print(clf.evaluate(X2,y2))
