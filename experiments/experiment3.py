#This code takes the dataset and the number of trials as argument to execute the satellite image classification task
#It saves the evaluation results and the best model. The model summary is printed out.
from datasets import create_dataset
import autokeras as ak
import numpy as np
import sys
import tensorflow as tf
dataname=sys.argv[1]
trial=sys.argv[2]

#run default ak classifier
X1,X2,y1,y2=create_dataset(dataname)
print("runing ",dataname)
#metrics
METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.Accuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),

]
clf = ak.SatelliteImageClassifier( overwrite=True, project_name=dataname+"_experiment3_1",directory='/data/experiments/experiment3/'+dataname+trial,max_trials=trial,metrics=METRICS)
clf.fit(X1,y1)
resultsm=np.asarray([clf.evaluate(X2,y2)])
print(resultsm)
with open("/data/experiments/metric_results3.csv", "a") as f:
    f.write("\n")
    f.write(trial+dataname)
    np.savetxt(f,resultsm,delimiter=",")
best_model=clf.export_model()
print(best_model.summary)
best_model.save("data/models/experiment3_1/"+dataname+trial)

