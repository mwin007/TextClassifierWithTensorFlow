from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np

# Data sets
SPAMBASE_DATA = "uci_spambase.csv"

# Load datasets.
spambase_dataset = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=SPAMBASE_DATA,
    target_dtype=np.int,
    features_dtype=np.float32)

# Split dataset to train (.7) and test (.3) sets (randomly)
X_train, X_test, y_train, y_test = train_test_split(spambase_dataset.data,
                                                    spambase_dataset.target,
                                                    test_size=0.3)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=57)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/spambase_model")
# print (X_train[0])
print (len(y_train))
# print (X_test[0])
print (len(y_test))

# Fit model.
classifier.fit(x=X_train,
               y=y_train,
               steps=2000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=X_test,
                                     y=y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))


# Classify a new email
new_samples = np.array(
    [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.83,4.83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.302,0,1.7,5,17]
     ,[0,0,0,0,0,0,0,0,0,0,0,1.53,0,0,0,0,0,0,4.61,0,0,0,0,0,0,0,0,0,1.53,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.222,5,20]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))



