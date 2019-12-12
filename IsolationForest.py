import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

'''
This is sample code for implementing Isolation Forest for outlier and anomaly 
detection. This code is meant for educational purposes only and can be freely 
used and shared for that purpose.
 - Kiret Dhindsa
'''

rng = np.random.RandomState(42)

# Generate train data - 100 2D points
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]

# Generate 20 regular test observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]

# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))


# Plot data and model output with increasing contaminatino proportion
y_pred_train = []
y_pred_test = []
y_pred_outliers = []

contamination_values = np.around(np.arange(0.,0.5,0.1), decimals=1)
for contamination_proportion in contamination_values:
    # fit the model
    model = IsolationForest(max_samples=100, random_state=rng, contamination=contamination_proportion)
    model.fit(X_train)
    
    # get predicted class labels
    y_pred_train.append(model.predict(X_train))
    y_pred_test.append(model.predict(X_test))
    y_pred_outliers.append(model.predict(X_outliers))
    
    
    # retrieve the decision function - outliers are where Z is <0
    plt.figure()
    xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # plot the data
    plt.title("Isolation Forest with contamination_proportion = {}".format(contamination_proportion))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                     s=20, edgecolor='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                     s=20, edgecolor='k')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                    s=20, edgecolor='k')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([b1, b2, c],
               ["training data",
                "regular test data", "test outliers"],
               loc="upper left")
    plt.show()
    

# Get "accuracies" by contamination_value
train_accuracy = [(y==1).mean() for y in y_pred_train]
test_accuracy = [(y==1).mean() for y in y_pred_test]
outlier_accuracy = [(y==-1).mean() for y in y_pred_outliers]

# get f1 scores
f1_train = [f1_score(np.ones(y.shape),y) for y in y_pred_train]
f1_test = [f1_score(np.ones(y.shape),y) for y in y_pred_test]
f1_outliers = [f1_score(-np.ones(y.shape),y) for y in y_pred_outliers]

# Plot the models' outputs
fig, ax = plt.subplots()
bar_width = 0.25
opacity = 0.7
index = np.arange(contamination_values.shape[0])

bars1 = ax.bar(index, train_accuracy, bar_width, alpha=opacity, color='b', label='Training Data')
bars2 = ax.bar(index+bar_width, test_accuracy, bar_width, alpha=opacity, color='g', label='Test (regular)')
bars3 = ax.bar(index+bar_width*2, outlier_accuracy, bar_width, alpha=opacity, color='r', label='Test (outlier)')

ax.set_xlabel('Contamination Values')
ax.set_ylabel('Proportion "Correct"')
ax.set_title('Performance by data type and contamination value')
ax.set_xticks(index + bar_width/2)
ax.set_xticklabels(contamination_values)
ax.legend(loc='lower right')
fig.tight_layout()
plt.show()

# Plot the f1 scores
fig, ax = plt.subplots()
bar_width = 0.25
opacity = 0.7
index = np.arange(contamination_values.shape[0])

bars1 = ax.bar(index, f1_train, bar_width, alpha=opacity, color='b', label='Training Data')
bars2 = ax.bar(index+bar_width, f1_test, bar_width, alpha=opacity, color='g', label='Test (regular)')
bars3 = ax.bar(index+bar_width*2, f1_outliers, bar_width, alpha=opacity, color='r', label='Test (outlier)')

ax.set_xlabel('Contamination Values')
ax.set_ylabel('Proportion "Correct"')
ax.set_title('F1 Score by data type and contamination value')
ax.set_xticks(index + bar_width/2)
ax.set_xticklabels(contamination_values)
ax.legend(loc='lower right')
fig.tight_layout()
plt.show()

#np.arange(contamination_values.shape[0])