print(m)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# clf = SVC(kernel='rbf')
grid_values = {'gamma': [0.01, 0.1, 1, 10],
               'C': [0.01, 0.1, 1, 10]}

# default metric to optimize over grid parameters: recall
grid_clf_rec = GridSearchCV(m, param_grid = grid_values, scoring = 'recall')
grid_clf_rec.fit(X_train, y_train)
y_decision_fn_scores_rec = grid_clf_rec.decision_function(X_test) 

svm = SVC(kernel='rbf', C=0.01, gamma=0.01).fit(X_train, y_train)
svm_predicted = svm.predict(X_test)

print('Grid best parameter (max. recall): ', grid_clf_rec.best_params_)
print('Grid best score (recall): ', grid_clf_rec.best_score_)
print('Precision: {:.2f}'.format(precision_score(y_test, svm_predicted)))
print(precision_score(y_test, svm_predicted))
