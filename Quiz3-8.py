print(m)

y_scores_lr = m.predict(X_test)

print('Macro-averaged precision = {:.2f} (treat classes equally)'
      .format(precision_score(y_test, y_scores_lr, average = 'macro')))

precision_score(y_test, y_scores_lr, average = 'macro')
