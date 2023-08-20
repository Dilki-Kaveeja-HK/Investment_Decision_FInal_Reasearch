from flask import Flask, render_template, request



app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    import numpy as np
    from sklearn.model_selection import train_test_split
    import nlpBase as nlp

    # Splitting data into train and test data
    x_train, x_test, y_train, y_test = train_test_split(nlp.tf_idf_matrix, nlp.y_df, random_state=0)

    # Importing classifiers
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    # Create the classifiers
    decision_tree_clf = DecisionTreeClassifier()
    logistic_regression_clf = LogisticRegression()
    random_forest_clf = RandomForestClassifier(n_estimators=100)

    # Fit the classifiers
    decision_tree_clf.fit(x_train, y_train)
    logistic_regression_clf.fit(x_train, y_train)
    random_forest_clf.fit(x_train, y_train)

    # Test each classifier and obtain their accuracies
    dt_accuracy = decision_tree_clf.score(x_test, y_test)
    lr_accuracy = logistic_regression_clf.score(x_test, y_test)
    rf_accuracy = random_forest_clf.score(x_test, y_test)

 #   print("Decision Tree Accuracy:", dt_accuracy * 100)
#    print("Logistic Regression Accuracy:", lr_accuracy * 100)
  #  print("Random Forest Accuracy:", rf_accuracy * 100)

    # Input data for prediction
    input_text = [user_input]

    # Use the count vectorizer from nlpBase to transform the input text
    nlp.count_vectorizer.fit_transform(nlp.x_df)
    freq_term_matrix_input = nlp.count_vectorizer.transform(input_text)

    # Get predictions from each classifier
    dt_prediction = decision_tree_clf.predict(freq_term_matrix_input)
    lr_prediction = logistic_regression_clf.predict(freq_term_matrix_input)
    rf_prediction = random_forest_clf.predict(freq_term_matrix_input)

    # Convert the predictions to integers
    dt_prediction = dt_prediction.astype(int)
    lr_prediction = lr_prediction.astype(int)
    rf_prediction = rf_prediction.astype(int)

    # Combine the predictions using   voting
    predictions_combined = np.array([dt_prediction, lr_prediction, rf_prediction])
    final_prediction = np.rint(np.mean(predictions_combined, axis=0)).astype(int)

    if final_prediction[0] == 1:
        predicted_sentiment = "Can Invest On This Company"
    else:
        predicted_sentiment = "It’s better to not Invest On This Company"

    return render_template('index.html', user_input=user_input, predicted_sentiment=predicted_sentiment)

if __name__ == '__main__':
    app.run(debug=True)
