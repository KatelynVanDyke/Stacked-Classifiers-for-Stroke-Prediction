import pandas as pd
from imblearn import over_sampling, under_sampling
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics, tree, ensemble, neighbors, linear_model


def combined_sample(data, labels):
    over = over_sampling.RandomOverSampler(sampling_strategy=0.25)
    new_data, new_labels = over.fit_resample(data, labels)
    under = under_sampling.RandomUnderSampler(sampling_strategy=0.5)
    new_data, new_labels = under.fit_resample(new_data, new_labels)
    return new_data, new_labels

def print_prediction_quality(truth, predict):
    accuracy = metrics.accuracy_score(truth, predict)
    precision = metrics.precision_score(truth, predict)
    recall = metrics.recall_score(truth, predict)
    f1 = metrics.f1_score(truth, predict)
    print("accuracy " + str("%.3f" % accuracy))
    print("precision " + str("%.3f" % precision))
    print("recall " + str("%.3f" % recall))
    print("f1 " + str("%.3f" % f1) + "\n")


if __name__ == '__main__':
    # read features
    original_data = pd.read_csv('original_data/data.csv', usecols=range(0, 5))

    # read stroke outcomes
    original_labels = pd.read_csv('original_data/data.csv', usecols=[5])

    # normalize data
    preprocessing.normalize(original_data)

    # perform random over and under sampling
    sample_data, sample_labels = combined_sample(original_data, original_labels)

    # export sample data to csv
    sample_all = pd.concat([sample_data, sample_labels], axis=1)
    sample_all.to_csv('sample_data/data.csv', float_format='%.3f', header=True, index=False,
                      columns=["gender", "age", "hypertension", "heart_disease", "avg_glucose_level", "stroke"])

    # separate stroke and non stroke sample data
    sample_stroke = sample_all[sample_all["stroke"] == 1]
    sample_stroke = sample_stroke.drop(columns=["stroke"])
    sample_non_stroke = sample_all[sample_all["stroke"] == -1]
    sample_non_stroke = sample_non_stroke.drop(columns=["stroke"])

    # calculate stroke and non stroke means
    sample_stroke_means = sample_stroke.astype('float32').mean(axis=0)
    sample_non_stroke_means = sample_non_stroke.astype('float32').mean(axis=0)

    # export stroke and non stroke means
    sample_all_means = pd.concat([sample_stroke_means, sample_non_stroke_means], axis=1)
    sample_all_means = sample_all_means.transpose()
    sample_all_means[''] = [1, -1]
    cols = [sample_all_means.columns[-1]] + sample_all_means.columns[:-1].tolist()
    sample_all_means = sample_all_means[cols]
    sample_all_means.to_csv('sample_data/mean.csv', float_format='%.3f', header=True, index=False,
                            columns=["", "gender", "age", "hypertension", "heart_disease", "avg_glucose_level"])

    # calculate stroke and non stroke standard deviations
    sample_stroke_stds = sample_stroke.astype('float32').std(axis=0)
    sample_non_stroke_stds = sample_non_stroke.astype('float32').std(axis=0)

    # export stroke and non stroke standard deviations
    sample_all_std_devs = pd.concat([sample_stroke_stds, sample_non_stroke_stds], axis=1)
    sample_all_std_devs = sample_all_std_devs.transpose()
    sample_all_std_devs[''] = [1, -1]
    cols = [sample_all_std_devs.columns[-1]] + sample_all_std_devs.columns[:-1].tolist()
    sample_all_std_devs = sample_all_std_devs[cols]
    sample_all_std_devs.to_csv('sample_data/std_dev.csv', float_format='%.3f', header=True, index=False,
                               columns=["", "gender", "age", "hypertension", "heart_disease", "avg_glucose_level"])

    # calculate correlation coefficients
    sample_corrs = sample_all.astype('float32').corr(method='pearson')

    # export correlation coefficients
    old_sample_corrs = original_data.astype('float32').corr(method='pearson')
    old_sample_corrs.to_csv('original_data/r_values.csv', float_format='%.3f')
    sample_corrs.to_csv('sample_data/r_values.csv', float_format='%.3f')

    # separate training and testing data
    train_all, test_all = train_test_split(sample_all, stratify=sample_all["stroke"], train_size=0.8)
    train_data = train_all.iloc[:, 0:5]
    train_labels = train_all.iloc[:, 5]
    test_data = test_all.iloc[:, 0:5]
    test_labels = test_all.iloc[:, 5]

    print("*** DECISION TREE ***\n")

    # train decision tree
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree = decision_tree.fit(train_data, train_labels)
    decision_tree_train = decision_tree.predict(train_data)
    print_prediction_quality(train_labels, decision_tree_train)

    # test decision tree
    decision_tree_test = decision_tree.predict(test_data)
    print_prediction_quality(test_labels, decision_tree_test)

    print("*** RANDOM FOREST ***\n")

    # train random forest
    random_forest = ensemble.RandomForestClassifier()
    random_forest = random_forest.fit(train_data, train_labels)
    random_forest_train = random_forest.predict(train_data)
    print_prediction_quality(train_labels, random_forest_train)

    # test decision tree
    random_forest_test = random_forest.predict(test_data)
    print_prediction_quality(test_labels, random_forest_test)

    print("*** K NEIGHBORS ***\n")

    # train k neighbors
    k_neighbors = neighbors.KNeighborsClassifier(n_neighbors=1)
    k_neighbors = k_neighbors.fit(train_data, train_labels)
    k_neighbors_train = k_neighbors.predict(train_data)
    print_prediction_quality(train_labels, k_neighbors_train)

    # test k neighbors
    k_neighbors_test = k_neighbors.predict(test_data)
    print_prediction_quality(test_labels, k_neighbors_test)

    print("*** STACKED ***\n")

    # combine classifier data
    stacked_data_train = {'tree': decision_tree_train, 'forest': random_forest_train, 'neighbors': k_neighbors_train}
    stacked_train = pd.DataFrame(data=stacked_data_train)
    stacked_data_test = {'tree': decision_tree_test, 'forest': random_forest_test, 'neighbors': k_neighbors_test}
    stacked_test = pd.DataFrame(data=stacked_data_test)

    # train stacked
    logistic = linear_model.LogisticRegression()
    logistic = logistic.fit(stacked_train, train_labels)
    logistic_train = logistic.predict(stacked_train)
    print_prediction_quality(train_labels, logistic_train)

    # test stacked
    logistic_test = logistic.predict(stacked_test)
    print_prediction_quality(test_labels, logistic_test)