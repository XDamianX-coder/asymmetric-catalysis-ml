
import sys
from pathlib import Path
prediction_mode_path = Path("../module")
sys.path.append(prediction_mode_path.as_posix())
import models_creation_cla as pred_model

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from matplotlib import pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

target = 'epox_cla'  # Update this to your actual classification target if different

def prepare_data(file_path):
    molecular_descriptors_df = pred_model.prepare_data(file_path)
    return molecular_descriptors_df

def select_features(mol_descriptors_target):
    x = []
    for i in range(100):
        a = pred_model.correlation_dataframe(mol_descriptors_target, i*0.01, target).shape[0]
        x.append(a)

    df = pd.DataFrame(data=x, columns=['Number of features'])
    df['Correlation threshold ('+target+')'] = [i*0.01 for i in range(100)]
    
    filtered_df = df[df['Number of features'] != 0]

    first_zero_index = df[df['Number of features'] == 0].index.min()

    if first_zero_index is not None:
        filtered_df = df.iloc[:first_zero_index]

    x_values = filtered_df['Correlation threshold ('+target+')']
    y_values = filtered_df['Number of features']

    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, color='blue')
    plt.xlabel('Correlation threshold ('+target+')')
    plt.ylabel('Number of features')
    plt.title('Scatter Plot of Number of Features vs Correlation Threshold')
    plt.grid(True)
    plt.savefig('../Data/Scatter Plot of Number of Features vs Correlation Threshold ('+target+')_.pdf', bbox_inches='tight')

    maximal_number_of_features = round(mol_descriptors_target.shape[0] / 2, 0)
    print("Maximal number of features: " + str(maximal_number_of_features))

    filtered_df_ = filtered_df[filtered_df['Number of features'] < maximal_number_of_features]

    correlation_low = filtered_df_['Correlation threshold ('+target+')'].min()
    correlation_high = filtered_df_['Correlation threshold ('+target+')'].max()

    return correlation_low, correlation_high

def do_qspr_mlr(data, correlation_low, correlation_high, random_state, target):
    step = 0.01
    initial_step = correlation_low
    last_step = correlation_high + 0.01
    first_list = [x / 100.0 for x in range(int(initial_step * 100), int(last_step * 100), int(step * 100))]
    second_list = []
    third_list = []
    f_list = []
    ta_list = []
    aq_list = []
    for i in first_list: # fix them all !!!
        without_standardization, train_accuracy, test_accuracy, train_prec, test_prec, _, h_, target_column_name = pred_model.prepare_data_and_create_model(
            molecular_descriptors_df=data,
            correlation_threshold=i,
            standardization=False,
            model_type='LogisticRegression',
            target_column_name=target,
            random_state=random_state,
            train_test_split_=True,
            verbose=False
        )
        second_list.append(train_accuracy)
        third_list.append(test_accuracy)
        ta_list.append(train_prec)
        aq_list.append(test_prec)
        f_list.append(len(h_))

    df_without_standardization = pd.DataFrame(data=first_list, columns=["Correlation threshold"])
    df_without_standardization['Training Accuracy'] = second_list
    df_without_standardization['Test Accuracy'] = third_list
    df_without_standardization['Training Precision'] = ta_list
    df_without_standardization['Test Precision'] = aq_list
    df_without_standardization['Number of features'] = f_list

    df_linear = df_without_standardization.copy()
    return df_linear

def do_qspr_dt(data, correlation_low, correlation_high, random_state, target):
    step = 0.01
    initial_step = correlation_low
    last_step = correlation_high + 0.01
    first_list = [x / 100.0 for x in range(int(initial_step * 100), int(last_step * 100), int(step * 100))]
    max_depth = [range(2, 30, 1)]
    corr_th = []
    second_list = []
    third_list = []
    f_list = []
    fif_list = []
    ta_list = []
    aq_list = []
    for i in first_list:
        for depth in max_depth[0]:
            without_standardization, train_accuracy, test_accuracy, train_prec, test_prec, _, h_, target_column_name = pred_model.prepare_data_and_create_model(
                molecular_descriptors_df=data,
                correlation_threshold=i,
                standardization=False,
                model_type='DecisionTreeClassifier',
                max_depth=depth,
                target_column_name=target,
                random_state=random_state,
                train_test_split_=True,
                verbose=False
            )
            corr_th.append(i)
            second_list.append(train_accuracy)
            third_list.append(test_accuracy)
            ta_list.append(train_prec)
            aq_list.append(test_prec)
            f_list.append(len(h_))
            fif_list.append(depth)

    df_without_standardization = pd.DataFrame(data=corr_th, columns=["Correlation threshold"])
    df_without_standardization['Training Accuracy'] = second_list
    df_without_standardization['Test Accuracy'] = third_list
    df_without_standardization['Training Precision'] = ta_list
    df_without_standardization['Test Precision'] = aq_list
    df_without_standardization['Number of features'] = f_list
    df_without_standardization['Depth number'] = fif_list

    df_dt = df_without_standardization.copy()
    return df_dt

def do_qspr_rf(data, correlation_low, correlation_high, random_state, target):
    step = 0.01
    initial_step = correlation_low
    last_step = correlation_high + 0.01
    first_list = [x / 100.0 for x in range(int(initial_step * 100), int(last_step * 100), int(step * 100))]
    n_estimators = [range(2, 21, 1)]
    corr_th = []
    second_list = []
    third_list = []
    f_list = []
    fif_list = []
    ta_list = []
    aq_list = []
    for i in first_list:
        for estimator in n_estimators[0]:
            without_standardization, train_accuracy, test_accuracy, train_prec, test_prec, _, h_, target_column_name = pred_model.prepare_data_and_create_model(
                molecular_descriptors_df=data,
                correlation_threshold=i,
                standardization=False,
                model_type='RandomForestClassifier',
                n_estimators_=estimator,
                target_column_name=target,
                random_state=random_state,
                train_test_split_=True,
                verbose=False
            )
            corr_th.append(i)
            second_list.append(train_accuracy)
            third_list.append(test_accuracy)
            ta_list.append(train_prec)
            aq_list.append(test_prec)
            f_list.append(len(h_))
            fif_list.append(estimator)

    df_without_standardization = pd.DataFrame(data=corr_th, columns=["Correlation threshold"])
    df_without_standardization['Training Accuracy'] = second_list
    df_without_standardization['Test Accuracy'] = third_list
    df_without_standardization['Training Precision'] = ta_list
    df_without_standardization['Test Precision'] = aq_list
    df_without_standardization['Number of features'] = f_list
    df_without_standardization['Number of estimators'] = fif_list

    df_random_forest = df_without_standardization.copy()
    return df_random_forest

def do_qspr_knn(data, correlation_low, correlation_high, random_state, target):
    step = 0.01
    initial_step = correlation_low
    last_step = correlation_high + 0.01
    first_list = [x / 100.0 for x in range(int(initial_step * 100), int(last_step * 100), int(step * 100))]
    n_neighbor_ = [range(3,10,1)] #[range(1,10,1)]
    corr_th = []
    second_list = []
    third_list = []
    f_list = []
    ta_list = []
    aq_list = []
    fif_list = []
    for i in first_list:
        for neigbor in n_neighbor_[0]:
            without_standardization, train_accuracy, test_accuracy, train_prec, test_prec, _, h_, target_column_name = pred_model.prepare_data_and_create_model(
                molecular_descriptors_df=data,
                correlation_threshold=i,
                standardization=False,
                model_type='KNeighborsClassifier',
                target_column_name=target,
                random_state=random_state,
                n_neighbors=neigbor,
                train_test_split_=True,
                verbose=False
            )
            corr_th.append(i)
            second_list.append(train_accuracy)
            third_list.append(test_accuracy)
            ta_list.append(train_prec)
            aq_list.append(test_prec)
            f_list.append(len(h_))
            fif_list.append(neigbor)

    df_without_standardization = pd.DataFrame(data=corr_th, columns=["Correlation threshold"])
    df_without_standardization['Training Accuracy'] = second_list
    df_without_standardization['Test Accuracy'] = third_list
    df_without_standardization['Training Precision'] = ta_list
    df_without_standardization['Test Precision'] = aq_list
    df_without_standardization['Number of features'] = f_list
    df_without_standardization['Number of neighbors'] = fif_list

    df_k_nearest = df_without_standardization.copy()
    return df_k_nearest

def do_qspr_svm(data, correlation_low, correlation_high, random_state, target):
    step = 0.01
    initial_step = correlation_low
    last_step = correlation_high + 0.01
    first_list = [x / 100.0 for x in range(int(initial_step * 100), int(last_step * 100), int(step * 100))]
    _c_ = [0.1, 1, 10]
    _gamma_ = [1, 0.1, 0.01, 'auto']
    _kernel_ = ['linear', 'rbf', 'sigmoid']
    corr_th = []
    second_list = []
    third_list = []
    f_list = []
    ta_list = []
    aq_list = []
    f_list = []
    _c_list = []
    _gamma_list = []
    _kernel_list = []
    for i in first_list:
        for c in _c_:
            for gamma in _gamma_:
                for kernel in _kernel_:
                    without_standardization, train_accuracy, test_accuracy, train_prec, test_prec, _, h_, target_column_name = pred_model.prepare_data_and_create_model(
                        molecular_descriptors_df=data,
                        correlation_threshold=i,
                        standardization=False,
                        model_type='SVC',
                        kernel_=kernel,#'linear',
                        gamma_=gamma,#'auto',
                        c_ = c,
                        target_column_name=target,
                        random_state=random_state,
                        train_test_split_=True,
                        verbose=False
                    )
                    corr_th.append(i)
                    second_list.append(train_accuracy)
                    third_list.append(test_accuracy)
                    ta_list.append(train_prec)
                    aq_list.append(test_prec)
                    f_list.append(len(h_))
                    _c_list.append(c)
                    _gamma_list.append(gamma)
                    _kernel_list.append(kernel)

    df_without_standardization = pd.DataFrame(data=corr_th, columns=["Correlation threshold"])
    df_without_standardization['Training Accuracy'] = second_list
    df_without_standardization['Test Accuracy'] = third_list
    df_without_standardization['Training Precision'] = ta_list
    df_without_standardization['Test Precision'] = aq_list
    df_without_standardization['C'] = _c_list
    df_without_standardization['Gamma'] = _gamma_list
    df_without_standardization['Kernel'] = _kernel_list
    df_without_standardization['Number of features'] = f_list

    df_support_vector_machine = df_without_standardization.copy()
    return df_support_vector_machine



def prepare_excel(df_linear, df_decision_tree, df_random_forest, df_k_nearest, df_svm, random_state):
    with pd.ExcelWriter('../Data/Quality_'+target+'_'+str(random_state)+'_cla_.xlsx') as writer:  
        df_linear.to_excel(writer, sheet_name='MLR')
        df_decision_tree.to_excel(writer, sheet_name='DT')
        df_random_forest.to_excel(writer, sheet_name='RF')
        df_k_nearest.to_excel(writer, sheet_name='KNN')
        df_svm.to_excel(writer, sheet_name='SVM')



if __name__ == "__main__":

    data = prepare_data('../Data/QSPR_epoxidation_classifier.xlsx')

    corr_low, corr_high = select_features(data)

    random_states = [15, 28, 42]

    for random_state in random_states:

        df_linear = do_qspr_mlr(data, corr_low, corr_high, random_state, target)

        df_dt = do_qspr_dt(data, corr_low, corr_high, random_state, target)

        df_rf = do_qspr_rf(data, corr_low, corr_high, random_state, target)

        df_knn = do_qspr_knn(data, corr_low, corr_high, random_state, target)

        df_svm = do_qspr_svm(data, corr_low, corr_high, random_state, target)

        prepare_excel(df_linear, df_dt, df_rf, df_knn, df_svm, random_state)