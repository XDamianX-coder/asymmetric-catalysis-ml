#Libraries import
import pandas as pd
from mordred import Calculator, descriptors
import mordred
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score

def prepare_data(file):
    df = pd.read_excel(file)
    targets = list(df.columns)[13]

    try:
        mol_objs = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]
    except:
        mol_objs = [Chem.MolFromSmiles(smi) for smi in df['AI_generated_SMILES']]

    calculate_descriptors = True
    
    if calculate_descriptors:
        calc = Calculator(descriptors, ignore_3D=False)
        molecular_descriptors = calc.pandas(mol_objs)
        molecular_descriptors = molecular_descriptors.applymap(is_morder_missing)
        molecular_descriptors = molecular_descriptors[sorted(molecular_descriptors.columns)]
        
                # Calculate Count-Based Morgan Fingerprints
        fingerprints_counts = []
        
        for mol in mol_objs:
            if mol is not None:
                # Use AllChem to compute Morgan Fingerprint with counts
                fp = AllChem.GetMorganFingerprint(mol, radius=2, useCounts=True)
                count_dict = fp.GetNonzeroElements()  # Corrected method name
                count_array = np.zeros(2048)
                for key, val in count_dict.items():
                    count_array[key % 2048] = val  # Use modulo to fit into array size
                fingerprints_counts.append(count_array)
            else:
                fingerprints_counts.append(np.nan * np.zeros(2048))  # Handle missing molecules

        # Convert fingerprint counts list to DataFrame
        counts_df = pd.DataFrame(fingerprints_counts, columns=[f'C_FP_{i}' for i in range(2048)])
        
        molecular_descriptors = pd.concat([molecular_descriptors, counts_df], axis=1)

    print("Data size (rows, columns): " + str(molecular_descriptors.shape))
    
    simple_preprocessing = True
    if simple_preprocessing:
        molecular_descriptors_cleaned = molecular_descriptors.dropna(axis=1, how='any')
        print("Data size after first reduction (rows, columns): " + str(molecular_descriptors_cleaned.shape))
        
    remove_all_zeros = True  #(FALSE) in the case if molecular descriptor from the features used in final model are missing set to False, in other case keep True  
    if remove_all_zeros:
        molecular_descriptors_cleaned = molecular_descriptors_cleaned.loc[:, (molecular_descriptors_cleaned != 0).any(axis=0)]
        print("Data size after second reduction (rows, columns): " + str(molecular_descriptors_cleaned.shape))
    
    
    #molecular_descriptors_cleaned = molecular_descriptors_cleaned.loc[:, (molecular_descriptors_cleaned != 0).any(axis=0)]
    #print("Data size after second reduction (rows, columns): " + str(molecular_descriptors_cleaned.shape))
    
    #molecular_descriptors_cleaned.to_excel("../Data/molecular_descriptors_with_molecular_fingerprints_training_test.xlsx")
    try:
        molecular_descriptors_cleaned[targets] = df[targets]
    except:
        print('There is an issue with the target values...')
    
    return molecular_descriptors_cleaned

def is_morder_missing(x):
    return np.nan if isinstance(x, (mordred.error.Missing, mordred.error.Error)) else x

def correlation_dataframe(molecular_descriptors_cleaned, correlation_threshold, target_column_name, verbose = False):
    
    if verbose:
        correlation_table = pd.DataFrame(data=molecular_descriptors_cleaned.columns.to_list(), 
                                         columns=["molecular descriptor name"])
        print(correlation_table.head())
        correlation_to_target = []
        for mol_desc in correlation_table['molecular descriptor name']:
            x = np.corrcoef(np.array(molecular_descriptors_cleaned[mol_desc]), 
                            np.array(molecular_descriptors_cleaned[target_column_name]))
            x = x.tolist()[0][1]
            correlation_to_target.append(x)
        correlation_table['corr_value'] = correlation_to_target
        print(correlation_table.head())
        correlation_table['absolute correlation value'] = [abs(x) for x in correlation_table['corr_value']]
        print(correlation_table[:-1].head())
    
        mol_desc_best_corr = correlation_table[correlation_table['absolute correlation value'] > correlation_threshold]
    
        print(mol_desc_best_corr.head())
        table_with_descriptors_to_be_used = mol_desc_best_corr[:-1]
        print(table_with_descriptors_to_be_used.head())
    else:
        correlation_table = pd.DataFrame(data=molecular_descriptors_cleaned.columns.to_list(), 
                                         columns=["molecular descriptor name"])
        
        correlation_to_target = []
        for mol_desc in correlation_table['molecular descriptor name']:
            x = np.corrcoef(np.array(molecular_descriptors_cleaned[mol_desc]), 
                            np.array(molecular_descriptors_cleaned[target_column_name]))
            x = x.tolist()[0][1]
            correlation_to_target.append(x)
        correlation_table['corr_value'] = correlation_to_target
        
        correlation_table['absolute correlation value'] = [abs(x) for x in correlation_table['corr_value']]
        
    
        mol_desc_best_corr = correlation_table[correlation_table['absolute correlation value'] > correlation_threshold]
    
        
        table_with_descriptors_to_be_used = mol_desc_best_corr[:-1]
        
    return table_with_descriptors_to_be_used
    
    

def transform(target_val):
    transformed = []
    for element in target_val:
        transformed.append(-1.0 * np.log10(element/1000000000))
    return transformed

def inverse_transform(transformed_values):
    inversed = []
    for element in transformed_values:
        inversed.append(np.power(10,-element)*1000000000)
    return inversed
    
def data_standardization(dataframe, target_column_name):
    
    dataframe_ = dataframe.drop([target_column_name], axis=1)
    
    to_be_returned = (dataframe_ - dataframe_.mean()) / dataframe_.std()
    to_be_returned[target_column_name] = dataframe[target_column_name]
    
    return to_be_returned
    
def prepare_model(data, features, model_type, target_column_name, random_state = 15, n_estimators_ = 2, max_depth = 2, n_neighbors=2, kernel_ = 'linear', gamma_ = 'auto', c_=1, train_test_split_ = False, verbose = False):

    if verbose:
        if model_type == 'RandomForestClassifier':
            model = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators_)
            print("The model used is: RandomForestClassifier...")
        elif model_type == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
            print("The model used is: DecisionTreeClassifier...")
        elif model_type == 'KNeighborsClassifier':
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            print("The model used is: KNeighborsClassifier...")
        elif model_type == 'SVC':
            model = SVC(gamma=gamma_, kernel=kernel_, C=c_)
            print("The model used is: SVC...")
        elif model_type == 'LogisticRegression':
            model = LogisticRegression()
            print("The model used is: LogisticRegression...")
        else:
            model = LogisticRegression()
            print("The model used is: LogisticRegression...")
    else:
        if model_type == 'RandomForestClassifier':
            model = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators_)
        elif model_type == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
        elif model_type == 'KNeighborsClassifier':
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif model_type == 'SVC':
            model = SVC(gamma=gamma_, kernel=kernel_, C=c_)
        elif model_type == 'LogisticRegression':
            model = LogisticRegression()
        else:
            model = LogisticRegression()

    if train_test_split_:
        X_train, X_test, y_train, y_test = train_test_split(data[features['molecular descriptor name']], 
                                                    data[target_column_name], 
                                                    test_size=0.15, random_state=random_state)

        model.fit(X_train, y_train)
        
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        
        training_accuracy = accuracy_score(y_train, pred_train)
        test_accuracy = accuracy_score(y_test, pred_test)

        train_prec = precision_score(y_train, pred_train) #train_prec = precision_score(y_train, pred_train, average='macro')
        test_prec = precision_score(y_test, pred_test) #test_prec = precision_score(y_test, pred_test, average='macro')



        if verbose:
            print("Training Accuracy: ", training_accuracy)
            print("Test Accuracy: ", test_accuracy)
            print("Training Precision: ", train_prec)
            print("Test Precision: ", test_prec)
            print("Confusion Matrix:\n", confusion_matrix(y_test, pred_test))
            print("Classification Report:\n", classification_report(y_test, pred_test))

    return model, training_accuracy, test_accuracy, train_prec, test_prec


def prepare_data_and_create_model(molecular_descriptors_df, correlation_threshold, standardization, model_type, target_column_name, random_state = 15, n_estimators_ = 12, max_depth = 2, n_neighbors=2, kernel_ = 'linear', gamma_ = 'auto', c_=1, train_test_split_ = True, verbose = False):
    
    if standardization:
        if verbose:
            print("I am doing standardization...")
        data_to_be_prepared = molecular_descriptors_df
        stand = data_standardization(data_to_be_prepared, target_column_name)
        corr = correlation_dataframe(stand, correlation_threshold, target_column_name, verbose)
    else:
        if verbose:
            print("I am not doing standardization...")
        data_to_be_prepared = molecular_descriptors_df
        corr = correlation_dataframe(data_to_be_prepared, correlation_threshold, target_column_name, verbose)

    if train_test_split_:
        model, train_acc, test_acc, train_prec, test_prec = prepare_model(data_to_be_prepared, corr, model_type, target_column_name, random_state, n_estimators_, max_depth, n_neighbors, kernel_, gamma_, c_, train_test_split_, verbose)
    else:
        print("There is no hardcoded validation data...")
    
    return model, train_acc, test_acc, train_prec, test_prec, data_to_be_prepared, corr, target_column_name
