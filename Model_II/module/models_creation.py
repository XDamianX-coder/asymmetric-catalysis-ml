## Load libraries
#Libraries import
import pandas as pd
from mordred import Calculator, descriptors
import mordred
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import linear_model


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import math


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

    molecular_descriptors_cleaned = molecular_descriptors #remove...

    simple_preprocessing = True #False
    if simple_preprocessing:
        molecular_descriptors_cleaned = molecular_descriptors.dropna(axis=1, how='any')
        print("Data size after first reduction (rows, columns): " + str(molecular_descriptors_cleaned.shape))
        
    #molecular_descriptors_cleaned = molecular_descriptors_cleaned.loc[:, (molecular_descriptors_cleaned != 0).any(axis=0)]
    print("Data size after second reduction (rows, columns): " + str(molecular_descriptors_cleaned.shape))
    
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
    

def prepare_model(data, features, model_type, target_column_name, random_state = 15, n_estimators_ = 2, max_depth = 2, n_neighbors=2,  kernel_ = 'linear', gamma_ = 'auto', c_=1, train_test_split_ = False, verbose = False):

    if verbose:
        if model_type == 'RandomForestRegressor':
            model = RandomForestRegressor(random_state=15, n_estimators=n_estimators_)
            print("The model used is: RandomForest...")
        elif model_type == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor(random_state=15, max_depth=max_depth)
            print("The model used is: DecisionTree...")
        elif model_type == 'KNeighborsRegressor':
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
            print("The model used is: KNeighbors...")
        elif model_type == 'SVR':
            model = SVR(gamma = gamma_, kernel=kernel_, C=c_)
            print("The model used is: SVR...")
        elif model_type == 'linear_model':
            model = linear_model.LinearRegression()
            print("The model used is: LinearReg...")
        else:
            model = linear_model.LinearRegression()
            print("The model used is: Linear...")
    else:
        if model_type == 'RandomForestRegressor':
            model = RandomForestRegressor(random_state=15, n_estimators=n_estimators_)
        elif model_type == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor(random_state=15, max_depth=max_depth)
        elif model_type == 'KNeighborsRegressor':
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
        elif model_type == 'SVR':
            model = SVR(gamma=gamma_, kernel=kernel_, C=c_)
        elif model_type == 'linear_model':
            model = linear_model.LinearRegression()
        else:
            model = linear_model.LinearRegression()

    if train_test_split_:
        X_train, X_test, y_train, y_test = train_test_split(data[features['molecular descriptor name']], 
                                                    data[target_column_name], 
                                                    test_size=0.15, random_state=random_state) #test_size=0.2
            
        model.fit(X_train, y_train)
            
        pred_train = model.predict(X_train)
        sqrt_r2_train = np.sqrt(r2_score(y_train, pred_train))
        training_data_r2 = r2_score(y_train, pred_train)
        pred_test = model.predict(X_test)
        sqrt_r2_test = np.sqrt(r2_score(y_test, pred_test))
        test_data_r2 = r2_score(y_test, pred_test)
        training_data_RMSE = math.sqrt(mean_squared_error(y_train, pred_train))
        test_data_RMSE = math.sqrt(mean_squared_error(y_test, pred_test))



        if verbose:
            try:
                print("Return the R^2 of the test set: ")
                print(model.score(X_test, y_test))
            except:
                pass
            print("Training R^2 score: "+ str(training_data_r2))
            print('Training R: '+ str(sqrt_r2_train))
            print('Training Root Mean Square Error: '+str(training_data_RMSE))
            print("Test data - unseen during training:")
            print("Test R^2 score: "+ str(test_data_r2))
            print('Test R: '+ str(sqrt_r2_test))
            print('Test Root Mean Square Error: '+str(test_data_RMSE))
            print("The test data: "+str(y_test))
            print("The predicted values: "+str(pred_test))

    return model, training_data_r2, test_data_r2, training_data_RMSE, test_data_RMSE

def data_standardization(dataframe, target_column_name):
    
    dataframe_ = dataframe.drop([target_column_name], axis=1)
    
    to_be_returned = (dataframe_ - dataframe_.mean()) / dataframe_.std()
    to_be_returned[target_column_name] = dataframe[target_column_name]
    
    return to_be_returned


def prepare_data_and_create_model(molecular_descriptors_df, correlation_threshold, standardization, model_type, target_column_name, random_state = 15, n_estimators_ = 12, max_depth = 2, n_neighbors=2, kernel_ = 'linear', gamma_ = 'auto', c_=1, train_test_split_ = True, verbose = False):
    
    if standardization == True:
        
        if verbose:
            print("I am doing standardization...")
        else:
            pass
        
        data_to_be_prepared = molecular_descriptors_df
        
        stand = data_standardization(data_to_be_prepared, target_column_name)
        
        corr = correlation_dataframe(stand, correlation_threshold, target_column_name, verbose)
        
        if train_test_split_:
            model, train_r2, test_r2, training_data_RMSE, test_data_RMSE = prepare_model(data_to_be_prepared, corr, model_type, target_column_name, random_state, n_estimators_, max_depth, n_neighbors, kernel_, gamma_, c_, train_test_split_, verbose)
        else:
            print("There is no hardcoded validation data...")
        
    elif standardization == False:
        
        if verbose:
            print("I am not doing standardization...")
        else:
            pass
        
        data_to_be_prepared = molecular_descriptors_df
        
        corr = correlation_dataframe(data_to_be_prepared, correlation_threshold, target_column_name, verbose)
        
        if train_test_split_:
            model, train_r2, test_r2, training_data_RMSE, test_data_RMSE = prepare_model(data_to_be_prepared, corr, model_type, target_column_name, random_state, n_estimators_, max_depth, n_neighbors, kernel_, gamma_, c_, train_test_split_, verbose)
        else:
            print("There is no hardcoded validation data...")
    else:
        print("Error...")
    
    return model, train_r2, test_r2, data_to_be_prepared, corr, target_column_name, training_data_RMSE, test_data_RMSE