'''
1. Run data_collection.py to initiate the file download
2. Parse dictionary into dataframe and save as a csv file
3. Move downloaded images into train, validation and test 
3. plot basic eda graphs
'''

import data_collection as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


metadata_lst = data.img_downloader()


def dataframe_converter(lst):
    """
    input:  list or json file downloaded from data.img_downloader
    output: csv file saved in a data folder
    """
    import pandas as pd
    df = pd.DataFrame(lst)

    # Convert "NA" in to NaN
    df.replace({'NA':None},inplace=True)

    # The scraped metadata contained photos linked with the emperors' nicknames
    # The following script will consolidate duplicates under a single name.
    duplicate_emperors = {'Augustus':['Divus Augustus',
                                      'Augustus Divus',
                                      'Octavian'],
                          'Valerian II': ['Valerian II Divus'],
                          'Constantine I': ['Constantine I Divus'],
                          'Claudius':['Claudius Divus'],
                          'Lucius Verus':['Lucius Verus Divus'],
                          'Marcus Aurelius':['Marcus Aurelius Divus',
                                             'under Marcus Aurelius'],
                          'Trajan': ['Divus Trajan'],
                          'Antoninus Pius':['Antoninus Pius Divus',
                                            'Antoninus'],
                          'Constantius I': ['Constantius I Divus'],
                          'Claudius Gothicus':['Claudius Gothicus Divus',
                                               'Claudius II Divus',
                                               'Claudius II Gothicus',
                                               'Claudius II'],
                          'Hadrian':['Hadrian Divus',
                                     'Divvs Hadrian'],
                          'Nerva':['Nerva Divus'],
                          'Commodus':['Commodus Divus'],
                          'Maximian':['Maximianus'],
                          'Claudius':['Claudius '],
                          'Constantius Chlorus':['Constantius Caesar',
                                                 'Constantius I'],
                          'Caligula':['Gaius)_Caligula'],
                          'Germanicus':['Drusus Caesar',
                                        'Drusus the Elder',
                                        'Germanus Indutilli L.'],
                          'Elagabalus':['Elagabulus'],
                          'Florianus':['Florian'],
                          'Julian the Apostate':['Julian II',
                                                 'Julian'],
                          'Julius Caesar':['Gaius Caesar'],
                          'Licinius':['Licinius I'],
                          'Lucius Aelius':['Lucius Aelius Caesar'],
                          'Maximinus Thrax':['Maximinus I'],
                          'Maximinus Daia':['Maximinus II'],
                          'Philip the Arab':['Philip I'],
                          'Valerian':['Valerian I']
                          }
    
    for key,values in duplicate_emperors:
        for value in values:
            df.loc[(df.portrait == value),'portrait'] = key
    
    # Limiting the scope to emperors with 1000+ coins
    target_df = df[df.groupby('portrait')['portrait'].transform('size') > 1000]
    
    target_df.fname = target_df.fname.apply(lambda x: '../img/' + x + '.jpg')
    
    # Adding a column with encoded values
    LE = LabelEncoder()
    df['code'] = LE.fit_transform(df['portrait'])
    
    # Saving the cleaned dataframe as a csv file
    target_df.to_csv('cleaned_metadata')

    return target_df

def photo_mover(df):
    """
    input:  dataframe or csv generated from dataframe_converter() function
    output: from the inputted dataframe, the function copies and places photos into 
            train,validation and test folders
    """
    print('Initiating file copy and transfer')
    
    from sklearn.model_selection import train_test_split
    import shutil, os

    # Splitting data into train, validation and test 
    X = df.drop('code',axis=1)
    y = df.code

    X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.1)
    
    # Extracting file names from X_train, X_val and X_test for physically separating
    # them into different folders
    train_filenames = [fname for fname in X_train.fname]
    val_filenames = [fname for fname in X_val.fname]
    test_filenames = [fname for fname in X_test.fname]



    for filename in test_filenames:
        
        folder_path = f'../img/'
        
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        shutil.copy(filename, folder_path)

    print('File copy and transfer complete.')