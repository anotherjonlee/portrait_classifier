'''
1. Run data_collection.py to initiate the file download
2. Parse dictionary into dataframe and save as a csv file
3. Move downloaded images into train, validation and test 
3. plot basic eda graphs
'''

import data_collection as data
import seaborn as sns
        
def dataframe_converter(lst):
    """
    input:  list or json file downloaded from data.img_downloader
    output: csv file saved in a data folder
    """
    
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    import pandas as pd
    import sys
    sys.path.append("..") ## resetting the path to the parent directory
    
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
    
    # Appending necessary path information 
    target_df['fname'] = target_df.fname.apply(lambda x: '../data/raw_imgs/' + x + '.jpg')
    
    # Adding a column with encoded values
    LE = LabelEncoder()
    target_df['code'] = LE.fit_transform(target_df['portrait'])
    
    # Converting the year column into a numeric values and remove 'BC' and 'AD'
    target_df['year'] = df['year'].apply(lambda x: year_converter(x))
    
    # Imputing missing years with average years by emperors
    target_df['year'] = target_df['year'].fillna(target_df.groupby('portrait')['year'].transform('mean').round())
    
    # Saving the cleaned dataframe as a csv file
    target_df.to_csv('../data/cleaned_metadata.csv')

    return target_df

def year_converter(year_string):
    
    year_string = str(year_string)
    
    split_string = year_string.split(' ')
    
    if len(split_string) > 1:
        if 'AD' in split_string:
            idx = split_string.index('AD')
            year = int(split_string[len(split_string) - 1 - idx])
        elif 'BC' in split_string:
            idx = split_string.index('BC')
            year = int('-' + split_string[len(split_string) - 1 - idx])
    else:
        if year_string == 'nan':
            year = None
        else:
            year = int(year_string)
    return year

def photo_mover(df):
    """
    input:  dataframe or csv generated from dataframe_converter() function
    output: from the inputted dataframe, the function copies and places photos into 
            train,validation and test folders
    """
    import shutil, os
    import sys
    sys.path.append("..") ## resetting the path to the parent directory
    
    print('Initiating file copy and transfer')
    
    from sklearn.model_selection import train_test_split
    import shutil, os

    # Splitting data into train, validation and test 

    X_train, X_test = train_test_split(X, test_size=0.1)
    X_train, X_val = train_test_split(X_train, test_size=0.1)
    
    # Extracting file names from X_train, X_val and X_test for physically separating
    # them into different folders
    train_filenames = [fname for fname in X_train.fname]
    val_filenames = [fname for fname in X_val.fname]
    test_filenames = [fname for fname in X_test.fname]

    # Running a for loop to copy and move target files into appropriate directories
    filenames = {'train_folder': train_filenames, 
                 'validation_folder': val_filenames, 
                 'holdout_folder': test_filenames}
    
    for foldername, filename in filenames.items():
        
        emperor = df[df.fname == filename]['portrait'].values[0]
        
        folder_path = f'../data/{foldername}/'
        
        # Making sure the target directories already exist before copying images
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        # Making a emperor sub-directories within the train, validation and holdout folders
        folder_path += f'{emperor}/'
        
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        # Copy and paste files to appropriate sub directories
        shutil.copy(filename, folder_path)

    print('File copy and transfer complete.')


    
if __name__ == '__main__':
    # Warning: data.img_downloader will attempt to download the entire image set from the source website.
    metadata_lst = data.img_downloader()
    cleaned_df = dataframe_converter(metadata_lst)
    photo_mover(cleaned_df)