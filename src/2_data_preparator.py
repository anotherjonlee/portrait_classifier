
def year_converter(year_string):
    """
    The function will remove 'AD' and 'BC' from the year and convert the string value 'nan'
    into a NaN for imputation.
    
    Input:  Pandas series object
    Output: String objects 
    """
    
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
        
def dataframe_converter(metadata):
    """
    The function will take in either a list or a json file produced by data_collection.py file.
    It will consolidate duplicates and discard emperors with less than 1000 coins.
    It will then impute the missing values in the year column with the average years by emperors.
    The function will return a pandas dataframe for further analysis, as well as a back .csv file.
    
    Input:  List from data_collection.py or json file downloaded from data.img_downloader
    Output: .csv file saved in a data folder as a backup and a pandas dataframe
    """
    
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import json
    import sys
    sys.path.append("..") ## resetting the path to the parent directory
    
    # Making sure that the inputed object is in a correct format
    assert type(metadata) == list or metadata[-4:] == 'json'
    
    if type(metadata) == list:
        df = pd.DataFrame(metadata)
    
    else:
        with open('../data/raw_metadata.json') as json_file:
            data = json.load(json_file)
            df = pd.DataFrame.from_dict(data)
    
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
    
    
    # Appending necessary path information 
    df['fname'] = df.fname.apply(lambda x: '../data/raw_imgs/' + x + '.jpg')
    
    # Adding a column with encoded values
    LE = LabelEncoder()
    df['code'] = LE.fit_transform(df['portrait'])
    
    # Converting the year column into a numeric values and remove 'BC' and 'AD'
    df['year'] = df['year'].apply(lambda x: year_converter(x))
    
    df.to_csv('../data/raw_metadata.csv')
    
    # Limiting the scope to emperors with 1000+ coins
    df = df[df.groupby('portrait')['portrait'].transform('size') > 1000]
    
    # Imputing missing years with average years by emperors
    df['year'] = df['year'].fillna(df.groupby('portrait')['year'].transform('mean').round())
    
    # Saving the cleaned dataframe as a csv file
    df.to_csv('../data/cleaned_metadata.csv')

    return df

def photo_mover(df):
    """
    The function will take in a processed dataframe from dataframe_converter() function.
    It will split the information into train, validation and test sets, copy and move images to training,
    validation and hold out folders for the tensorflow function to ingest.
    
    input:  dataframe or csv generated from dataframe_converter() function
    output: from the inputted dataframe, the function copies and places photos into 
            train,validation and test folders
    """
    import shutil, os
    import sys
    sys.path.append("..") 
    
    # Making sure that the function is taking in a cleaned dataframe.
    assert (df['portrait'].value_counts()>1000).sum() == 12
    
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
    filenames = {
        'train_folder': train_filenames, 
        'validation_folder': val_filenames, 
        'holdout_folder': test_filenames
    }
    
    # The loop will create missing directories and copy pictures to appropriate sub-directories
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
    import 1_data_scraper as data
    metadata_lst = data.img_downloader()
    cleaned_df = dataframe_converter(metadata_lst)
    photo_mover(cleaned_df)