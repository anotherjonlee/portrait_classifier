def year_converter(year_string):
    """
    The function will remove 'AD' and 'BC' from the year and convert the string value 'nan'
    into a NaN for imputation.
    
    Input:  Pandas series object
    Output: String objects 
    """
    # Convert "NA" in to NaN
    #df['year'].replace({'NA':None},inplace=True)
    
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
        if year_string == 'nan' or year_string == 'NA':
            year = None
        else:
            year = int(year_string)
    return year

def df_splitter(df, n):
    import pandas as pd

    subDf = df.groupby('portrait').apply(pd.DataFrame.sample, n=n)\
        .reset_index(level=1)
    
    subDf.rename(columns = {'level_1':'original_idx'}, inplace=True)
    
    drop_indx = subDf.original_idx.values
    
    remainderDf = df.drop(drop_indx)
    
    return subDf, remainderDf

def photo_mover(df, fname_dict):
    import shutil, os
    import sys
    sys.path.append("..")

    for foldername, filenames in fname_dict.items():
        for filename in filenames:
            emperor = df[df.fname == filename]['portrait'].values[0]
            
            folder_path = f'../data/{foldername}/'
            
            # Making sure the target directories already exist before copying images
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            
            # Making a emperor sub-directories within the train, validation and holdout folders
            final_path = folder_path + f"{emperor}/"
            
            if not os.path.exists(final_path):
                os.mkdir(final_path)
            
            # Copy and paste files to appropriate sub directories

            shutil.copy(filename, final_path)
    
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
    import json
    import sys
    sys.path.append("..") ## resetting the path to the parent directory
    
    # Making sure that the inputed object is in a correct format
    #assert type(metadata) == list or metadata[-4:] == 'json'
    
    if type(metadata) == list:
        df = pd.DataFrame(metadata)
    
    else:
        with open(metadata) as json_file:
            data = json.load(json_file)
            df = pd.DataFrame.from_dict(data)
    
    # Convert "NA" in to NaN
    #df.replace({'NA':None},inplace=True)

    # The scraped metadata contained photos linked with the emperors' nicknames
    # The following script will consolidate duplicates under a single name.
    duplicate_emperors = {'Augustus':['Divus Augustus',
                                    'Augustus Divus',
                                    'Octavian'],
                        'Valerian II': ['Valerian II Divus'],
                        'Constantine I': ['Constantine I Divus'],
                        'Claudius':['Claudius Divus', 'Claudius '],
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
    
    for key,values in duplicate_emperors.items():
        for value in values:
            df.loc[(df.portrait == value),'portrait'] = key
    
    
    # Appending necessary path information 
    df['fname'] = df.fname.apply(lambda x: '../data/raw_imgs/' + x + '.jpg')
    
    # Adding a column with encoded values
    LE = LabelEncoder()
    df['code'] = LE.fit_transform(df['portrait'])
    
    # Converting the year column into a numeric values and remove 'BC' and 'AD'
    df['year'] = df['year'].apply(lambda x: year_converter(x))
    
    df.to_csv('../data/raw_dataframe.csv')
    
    # Limiting the scope to emperors with 500+ coins
    df = df[df.groupby('portrait')['portrait'].transform('size') > 500]
    
    # Imputing missing years with average years by emperors
    df['year'] = df['year'].fillna(df.groupby('portrait')['year'].transform('mean').round())
    
    # Saving the cleaned dataframe as a csv file
    df.to_csv('../data/cleaned_dataframe.csv')

    return df

def data_mover(df):
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
    
    print('Initiating file transfer.')

    # Splitting data into train, validation and test 

    # Using 200 images as a train set
    train_df, val_test = df_splitter(df,200) 
    
    # Using 100 images to validate and set aside the rest to test 
    # final model
    validation_df, test_df = df_splitter(val_test, 100)
    
    # Extracting file names from train,validation and test df's for 
    # physically separating them into different folders
    
    train_filenames = [fname for fname in train_df.fname]
    val_filenames = [fname for fname in validation_df.fname]
    test_filenames = [fname for fname in test_df.fname]
    
    train_fname_dict = {'train_folder': train_filenames} 
    val_fname_dict = {'validation_folder': val_filenames}
    test_fname_dict = {'holdout_folder': test_filenames}
    
    photo_mover(train_df, train_fname_dict)
    photo_mover(validation_df, val_fname_dict)
    photo_mover(test_df, test_fname_dict)

    print('File transfer complete.')

if __name__ == '__main__':
    fname = '../data/raw_metadata.json'
    cleaned_df = dataframe_converter(fname)
    data_mover(cleaned_df)