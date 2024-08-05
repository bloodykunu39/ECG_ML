# module
import os
import wfdb
import numpy as np
import pandas as pd


def disease_codes(data):
    disease_codes = data[1]['comments'][2].split()[1].split(',')
    return disease_codes

def datasepration_single(code :int,path:str = 'data/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords')->list:
    """ This function will seperate sample has only one disease and it code is same as input code.
    Args:
        code : int : code of disease
        path : str : path of data folder(..\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords)
    Returns:
        list : list of sample has only one disease and it code is same as input code.
    """
    disease_list = []
    path_1_list = os.listdir(path) #list of folder in the path

    for i in range(len(path_1_list)):

        relative_path_1 = path+'/'+path_1_list[i] # ex : KARAN_ECG\data_prep\data\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords\01
        
        path_2_list = os.listdir(relative_path_1)#list of folder in the path_1


        for j in range(len(path_2_list)):# exploring all folder in the path_1
            path_2 = path_2_list[j]


            path_to_dir = relative_path_1 + '/' + path_2 + '/'

            records = open(path_to_dir+'RECORDS','r').read().split()
            # reocords is a list of all the records in the path2 folder

            for k in range(len(records)):
                data = wfdb.rdsamp(path_to_dir+records[k])# explained above
                disease_codes_ = disease_codes(data)# return the desease codes of the record AS SHOWN ABOVE
                #len=0 and code is in disease_codes_
                if len(disease_codes_)==1 and disease_codes_[0]==str(code):
                    disease_list.append(data[0])
    return disease_list



def datasepration_multiple(code:list,claass:list,path : str='data/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords')->dict:
    """"
    this function will return a dictionary of samples {may have multiple diseases} and their respective classes
    Args:
        code : list : list of list of disease codes
        claass : list : list of classes or labels
        path : str : path of data folder(..\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords)
        Returns:
            dict : dictionary of samples {may have multiple diseases} and their respective labels


        example:
            code = [[426783006, 164889003], [164889003, 426783006, 713427006]]
            claass = ['Myocardial infarction', 'Myocardial infarction and Cardiomyopathy']
            datasepration_multiple(code,claass)
            >>> {'Myocardial infarction': [array([[0.0000e+00, 0.0000e+00, 0.0000e+00, ..., 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, ..., 0.0000e+00, 0.0000e+00, 0.0000e+00],...],
            'Myocardial infarction and Cardiomyopathy': [array([[0.0000e+00, 0.0000e+00, 0.0000e+00, ..., 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [0.0000e+00, 0.0000e+00, 0.0000e+00, ..., 0.0000e+00, 0.0000e+00, 0.0000e+00],...]}
            """

    dic = {}
    path_1_list = os.listdir(path) #list of folder in the path


    for i in range(len(claass)):
        dic[claass[i]] = []
    for i in range(len(path_1_list)):
        path_1 = path_1_list[i]#path_1 is the folder name inside the WFDBRecords
        relative_path_1 = path+'/'+path_1
        path_2_list = os.listdir(relative_path_1)#list of folder in the path_1
 


        for j in range(len(path_2_list)):# checking each folder in the path_1
            path_2 = path_2_list[j]


            path_to_dir = relative_path_1 + '/' + path_2 + '/'

            records = open(path_to_dir+'RECORDS','r').read().split()#WFDBRecords/path1/path2/RECORDS


            for k in range(len(records)):
                data = wfdb.rdsamp(path_to_dir+records[k])# explained above
                disease_codes_ = disease_codes(data)# return the desease codes of the record AS SHOWN ABOVE
                disease_codes_ = [int(element) for element in disease_codes_]
                checker = 0
                for g in range(len(code)):
                    if checker >= 2:
                        break                    
                    for h in range(len(disease_codes_)):
                        if disease_codes_[h] in code[g]:
                            found_in=g
                            checker += 1
                            #print('found_in:',found_in)
                            break
                    
                if checker == 1:
                    dic[claass[found_in]].append(data[0])
    # return dic as pd dataframe
    return dic

