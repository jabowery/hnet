import pandas as pd
import os

from Config import Config
def ioLoadCredit(name, path=None):
    assert isinstance(name, str) and name != '', 'name must be a non-empty string'
    if path is None or path == '':
        path = os.path.join(ComputerProfile.DatasetDir(), 'credit', name)
    dataset = {}
    if name == 'uci_credit_screening':
        dataset['t'] = pd.read_csv(os.path.join(path, 'crx.csv'))
        dataset['t'].columns = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15','dv']
        dataset['t']['a9'] = dataset['t']['a9'] == 't'
        dataset['t']['a10'] = dataset['t']['a10'] == 't'
        dataset['t']['a12'] = dataset['t']['a12'] == 't'
        dataset['t']['dv'] = dataset['t']['dv'] == '+'
        dataset['uniq_a1'] = ['a','b']
        dataset['uniq_a4'] = ['u','y','l','t']
        dataset['uniq_a5'] = ['g','p','gg']
        dataset['uniq_a6'] = ['c','d','cc','i','j','k','m','r','q','w','x','e','aa','ff']
        dataset['uniq_a7'] = ['v','h','bb','j','n','z','dd','ff','o']
        dataset['uniq_a13'] = ['g','p','s']
        dataset['t_bin'] = dataset['t'].copy()
        for i in range(len(dataset['uniq_a1'])):
            dataset['t_bin'][f'a1_{dataset["uniq_a1"][i]}'] = dataset['t']['a1'] == dataset['uniq_a1'][i]
        dataset['t_bin'].drop(columns=['a1'], inplace=True)
        for i in range(len(dataset['uniq_a4'])):
            dataset['t_bin'][f'a4_{dataset["uniq_a4"][i]}'] = dataset['t']['a4'] == dataset['uniq_a4'][i]
        dataset['t_bin'].drop(columns=['a4'], inplace=True)
        for i in range(len(dataset['uniq_a5'])):
            dataset['t_bin'][f'a5_{dataset["uniq_a5"][i]}'] = dataset['t']['a5'] == dataset['uniq_a5'][i]
        dataset['t_bin'].drop(columns=['a5'], inplace=True)
        for i in range(len(dataset['uniq_a6'])):
            dataset['t_bin'][f'a6_{dataset["uniq_a6"][i]}'] = dataset['t']['a6'] == dataset['uniq_a6'][i]
        dataset['t_bin'].drop(columns=['a6'], inplace=True)
        for i in range(len(dataset['uniq_a7'])):
            dataset['t_bin'][f'a7_{dataset["uniq_a7"][i]}'] = dataset['t']['a7'] == dataset['uniq_a7'][i]
        dataset['t_bin'].drop(columns=['a7'], inplace=True)
        for i in range(len(dataset['uniq_a13'])):
            dataset['t_bin'][f'a13_{dataset["uniq_a13"][i]}'] = dataset['t']['a13'] == dataset['uniq_a13'][i]
        dataset['t_bin'].drop(columns=['a13'], inplace=True)
    elif name == 'uci_statlog_australian_credit':
        dataset['t'] = pd.read_csv(os.path.join(path, 'australian.csv'))
        dataset['t'].columns = ['a1_idx','a2','a3','a4_idx','a5_idx','a6_idx','a7','a8','a9','a10','a11','a12_idx','a13','a14','dv']
        dataset['t']['a1_idx'] = dataset['t']['a1_idx'] + 1
        dataset['t']['a8'] = dataset['t']['a8'].astype(bool)
        dataset['t']['a9'] = dataset['t']['a9'].astype(bool)
        dataset['t']['a11'] = dataset['t']['a11'].astype(bool)
        dataset['t']['dv'] = dataset['t']['dv'].astype(bool)
        dataset['uniq_a1'] = ['a','b']
        dataset['uniq_a4'] = ['p','g','gg']
        dataset['uniq_a5'] = ['ff','d','i','k','j','aa','m','c','w','e','q','r','cc','x']
        dataset['uniq_a6'] = ['ff','dd','j','bb','v','n','o','h','z']
        dataset['uniq_a12'] = ['s','g','p']
        dataset['t_bin'] = dataset['t'].copy()
        dataset['t_bin']['a1'] = dataset['t']['a1_idx'] - 1
        dataset['t_bin'].drop(columns=['a1_idx'], inplace=True)
        dataset['t_bin']['a4_p'] = dataset['t']['a4_idx'] == 1
        dataset['t_bin']['a4_g'] = dataset['t']['a4_idx'] == 2
        dataset['t_bin']['a4_gg'] = dataset['t']['a4_idx'] == 3
        dataset['t_bin'].drop(columns=['a4_idx'], inplace=True)
        for i in range(len(dataset['uniq_a5'])):
            dataset['t_bin'][f'a5_{dataset["uniq_a5"][i]}'] = dataset['t']['a5_idx'] == i
        dataset['t_bin'].drop(columns=['a5_idx'], inplace=True)
        for i in range(len(dataset['uniq_a6'])):
            dataset['t_bin'][f'a6_{dataset["uniq_a6"][i]}'] = dataset['t']['a6_idx'] == i
        dataset['t_bin'].drop(columns=['a6_idx'], inplace=True)
        dataset['t_bin']['a12_s'] = dataset['t']['a12_idx'] == dataset['uniq_a12'].index('s')
        dataset['t_bin']['a12_g'] = dataset['t']['a12_idx'] == dataset['uniq_a12'].index('g')
        dataset['t_bin']['a12_p'] = dataset['t']['a12_idx'] == dataset['uniq_a12'].index('p')
        dataset['t_bin'].drop(columns=['a12_idx'], inplace=True)
    elif name == 'uci_statlog_german_credit':
        dataset['t'] = pd.read_csv(os.path.join(path, 'german.txt'), sep=' ')
        dataset['t'].columns = ['a1_idx','a2_duration','a3_idx','a4_idx','a5_creditscore','a6_idx','a7_idx','a8_percent','a9_idx','a10_idx','a11_presentresidencesince','a12_idx','a13_age','a14_idx','a15_idx','a16_ncredits','a17_idx','a18_ndependents','a19_hasphone','a20_isforeign','dv']
        dataset['t']['a19_hasphone'] = dataset['t']['a19_hasphone'] == 'A192'
        dataset['t']['a20_isforeign'] = dataset['t']['a20_isforeign'] == 'A201'
        dataset['t']['dv'] = dataset['t']['dv'] - 1
        dataset['category_info'] = {}
        dataset['category_info']['a1_A11'] = 'salary for at least 1 year = under zero dm'
        dataset['category_info']['a1_A12'] = 'salary for at least 1 year = under 200 dm'
        dataset['category_info']['a1_A13'] = 'salary for at least 1 year = over 200 dm'
        dataset['category_info']['a1_A14'] = 'no checking acct'
        dataset['category_info']['a3_A30'] = 'no credits taken / all credits paid back duly'
        dataset['category_info']['a3_A31'] = 'all credits at this bank paid back duly'
        dataset['category_info']['a3_A32'] = 'existing credits paid back duly till now'
        dataset['category_info']['a3_A33'] = 'delay in paying off in the past'
        dataset['category_info']['a3_A34'] = 'critical account / other credits existing (not at this bank)'
        dataset['category_info']['a4_A40'] = 'purpose = car (new)'
        dataset['category_info']['a4_A41'] = 'purpose = car (used)'
        dataset['category_info']['a4_A42'] = 'purpose = furniture / equipment'
        dataset['category_info']['a4_A43'] = 'purpose = radio / television'
        dataset['category_info']['a4_A44'] = 'purpose = domestic appliances'
        dataset['category_info']['a4_A45'] = 'purpose = repairs'
        dataset['category_info']['a4_A46'] = 'purpose = education'
        dataset['category_info']['a4_A47'] = 'purpose = (vacation - does not exist?)'
        dataset['category_info']['a4_A48'] = 'purpose = retraining'
        dataset['category_info']['a4_A49'] = 'purpose = business'
        dataset['category_info']['a4_A410'] = 'purpose = other'
        dataset['category_info']['a6_A61'] = 'savings and bonds = < 100 dm'
        dataset['category_info']['a6_A62'] = 'savings and bonds = 100 to 500 dm'
        dataset['category_info']['a6_A63'] = 'savings and bonds = 500 to 1000 dm'
        dataset['category_info']['a6_A64'] = 'savings and bonds = over 1000 dm'
        dataset['category_info']['a6_A65'] = 'savings and bonds = unknown / none'
        dataset['category_info']['a7_A71'] = 'unemployed'
        dataset['category_info']['a7_A72'] = 'present job held < 1 year'
        dataset['category_info']['a7_A73'] = 'present job held 1 to 4 years'
        dataset['category_info']['a7_A74'] = 'present job held 4 to 7 years'
        dataset['category_info']['a7_A75'] = 'present job held over 7 years'
        dataset['category_info']['a9_A91'] = 'male, divorced / separated'
        dataset['category_info']['a9_A92'] = 'female, divorced / separated / married'
        dataset['category_info']['a9_A93'] = 'male, single'
        dataset['category_info']['a9_A94'] = 'male, married / widowed'
        dataset['category_info']['a9_A95'] = 'female, single'
        dataset['category_info']['a10_A101'] = 'other debtors / guarantors = none'
        dataset['category_info']['a10_A102'] = 'other debtors / guarantors = co-applicant'
        dataset['category_info']['a10_A103'] = 'other debtors / guarantors = guarantor'
        dataset['category_info']['a12_A121'] = 'property = real estate'
        dataset['category_info']['a12_A122'] = 'property = if not A121 : building society savings agreement / life insurance'
        dataset['category_info']['a12_A123'] = 'property = if not A121/A122 : car or other, not in attribute 6'
        dataset['category_info']['a12_A124'] = 'property = unknown / none'
        dataset['category_info']['a14_A141'] = 'other installment plans = bank'
        dataset['category_info']['a14_A142'] = 'other installment plans = stores'
        dataset['category_info']['a14_A143'] = 'other installment plans = none'
        dataset['category_info']['a15_A151'] = 'housing = rent'
        dataset['category_info']['a15_A152'] = 'housing = own'
        dataset['category_info']['a15_A153'] = 'housing = for free'
        dataset['category_info']['a17_A171'] = 'job = unemployed / unskilled - non-resident'
        dataset['category_info']['a17_A172'] = 'job = unskilled - resident'
        dataset['category_info']['a17_A173'] = 'job = skilled employee / official'
        dataset['category_info']['a17_A174'] = 'job = management / self-employed / highly qualified employee / officer'
        for i in [1,3,4,6,7,9,10,12,14,15,17]:
            dataset[f'uniq_a{i}_idx'] = dataset['t'][f'a{i}_idx'].unique()
        dataset['t_bin'] = dataset['t'].copy()
        for i in [1,3,4,6,7,9,10,12,14,15,17]:
            varName = f'a{i}_idx'
            for j in range(len(dataset[f'uniq_{varName}'])):
#                print(f'{varName}_{j} = uniq_{varName}')
                dataset['t_bin'][f'{varName}_{dataset[f"uniq_{varName}"][j]}'] = dataset['t'][varName] == dataset[f'uniq_{varName}'][j]
            dataset['t_bin'].drop(columns=[varName], inplace=True)
    elif name == 'kaggle_icl_loan_default_prediction':
        raise NotImplementedError('TODO')
    else:
        raise ValueError('unexpected name')
    return dataset


