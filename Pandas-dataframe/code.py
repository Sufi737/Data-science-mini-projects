# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 
bank = pd.read_csv(path,sep=',')

# code starts here
categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)

numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)



# code ends here


# --------------
# code starts here


#code ends here
del bank['Loan_ID']
banks = bank

bank_mode = banks.mode()
banks = banks.fillna(mode)
print(banks)


# --------------
# code starts here

# check the avg_loan_amount



avg_loan_amount = pd.pivot_table(banks,values="LoanAmount", index=["Gender","Married","Self_Employed"], aggfunc=np.mean)
print (avg_loan_amount)
# code ends here


# --------------
# code starts here






loan_approved_se = banks[(banks['Self_Employed'] == "Yes") & (banks['Loan_Status']=='Y')]
loan_approved_nse = banks[(banks['Self_Employed'] == "No") & (banks['Loan_Status']=='Y')]

se_count = len(loan_approved_se.index)
nse_count = len(loan_approved_nse.index)

percentage_se = (se_count/614)*100
percentage_nse = (nse_count/614)*100

print(percentage_se)
print(percentage_nse)
# code ends here


# --------------
# code starts here

loan_term = banks['Loan_Amount_Term'].apply(lambda x: x/12)

big_loan_term_list = loan_term[loan_term >= 25]
big_loan_term = len(big_loan_term_list)
print(big_loan_term)



# code ends here


# --------------
# code starts here




loan_groupby = banks.groupby('Loan_Status')['ApplicantIncome','Credit_History']

mean_values = loan_groupby.agg(np.mean)
print(mean_values)


# code ends here


