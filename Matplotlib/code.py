# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




#Code starts here
data = pd.read_csv(path)
loan_status = data['Loan_Status'].value_counts()
loan_status.plot(kind = 'bar')


# --------------
#Code starts here
property_and_loan = data.groupby(['Property_Area','Loan_Status']).size().unstack()

property_and_loan.plot(kind='bar',stacked=False)
plt.xlabel('Property Area')
plt.ylabel('Loan Status')
plt.xticks(rotation=45, ha='right')


# --------------
#Code starts here
education_and_loan = data.groupby(['Education','Loan_Status']).size().unstack()
education_and_loan.plot(kind='bar',stacked=False)
plt.xlabel('Education Status')
plt.ylabel('Loan Status')
plt.xticks(rotation=45, ha='right')


# --------------
#Code starts here

cond_grad = data['Education']=='Graduate'
graduate = data[cond_grad]


cond_notgrad = data['Education']=='Not Graduate'
not_graduate = data[cond_notgrad]


graduate.plot(kind='density',label='Graduate')
not_graduate.plot(kind='density',label='Graduate')









#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here
fig ,(ax_1,ax_2,ax_3) = plt.subplots(3,1, figsize=(20,10))
res = data.groupby(['ApplicantIncome','LoanAmount']).size().unstack()
data.plot(kind='scatter', stacked=True, ax=ax_1,x='ApplicantIncome',y='LoanAmount')
data.plot(kind='scatter', stacked=True, ax=ax_2,x='CoapplicantIncome',y='LoanAmount')
data.plot(kind='scatter', stacked=True, ax=ax_2,x='CoapplicantIncome',y='LoanAmount')
data['TotalIncome']=data['ApplicantIncome']+data['CoapplicantIncome']
data.plot(kind='scatter', stacked=True, ax=ax_3,x='TotalIncome',y='LoanAmount')
ax_1.set_title('Applicant Income')
ax_2.set_title('Coapplicant Income')
ax_3.set_title('Total Income')


