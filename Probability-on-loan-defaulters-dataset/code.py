# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)
num_pico = len(df[df['fico'] > 700])
total = len(df)
p_a = num_pico/total
print(p_a)

num_purpose = len(df[df['purpose'] == 'debt_consolidation'])
p_b = num_purpose/total
print(p_b)

df1 = df[df['purpose'] == 'debt_consolidation']
num_ab = len(df[df['fico'] > 700])
p_ab = num_ab/total
p_a_b = p_ab/p_a
print(p_a_b)

result = p_a_b == p_a
print(result)
# code ends here


# --------------
# code starts here
num_pb = len(df[df['paid.back.loan'] == 'Yes'])
prob_lp = num_pb/total
print(prob_lp)

num_cs = len(df[df['credit.policy'] == 'Yes'])
prob_cs = num_cs/total
print(prob_cs)

new_df = df[df['paid.back.loan'] == 'Yes']
num_pd_cs = len(new_df[new_df['credit.policy'] == 'Yes'])
prob_intersect = num_pd_cs/total
prob_pd_cs = prob_intersect/prob_lp
print(prob_pd_cs)

bayes = (prob_pd_cs*prob_lp)/prob_cs
print(bayes)

# code ends here


# --------------
# code starts here
df['purpose'].value_counts().plot(kind='bar')
df1 = df[df['paid.back.loan'] == 'No']
df1['purpose'].value_counts().plot(kind='bar')
# code ends here


# --------------
# code starts here
inst_median = df['installment'].median()
inst_mean = df['installment'].mean()

df['installment'].value_counts().plot(kind='hist')
df['log.annual.inc'].value_counts().plot(kind='hist')
# code ends here


