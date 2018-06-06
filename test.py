import datetime
import numpy as np
months_dict = {'January' : 1, 'February' : 2, 'March' : 3, 'April' : 4, 'May' : 5, 'June' : 6, \
'July' : 7, 'August' : 8, 'September' : 9, 'October' : 10, 'November' : 11, 'December' : 12}

Current_month_str = datetime.datetime.today().strftime('%B')

H_m_ind = ['July', 'August', 'August', 'June', 'October', 'October']

H_m_ind_1 = []

for i in range(len(H_m_ind)):
	H_m_ind_1.append(months_dict[H_m_ind[i]])

print(H_m_ind_1)
if np.any(H_m_ind_1 > [12]): print('YO')