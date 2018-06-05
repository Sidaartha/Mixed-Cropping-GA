#----------------------------------------- Importing Libraries -----------------------------------------

import random
import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import style
from collections import Counter
from operator import itemgetter
from prettytable import PrettyTable
from deap import algorithms, base, tools, creator
style.use('ggplot')

#-------------------------------------------- Reading Data ---------------------------------------------

# Reading CSV file
df = pd.read_csv('Gudur_Rythu_Bazar_2017.csv')
df.drop(['Comments'], axis = 1, inplace=True)	# Dropping 'Comments' column

# np arrays of colomns
Harvest_time = df['Maturity_mo']
Harvest_time = np.array(Harvest_time)
Crop_name 	= df['Type']
Crop_name 	= np.array(Crop_name)
Culti_cost 	= df['Cost_Culti_acre']
Culti_cost 	= np.array(Culti_cost)
Root_depth 	= df['Root_Depth']
Root_depth 	= np.array(Root_depth)
Water_req 	= df['Water_Req']
Water_req 	= np.array(Water_req)
Profit 	= df['Profit']
Profit 	= np.array(Profit)
Month 	= df['Month']
Month 	= np.array(Month)
Type 	= df['Type_Code']
Type 	= np.array(Type)

#--------------------------------------------- Variable info ----------------------------------------------

months_ = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
months_dict = {'January' : 1, 'February' : 2, 'March' : 3, 'April' : 4, 'May' : 5, 'June' : 6, \
'July' : 7, 'August' : 8, 'September' : 9, 'October' : 10, 'November' : 11, 'December' : 12}
Current_month = datetime.datetime.now().month
Current_month = 9
Current_month_str = datetime.datetime.today().strftime('%B')

Debug = False
print_ = False

n 	= 300
m	= 6 		# No.of crops to decide
n_i = Type[0] 	# Lower limit of no.of crops
n_f	= Type[-1] 	# Upper limit of no.of crops / Total no.of crops
NGen 	= 10		# Number of generations/Number of itterations			
CXPB	= 0.5		# CXPB  is the probability with which two individuals are crossed
MUTPB 	= 0.2		# MUTPB is the probability for mutating an individual

# Weights to cal weighted avg
profit_wt	= 0.7
risk_wt 	= -0.3
root_risk_wt	= 0.5
water_risk_wt	= 0.5
crops_cycle = []

#----------------------------------------------- Fitness Function ----------------------------------------------
# Objective fun : [1] Maximize Profit
#				  [2] Mininize Risk 
# 			i.e : Max(W1*Profit -W2*Risk)
# Subject to constrains : [1] Harvest time.
#						  [2] Based on root system.
#						  [3] Based on water requirement.

def Fitness_value(individual):

	global profit
	global harvest_month
	global planting_month

	profit = []
	harvest_month = []
	planting_month = []
	root_depth = []
	water_req = []

	#---------------------------------------------- Estimating Profit -----------------------------------------

	if len(set(crops_cycle) & set(individual)) == 0:
		if len(set(individual))==m:
			for i in range(len(individual)):
				harvest_month_itt = []
				planting_month_itt = []
				Crop = individual[i]
				for e in range(len(Type)):
					if Type[e]==Crop:
						type_id = e
						break
					else:
						pass
				profit_id = type_id + Current_month + Harvest_time[type_id] -1
				id_verify = Current_month + Harvest_time[type_id] -1
				if id_verify < 12:
					profit_i = Profit[profit_id]
					planting_month_itt=Month[profit_id - Harvest_time[type_id]]
					harvest_month_itt=Month[profit_id]
					# break
				else:
					profit_i = Profit[type_id + profit_id%12]
					planting_month_itt=Month[type_id + profit_id%12 - Harvest_time[type_id]]
					harvest_month_itt=Month[type_id + profit_id%12]
					# break
				profit.append(profit_i)
				planting_month.append(planting_month_itt)
				harvest_month.append(harvest_month_itt)
				root_depth.append(Root_depth[type_id])
				water_req.append(Water_req[type_id])
		else:
			profit=[0]
	else:
		profit=[0]

	# if len(set(crops_cycle) & set(individual)) != 0: profit=[0]

	Profit_percent = sum(profit)/10**4

	#---------------------------------------------- Estimating Risk -------------------------------------------

	list_risk=[]

	# Risk due to competition over nitrogen from the soil
	# lower limit = 0
	# upper limit = 100
	list_abc_1=[]
	counts = Counter(root_depth)
	per_s = counts['Shallow']*100/m
	per_m = counts['Medium']*100/m
	per_d = counts['Deep']*100/m
	
	if per_s and per_m != 0:
		a_1 = abs(per_s - per_m)
		list_abc_1.append(a_1)
	if per_s and per_d != 0:
		b_1 = abs(per_s - per_d)
		list_abc_1.append(b_1)
	if per_m and per_d != 0:
		c_1 = abs(per_m - per_d)
		list_abc_1.append(c_1)
	if len(list_abc_1) != 0:
		avg_abc_1 = sum(list_abc_1)/len(list_abc_1)
	else:
		avg_abc_1 = 100
	list_risk.append(avg_abc_1)

	# Risk due to competition over water requirement
	# lower limit = m*20
	# upper limit = m*50
	list_abc_2=[]
	counts_water = Counter(water_req)
	per_L = counts_water['L']
	per_M = counts_water['M']
	per_H = counts_water['H']
	
	avg_abc_2 = 20*per_L + 30*per_M + 50*per_H

	list_risk.append(avg_abc_2)
	
	# risk = (root_risk_wt*list_risk[0] + water_risk_wt*list_risk[1])/(root_risk_wt + water_risk_wt)
	risk = (root_risk_wt*list_risk[0] + water_risk_wt*list_risk[1])
	Risk_percent = risk

	#-----------------------------------------------------------------------------------------------------------
	
	combined_val = (profit_wt*Profit_percent+risk_wt*Risk_percent)/(profit_wt+risk_wt)
	# combined_val = (profit_wt*Profit_percent+risk_wt*Risk_percent)
	
	if Debug == True:
		print('-- Debugging --')
		print('Profit_val 	: %s \nRisk_val 	: %s \nCombined_val 	: %s \nRisk_root 	: %s \nRisk_water 	: %s' \
			%(Profit_percent, Risk_percent, combined_val, avg_abc_1, avg_abc_2) )
	else:
		pass

	# return sum(profit), risk
	return combined_val, 

# ------------------------------------------------ Creating class -----------------------------------------------

# creator.create('FitnessMax', base.Fitness, weights = (1.0, -1.0))
creator.create('FitnessMax', base.Fitness, weights = (1.0, ))
creator.create('Individual', list, fitness = creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('attr_value', random.randint, n_i, n_f)	# generator

# genetic operators required for the evolution
toolbox.register('evaluate', Fitness_value)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutUniformInt, low=n_i, up=n_f, indpb=0.2)
toolbox.register('select', tools.selTournament, tournsize=3)

#------------------------------------------ Evolution operation ----------------------------------------------

def Evolution(m, n, CXPB, MUTPB, NGen):

	Max_=[]
	Avg_=[]
	Std_=[]

	# Structure initializers
	toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_value, m)	
	toolbox.register('population', tools.initRepeat, list, toolbox.individual)

	# create an initial population of 'n' individuals
	pop = toolbox.population(n)

	if print_ == True: print("Start of evolution")
	
	# Evaluate the entire population
	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
	
	if print_ == True: print("  Evaluated %i individuals" % len(pop))

	# Extracting all the fitnesses of 
	fits = [ind.fitness.values[0] for ind in pop]

	# Begin the evolution
	for g in range(NGen):

		gen = g+1
		if print_ == True: print("-- Generation %i --" % gen)
		
		# Select the next generation individuals
		offspring = toolbox.select(pop, len(pop))
		# Clone the selected individuals
		offspring = list(map(toolbox.clone, offspring))
	
		# Apply crossover and mutation on the offspring
		for child1, child2 in zip(offspring[::2], offspring[1::2]):

			# cross two individuals with probability CXPB
			if random.random() < CXPB:
				toolbox.mate(child1, child2)

				# fitness values of the children
				# must be recalculated later
				del child1.fitness.values
				del child2.fitness.values

		for mutant in offspring:

			# mutate an individual with probability MUTPB
			if random.random() < MUTPB:
				toolbox.mutate(mutant)
				del mutant.fitness.values
	
		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit
		if print_ == True: print("  Evaluated %i individuals" % len(invalid_ind))
		
		# The population is entirely replaced by the offspring
		pop[:] = offspring
		
		# Gather all the fitnesses in one list and print the stats
		fits = [ind.fitness.values[0] for ind in pop]
		
		length = len(pop)
		mean = sum(fits) / length
		sum2 = sum(x*x for x in fits)
		std = abs(sum2 / length - mean**2)**0.5

		Max_.append(max(fits))
		Avg_.append(mean)
		Std_.append(std)

		if print_ == True: print("  Min %s" % min(fits))
		if print_ == True: print("  Max %s" % max(fits))
		if print_ == True: print("  Avg %s" % mean)
		if print_ == True: print("  Std %s" % std, '\n')

	if print_ == True: print("-- End of successful evolution --")

	Best = tools.selBest(pop, 1)[0]	

	#---------------------------------------------- Visualisation --------------------------------------------

	# x_ = np.arange(1,len(Max_)+1)
	# plt.bar(x_-0.2, Max_, width = 0.2,align='center', label='Max')
	# plt.bar(x_, Avg_, width = 0.2,align='center', label='Avg')
	# plt.bar(x_+0.2, Std_, width = 0.2,align='center', label='Std')
	# plt.axis([0, NGen+1, 0, 1.4*max(Max_)])
	# plt.axes().xaxis.set_major_locator(ticker.MultipleLocator(1))
	# plt.xlabel('Generation')
	# plt.ylabel('Total Profit')
	# plt.title('Max - Avg - Std')
	# plt.legend()
	# plt.show()

	#---------------------------------------- Storing output to 't' ------------------------------------------

	# To access global variables, To store output of each crop of 'Best' individual
	Fitness_value(Best)

	# Data in table format
	Total_profit = 0
	t = PrettyTable(['Crop','Planting Month', 'Harvest Month', 'Root Sys', \
		'Water Req', 'Culti Cost', 'Profit'])
	for i in range(len(Best)):
		val = Best[i]
		t.add_row([Crop_name[val*12-1], planting_month[i], harvest_month[i], \
		Root_depth[val*12-1], Water_req[val*12-1], len(harvest_month[i])*Culti_cost[val*12-1], profit[i]])
		Total_profit = Total_profit + profit[i]

	return Best, t, Total_profit, harvest_month, planting_month

#--------------------------------------------- Single best crop ----------------------------------------------

def SingleCrop(single_crop, c_s):

	Current_month = H_m_ind_val[single_crop]+1
	profit_single = []
	for i in range(n_f):
		id_verify_s = Current_month-1+Harvest_time[12*i]
		if id_verify_s <12:
			SC_id = Current_month-1+12*i+Harvest_time[Current_month-1+12*i]
			profit_single.append([Profit[SC_id], SC_id])
		else:
			SC_id = 12*i + id_verify_s%12
			profit_single.append([Profit[SC_id], SC_id])
	profit_single = sorted(profit_single, key=itemgetter(0))
	for i in range(len(profit_single)):
		profit_single_m = profit_single[-(i+1)][0]
		single_id = profit_single[-(i+1)][1]
		if all(Type[single_id] != c_s): break

	# Data in table format
	t_s = PrettyTable(['Crop','Planting Month', 'Harvest Month', 'Root Sys', 'Water Req', 'Culti Cost', 'Profit'])
	t_s.add_row([Crop_name[single_id], Month[Current_month-1], Month[single_id], \
	Root_depth[single_id], Water_req[single_id], Culti_cost[single_id], Profit[single_id]])

	return Type[single_id], t_s, profit_single_m, Month[single_id], months_dict[Month[Current_month-1]]

# ======================================== Running Genetic Algorithm =========================================

count_ga=0
TotalProfit=[]

visual = []

while True:

	if count_ga == 0:
		print('\n Crop cycle : %s\n'%(count_ga +1))

		Best_ind, t_ind, T_p_ind, H_m_ind, _ = Evolution(m, n, CXPB, MUTPB, NGen)
		TotalProfit.append(T_p_ind)

		# [visual.append([Current_month, months_dict[H_m_ind[vi]], vi+1, Best_ind[vi]]) for vi in range(len(Best_ind))]
		# print(visual)
		
		print("Best individual is %s, %s" % (Best_ind, Best_ind.fitness.values))
		print(t_ind)
		print("Total Profit : %s " % T_p_ind)
		print('\n Crop cycle : %s\n'%(count_ga +2))
	else : 
		print('\n Crop cycle : %s\n'%(count_ga +2))

	H_m_ind_val = []
	for e in range(len(H_m_ind)):
		H_m_ind_val.append(months_dict[H_m_ind[e]])
	H_m_ind_val=sorted(H_m_ind_val, key=int)
	counter_h = Counter(H_m_ind_val)
	H_m_ind_val=list(set(H_m_ind_val))
	crop_s=[]
	H_m_ind=[]
	crops_cycle = []
	for H_cycles in range(len(H_m_ind_val)):
		Current_month = H_m_ind_val[H_cycles]
		m = counter_h[Current_month]
		if m == 1 :
			crop_s.append(H_cycles)
		else:
			Current_month = H_m_ind_val[H_cycles]+1

			Best_ind, t_ind, T_p_ind, H_m_ind_i, _ = Evolution(m, n, CXPB, MUTPB, NGen)
			[ H_m_ind.append(H_m_ind_i[ii]) for ii in range(len(H_m_ind_i)) ]
			TotalProfit.append(T_p_ind)
			[crops_cycle.append(Best_ind[ii]) for ii in range(len(Best_ind))]

			# crop_id = []
			# mo_ = Current_month -1
			# for iii in range(len(visual)):
			# 	if visual[iii][1] == mo_: crop_id.append(visual[iii][2])
			# [visual.append([Current_month, months_dict[H_m_ind[vi]], crop_id[vi], Best_ind[vi]]) for vi in range(len(Best_ind))]
			# print(visual)

			print("Best individual is %s, %s" % (Best_ind, Best_ind.fitness.values))
			print(t_ind)
			print("Total Profit : %s " % T_p_ind)

	for S_cycle in range(len(crop_s)):
		Best_ind, t_ind, T_p_ind, H_m_ind_i, cm = SingleCrop(crop_s[S_cycle], crops_cycle)
		H_m_ind.append(H_m_ind_i)
		TotalProfit.append(T_p_ind)
		crops_cycle.append(Best_ind)

		# mo_ = cm -1
		# for iii in range(len(visual)):
		# 	if visual[iii][1] == mo_: crop_id_s = visual[iii][2]
		# visual.append([cm, months_dict[H_m_ind_i]+12, crop_id_s, Best_ind])
		# print(visual)

		print("Best individual is ", Best_ind)
		print(t_ind)
		print("Total Profit : %s " % T_p_ind)
	print('Best individual for crop cycle-%s is : %s' %((count_ga +2), crops_cycle))
	if count_ga == 2 : break
	count_ga+=1

# print(sum(TotalProfit))

#----------------------------------------------- Visualisation ------------------------------------------------

# for i in range(len(visual)):
# 	plt.plot([visual[i][0], visual[i][1]], [visual[i][2], visual[i][2]], label=Crop_name[visual[i][3]*12-1])
# 	plt.scatter(visual[i][0], visual[i][2],marker='>')
# 	plt.scatter(visual[i][1], visual[i][2],marker='o')
# m=6
# plt.yticks(range(1, m+4), [str(x+1)+'st crop' for x in range(m)])
# plt.xticks(range(1, 26), months_+months_)
# plt.ylabel('Crops')
# plt.xlabel('Months')
# plt.title('Crop Cycles')
# plt.legend()
# plt.show()

#------------------------------------------------ Debugging ---------------------------------------------------

Debug = True
# Fitness_value(best_month[id_month])
# Fitness_value([1, 3, 1, 16, 13])

# Best_ind, t_ind, T_p_ind, H_m_ind_i = SingleCrop([0,2,3])
# H_m_ind.append(H_m_ind_i)
# print("Best individual is ", Best_ind)
# print(t_ind)
# print("Total Profit : %s " % T_p_ind)
# print(H_m_ind_i)
# print(H_m_ind)
