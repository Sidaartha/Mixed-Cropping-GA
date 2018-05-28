#----------------------------------------- Importing Libraries -----------------------------------------

import random
import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import style
from collections import Counter
from num2words import num2words
from prettytable import PrettyTable
from deap import algorithms, base, tools, creator
style.use('ggplot')

#--------------------------------------------------------------------------------------------------------

# Reading CSV file
df = pd.read_csv('Gudur_Rythu_Bazar_2017.csv')
df.drop(['Comments'], axis = 1, inplace=True)	#Dropping 'Comments' column

# np arrays of colomns
Harvest_time = df['Maturity_mo']
Harvest_time = np.array(Harvest_time)
Month = df['Month']
Month = np.array(Month)
Crop_name = df['Type']
Crop_name = np.array(Crop_name)
Profit = df['Profit']
Profit = np.array(Profit)
Culti_cost = df['Cost_Culti_acre']
Culti_cost = np.array(Culti_cost)
Type = df['Type_Code']
Type = np.array(Type)
Root_depth = df['Root_Depth']
Root_depth = np.array(Root_depth)
Current_month = datetime.datetime.now().month
Current_month_str = datetime.datetime.today().strftime('%B')
Max_=[]
Avg_=[]
Std_=[]
n_i=1
n_f=20
m=5
count=0

#--------------------------------------------------------------------------------------------------------

# Outputs list of harvest months
def Harvest_month(code_val):

	harvest_month=[]
	for i in range(12):
		if (i+1)*(Harvest_time[(code_val-1)*12]+1) <= 12: 
			crop_id = (code_val-1)*12 + Harvest_time[(code_val-1)*12] + (Current_month -1) + \
			i*(Harvest_time[(code_val-1)*12]+1)		#continuation
			crop_id_verify = Harvest_time[(code_val-1)*12] + (Current_month -1) + i*(Harvest_time[(code_val-1)*12]+1)
			if crop_id_verify < 12:
				crop_id = crop_id
				harvest_month.append(Month[crop_id])
				# break
			else :
				crop_id = (code_val-1)*12 + crop_id%12
				harvest_month.append(Month[crop_id])
				# break
		else:
			break
	return harvest_month

# Outputs list of planting months
def Planting_month(code_val):

	planting_month=[]
	for i in range(12):
		if (i+1)*(Harvest_time[(code_val-1)*12]+1) <= 12: 
			crop_id = (code_val-1)*12 + (Current_month -1) + i*(Harvest_time[(code_val-1)*12]+1)
			crop_id_verify = (Current_month -1) + i*(Harvest_time[(code_val-1)*12]+1)
			if crop_id_verify < 12:
				crop_id = crop_id
				planting_month.append(Month[crop_id])
				# break
			else :
				crop_id = (code_val-1)*12 + crop_id%12
				planting_month.append(Month[crop_id])
				# break
		else:
			break
	return planting_month

# Total profit of each individual 
# Objective fun : Sum of profits of 'm' crops
# Subject to constrains : [1] Harvest time.
#						  [2] Crop cycle in a year based on harvest time.
def Fitness_value(individual):

	global profit
	profit = []
	root_depth = []
	if len(set(individual))==m:
		for i in range(len(individual)):
			profit_itt = []
			# print(i)
			Crop = individual[i]
			# print(Crop)
			for e in range(len(Type)):
				if Type[e]==Crop:
					type_id = e
					# print(type_id)
					# print(Type[type_id])
					break
				else:
					pass

			for i in range(12):
				if (i+1)*(Harvest_time[type_id]+1) <= 12: 

					profit_id = type_id + Current_month + Harvest_time[type_id] -1 + i*(Harvest_time[type_id]+1)
					id_verify = Current_month + Harvest_time[type_id] -1 + i*(Harvest_time[type_id]+1)
					if id_verify < 12:
						profit_i = Profit[profit_id]
						# print(profit_id)
						# print(profit_i)
						profit_itt.append(profit_i)
						# break
					else:
						profit_i = Profit[type_id + profit_id%12]
						# print(type_id + profit_id%12)
						# print(profit_i)
						profit_itt.append(profit_i)
						# break
				else:
					break
			# print(profit_itt)
			profit.append(sum(profit_itt))
			# print(profit)
			root_depth.append(Root_depth[type_id])
		# print(root_depth)
	else:
		profit=[0]

	list_abc=[]
	counts = Counter(root_depth)
	per_s = counts['Shallow']*100/m
	per_m = counts['Medium']*100/m
	per_d = counts['Deep']*100/m
	
	if per_s and per_m != 0:
		a = abs(per_s - per_m)
		list_abc.append(a)
	if per_s and per_d != 0:
		b = abs(per_s - per_d)
		list_abc.append(b)
	if per_m and per_d != 0:
		c = abs(per_m - per_d)
		list_abc.append(c)
	# print(list_abc)
	if len(list_abc) != 0:
		avg_abc = sum(list_abc)/len(list_abc)
	else:
		avg_abc = 100
	# print(avg_abc)
	return sum(profit), avg_abc

# Creating class
creator.create('FitnessMax', base.Fitness, weights = (1.0,-1.0))
creator.create('Individual', list, fitness = creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('attr_value', random.randint, n_i, n_f)	#generator
# Structure initializers
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_value, m)	
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
# genetic operators required for the evolution
toolbox.register('evaluate', Fitness_value)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutUniformInt, low=n_i, up=n_f, indpb=0.2)
toolbox.register('select', tools.selTournament, tournsize=3)

#------------------------------------------ Evolution operation ----------------------------------------------

def main():

	# create an initial population of 300 individuals
	pop = toolbox.population(n=300)
	# CXPB  is the probability with which two individuals are crossed
	# MUTPB is the probability for mutating an individual
	# Number of generations/Number of itterations
	global NGen
	CXPB, MUTPB, NGen = 0.5, 0.2, 15

	print("Start of evolution")
	
	# Evaluate the entire population
	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
	
	print("  Evaluated %i individuals" % len(pop))

	# Extracting all the fitnesses of 
	fits = [ind.fitness.values[0] for ind in pop]

	# Begin the evolution
	for g in range(NGen):

		gen = g+1
		print("-- Generation %i --" % gen)
		
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
		
		print("  Evaluated %i individuals" % len(invalid_ind))
		
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
		
		print("  Min %s" % min(fits))
		print("  Max %s" % max(fits))
		print("  Avg %s" % mean)
		print("  Std %s" % std, '\n')
	
	print("-- End of successful evolution --")
	Best = tools.selBest(pop, 1)[0]
	print("Best individual is %s, %s" % (Best, Best.fitness.values))
	# print("Best individual is %s " % Best)	

	Fitness_value(Best)

	t = PrettyTable(['Crop','Planting Month', 'Harvest Month', 'Harvest Cycles', 'Root Sys', 'Culti Cost', 'Profit'])
	for i in range(len(Best)):
		val = Best[i]
		t.add_row([Crop_name[val*12-1], ', '.join(Planting_month(val)), ', '.join(Harvest_month(val)), \
		len(Harvest_month(val)), Root_depth[val*12-1], len(Harvest_month(val))*Culti_cost[val*12-1], profit[i]])
	print(t)
	print("Total Profit : %s " % Best.fitness.values[0])

if __name__ == "__main__":
	main()

#---------------------------------------------- Visualisation ------------------------------------------------

Max_ = np.array(Max_)
x_ = np.arange(1,len(Max_)+1)

plt.bar(x_-0.2, Max_, width = 0.2,align='center', label='Max')
plt.bar(x_, Avg_, width = 0.2,align='center', label='Avg')
plt.bar(x_+0.2, Std_, width = 0.2,align='center', label='Std')
plt.axis([0, NGen+1, 0, 1.4*max(Max_)])
plt.axes().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.xlabel('Generation')
plt.ylabel('Total Profit')
plt.title('Max - Avg - Std')
plt.legend()
# plt.show()