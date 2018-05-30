i=0
x=1.61803
while True:
	x=1+1/x
	print('--', i, '--')
	print(x)
	i+=1
	if i == 100:
		break
	else:
		pass