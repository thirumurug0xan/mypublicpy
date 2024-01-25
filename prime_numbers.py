try:
    rang = int(input('enter the range default is 100 :'))
except:
    rang = None
if not rang:
    rang = 100

for i in range(0,rang+1):
    if i > 1:
        for j in range(2,i):
            if (i % j) == 0:
                break
        else:
            print(i)

