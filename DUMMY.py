print("Running infinite loop...")
i=0
while(True):
    i = (i + 1) % 100000000
    if i%1000 == 0:
        print(i)