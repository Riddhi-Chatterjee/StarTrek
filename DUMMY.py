print("Running infinite loop...")
i=0
while(True):
    i = (i + 1) % 100000000
    if i%1000000 == 0:
        with open("DUMMY.txt", "a") as d:
            d.write("Current value of i: "+str(i)+"\n")
    break