print("Running infinite loop...")
i=200
while(True):
    i = (i + 1) % 100000000
    with open("DUMMY.txt", "a") as d:
        d.write("Current value of i: "+str(i)+"\n")
    break