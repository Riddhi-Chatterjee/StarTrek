from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import schedule
import time
from os.path import exists
from datetime import datetime
import threading

def extractDate(dt):
    return str(dt)[:10]

def loadPrice(stock_name, isStartOfDay, chromeDriverPath):
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(service=Service(chromeDriverPath), options=options)
    
    stock_url = "https://www.google.com/search?q="+stock_name.replace(" ", "+")+"+stock+price"
    driver.get(stock_url) 
    try:
        data = driver.find_element(By.XPATH, "//span[@class='knowledge-finance-wholepage-chart__hover-card-value']") 
        price = data.get_attribute('innerHTML')
        price = str(price).split(' ')[0]
        price = price.replace(",", "")
        price = price.replace("USD", "")
        if not exists("datasets/5mins_"+stock_name+".csv"):
            with open("datasets/5mins_"+stock_name+".csv", "a") as ds:
                string = "Date, 09:15, 09:30, 09:45, 09:50, 09:55, "
                temp = ""
                for i in range(10, 15):
                    temp = temp + str(i) + ":00, " + str(i) + ":05, " + str(i) + ":10, " + str(i) + ":15, " + str(i) + ":20, " + str(i) + ":25, " + str(i) + ":30, " + str(i) + ":35, " + str(i) + ":40, " + str(i) + ":45, " + str(i) + ":50, " + str(i) + ":55, "

                string = string + temp + "15:00, " + "15:05, " + "15:10, " + "15:15, " + "15:20, " + "15:25, " + "15:30"
                ds.write(string + "\n")
                ds.write(str(extractDate(datetime.today())) + ", " + str(price))
        else:
            with open("datasets/5mins_"+stock_name+".csv", "a") as ds:
                if isStartOfDay:
                    ds.write("\n"+str(extractDate(datetime.today())) + str(price))
                else:
                    ds.write(", "+str(price))
        
    except NoSuchElementException:
        price = 0.0
        with open("datasets/5mins_"+stock_name+".csv", "r") as ds:
            line1 = ds.readline()
            line2 = ds.readline()
            for line in ds:
                line1 = line2
                line2 = line
            price = line1.split(", ")[len(line2.split(", "))]
        with open("datasets/5mins_"+stock_name+".csv", "a") as ds:
            if isStartOfDay:
                ds.write("\n"+str(price))
            else:
                ds.write(", "+str(price))
            
def job(isStartOfDay, chromeDriverPath):
    stock_list = ['larsen and toubro', 'tata steel', 'paytm']
    threads = []
    
    for stock_name in stock_list:
        threads.append(threading.Thread(target=loadPrice, args=(stock_name, isStartOfDay,chromeDriverPath,)))
    
    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()

def sched(hr, chromeDriverPath):
    
    schedule.every().day.at(str(hr) + ":37").do(job, True, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":38").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":39").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":40").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":41").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":42").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":43").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":44").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":45").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":46").do(job, False, chromeDriverPath)

chromeDriverPath = ChromeDriverManager().install()
for i in range(17,24):
    sched(i, chromeDriverPath)

while True:
    schedule.run_pending()
    time.sleep(1)