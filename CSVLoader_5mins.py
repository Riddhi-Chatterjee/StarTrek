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
import threading

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
                ds.write(str(price))
        else:
            with open("datasets/5mins_"+stock_name+".csv", "a") as ds:
                if isStartOfDay:
                    ds.write("\n"+str(price))
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
    
    schedule.every().day.at(str(hr) + ":49").do(job, True, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":50").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":51").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":52").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":31").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":40").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":45").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":50").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":55").do(job, False, chromeDriverPath)
    schedule.every().day.at(str(hr) + ":00").do(job, False, chromeDriverPath)

chromeDriverPath = ChromeDriverManager().install()
for i in range(23,24):
    sched(i, chromeDriverPath)

while True:
    schedule.run_pending()
    time.sleep(1)