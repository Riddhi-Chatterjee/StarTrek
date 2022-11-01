from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import schedule
import time
def job():

    driver = webdriver.Chrome("./chromedriver")
    driver.get("https://www.google.com/search?q=apple+stock+price&oq=apple+stock+price&aqs=chrome..69i57j0i131i433i512j0i512l7j0i457i512.2575j1j7&sourceid=chrome&ie=UTF-8") 
    try:
        data = driver.find_element(By.XPATH, "//span[@class='knowledge-finance-wholepage-chart__hover-card-value']") 
        print(data.get_attribute('innerHTML'))
        
    except NoSuchElementException:
        print("Login Failed")
        exit()
   

def sched(hr):
    
    schedule.every().day.at(str(hr) + ":00").do(job)
    schedule.every().day.at(str(hr) + ":01").do(job)
    schedule.every().day.at(str(hr) + ":02").do(job)
    schedule.every().day.at(str(hr) + ":03").do(job)
    schedule.every().day.at(str(hr) + ":04").do(job)
    schedule.every().day.at(str(hr) + ":40").do(job)
    schedule.every().day.at(str(hr) + ":45").do(job)
    schedule.every().day.at(str(hr) + ":50").do(job)
    schedule.every().day.at(str(hr) + ":55").do(job)
    schedule.every().day.at(str(hr) + ":00").do(job)

for i in range(18,20):
    sched(i)

while True:
    schedule.run_pending()
    time.sleep(1)