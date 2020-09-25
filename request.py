import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Dependents':1, 'ApplicantIncome':6065, 'CoapplicantIncome':2004.0,'LoanAmount':250,
'Loan_Amount_Term':360,'Credit_History':1.0,'Gender_Female':0,'Married_No':0,'Education_Graduate':1,'Self_Employed_No':1,0
'Property_Area_Rural':0})

#print(r.json())