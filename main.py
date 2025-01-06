from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, List
import math
import pickle
import numpy as np

app = FastAPI()
data_transformers = None
lgb_classifier = None

def import_variables(file_path):
    with open(file_path, 'rb') as f:
        variables = pickle.load(f)
    print(f"Variables loaded from {file_path}")
    return variables


@app.on_event("startup")
async def load_variables():
    global data_transformers, lgb_classifier
    try:
        loaded_vars = import_variables("variables.pkl")
        data_transformers = loaded_vars.get("data_transformers")
        lgb_classifier = loaded_vars.get("lgb_classifier")
        print("Variables was imported")
    except FileNotFoundError:
        print("File not found.")


class PredictionResponse(BaseModel):
    predicted_class: int  # Predicted class (e.g., 0 or 1)
    probability: List[float]  # List of probabilities for each class
    

class InsuranceData(BaseModel):
    Agenturname: Literal['CBH', 'CWT', 'JZI', 'KML', 'EPX', 'C2B', 'JWT', 'RAB', 'SSI',
                         'ART', 'CSR', 'CCR', 'ADM', 'LWC', 'TTW', 'TST'] = Field(..., description="Name of the agency")
    Agenturtyp: Literal['Travel Agency', 'Airlines'] = Field(..., description="Type of the agency")
    Vertriebskanal: Literal['Online', 'Offline'] = Field(..., description="Sales channel")
    Produktname: Literal['Comprehensive Plan', 'Rental Vehicle Excess Insurance',
                         'Value Plan', 'Basic Plan', 'Premier Plan',
                         '2 way Comprehensive Plan', 'Bronze Plan', 'Silver Plan',
                         'Annual Silver Plan', 'Cancellation Plan',
                         '1 way Comprehensive Plan', 'Ticket Protector', '24 Protect',
                         'Gold Plan', 'Annual Gold Plan',
                         'Single Trip Travel Protect Silver',
                         'Individual Comprehensive Plan',
                         'Spouse or Parents Comprehensive Plan',
                         'Annual Travel Protect Silver',
                         'Single Trip Travel Protect Platinum',
                         'Annual Travel Protect Gold', 'Single Trip Travel Protect Gold',
                         'Annual Travel Protect Platinum', 'Child Comprehensive Plan',
                         'Travel Cruise Protect', 'Travel Cruise Protect Family'] = Field(..., description="Name of the product")
    Reisedauer: int = Field(..., ge=0, le=450, description="Duration of the trip in days")
    Reiseziel: Literal['MALAYSIA', 'AUSTRALIA', 'ITALY', 'UNITED STATES', 'THAILAND',
                       "'KOREA DEMOCRATIC PEOPLE'S REPUBLIC OF'", 'NORWAY', 'VIET NAM',
                       'DENMARK', 'SINGAPORE', 'JAPAN', 'UNITED KINGDOM', 'INDONESIA',
                       'INDIA', 'CHINA', 'FRANCE', "'TAIWAN PROVINCE OF CHINA'",
                       'PHILIPPINES', 'MYANMAR', 'HONG KONG', "'KOREA REPUBLIC OF'",
                       'UNITED ARAB EMIRATES', 'NAMIBIA', 'NEW ZEALAND', 'COSTA RICA',
                       'BRUNEI DARUSSALAM', 'POLAND', 'SPAIN', 'CZECH REPUBLIC',
                       'GERMANY', 'SRI LANKA', 'CAMBODIA', 'AUSTRIA', 'SOUTH AFRICA',
                       "'TANZANIA UNITED REPUBLIC OF'",
                       "LAO PEOPLE'S DEMOCRATIC REPUBLIC", 'NEPAL', 'NETHERLANDS',
                       'MACAO', 'CROATIA', 'FINLAND', 'CANADA', 'TUNISIA',
                       'RUSSIAN FEDERATION', 'GREECE', 'BELGIUM', 'IRELAND',
                       'SWITZERLAND', 'CHILE', 'ISRAEL', 'BANGLADESH', 'ICELAND',
                       'PORTUGAL', 'ROMANIA', 'KENYA', 'GEORGIA', 'TURKEY', 'SWEDEN',
                       'MALDIVES', 'ESTONIA', 'SAUDI ARABIA', 'PAKISTAN', 'QATAR', 'PERU',
                       'LUXEMBOURG', 'MONGOLIA', 'ARGENTINA', 'CYPRUS', 'FIJI',
                       'BARBADOS', 'TRINIDAD AND TOBAGO', 'ETHIOPIA', 'PAPUA NEW GUINEA',
                       'SERBIA', 'JORDAN', 'ECUADOR', 'BENIN', 'OMAN', 'BAHRAIN',
                       'UGANDA', 'BRAZIL', 'MEXICO', 'HUNGARY', 'AZERBAIJAN', 'MOROCCO',
                       'URUGUAY', 'MAURITIUS', 'JAMAICA', 'KAZAKHSTAN', 'GHANA',
                       'UZBEKISTAN', 'SLOVENIA', 'KUWAIT', 'GUAM', 'BULGARIA',
                       'LITHUANIA', 'NEW CALEDONIA', 'EGYPT', 'ARMENIA', 'BOLIVIA',
                       "'VIRGIN ISLANDS U.S.'", 'PANAMA', 'SIERRA LEONE', 'COLOMBIA',
                       'PUERTO RICO', 'UKRAINE', 'GUINEA', 'GUADELOUPE',
                       "'MOLDOVA REPUBLIC OF'", 'GUYANA', 'LATVIA', 'ZIMBABWE', 'VANUATU',
                       'VENEZUELA', 'BOTSWANA', 'BERMUDA', 'MALI', 'KYRGYZSTAN',
                       'CAYMAN ISLANDS', 'MALTA', 'LEBANON', 'REUNION', 'SEYCHELLES',
                       'ZAMBIA', 'SAMOA', 'NORTHERN MARIANA ISLANDS', 'NIGERIA',
                       'DOMINICAN REPUBLIC', 'TAJIKISTAN', 'ALBANIA',
                       "'MACEDONIA THE FORMER YUGOSLAV REPUBLIC OF'F'",
                       'LIBYAN ARAB JAMAHIRIYA', 'ANGOLA', 'BELARUS',
                       'TURKS AND CAICOS ISLANDS', 'FAROE ISLANDS', 'TURKMENISTAN',
                       'GUINEA-BISSAU', 'CAMEROON', 'BHUTAN', 'RWANDA', 'SOLOMON ISLANDS',
                       "'IRAN ISLAMIC REPUBLIC OF'", 'GUATEMALA', 'FRENCH POLYNESIA',
                       'TIBET', 'SENEGAL', 'REPUBLIC OF MONTENEGRO',
                       'BOSNIA AND HERZEGOVINA'] = Field(..., description="Destination of the trip")
    Nettoumsatz: int = Field(..., ge=0, le=810, description="Net revenue in EUR")
    Kommission: float = Field(..., ge=0, le=283.5, description="Commission in EUR")
    Geschlecht: Literal['M', 'F'] | None = Field(None, description="Gender of the insured person")
    Alter: int = Field(..., ge=0, le=118, description="Age of the insured person")

def transform_data(data: InsuranceData):	
    global data_transformers
    transformed = dict()

    data = dict(data)
    
    transformed['ReisedauerLog'] = math.log(1 + data['Reisedauer'])
    transformed['ReisedauerSqrt'] = math.sqrt(data['Reisedauer'])
    transformed['ReisedauerInv'] = 1 / data['Reisedauer']
    transformed['LangeReise'] = 1 if data['Reisedauer'] > 363 else 0
    
    transformed['NettoumsatzLt0'] = 0
    
    for field in ['Alter', 'Nettoumsatz', 'Kommission', 'Reisedauer']:
        transformed[field] = data[field]

    for field in ['Agenturtyp', 'Vertriebskanal']:
        transformed[field] = data_transformers[field][data[field]]
    
    transformed['Mann'] = 1 if data['Geschlecht'] == 'M' else 0
    transformed['Frau'] = 1 if data['Geschlecht'] == 'F' else 0
    
    transformed['KommissionNettoumsatzRatio'] = data_transformers['KommissionNettoumsatzRatio'][data['Produktname']]
    
    field = 'RueckerstattungsprozentsatzNachAgenturname'
    transformed[field] = data_transformers[field][data['Agenturname']]

    field = 'GewichtetReisezielLeistungseintrittRatio'
    transformed[field] = data_transformers[field][data['Reiseziel']]
    field = 'ReisezielLeistungseintrittRatio'
    transformed[field] = data_transformers[field][data['Reiseziel']]

    feature_order = ['Agenturtyp', 'Vertriebskanal', 'Reisedauer',
                       'Nettoumsatz', 'Kommission', 'Alter', 'ReisedauerLog', 'ReisedauerSqrt',
                       'ReisedauerInv', 'LangeReise', 'NettoumsatzLt0',
                       'KommissionNettoumsatzRatio', 'Mann', 'Frau',
                       'RueckerstattungsprozentsatzNachAgenturname',
                       'ReisezielLeistungseintrittRatio',
                       'GewichtetReisezielLeistungseintrittRatio']
    
    model_input = np.array([[transformed[feature] for feature in feature_order]])
    return model_input


@app.post("/predict/", response_model=PredictionResponse)
async def predict(data: InsuranceData):
    """
    Endpoint to process insurance data. Validates and processes the input.
    """
    global lgb_classifier, data_transformers

    if lgb_classifier is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Ensure the startup event executed properly.")

    X = transform_data(data)

    prediction = lgb_classifier.predict(X)
    prediction_proba = lgb_classifier.predict_proba(X)
    
    return PredictionResponse(
            predicted_class=int(prediction[0]),
            probability=[float(p) for p in prediction_proba[0]]
    )