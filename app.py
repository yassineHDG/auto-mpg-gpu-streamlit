"""Auto MPG — Analyse, comparaison et prédiction avec accélération CUDA.

Lancer : python -m streamlit run app.py
Dépendances de base : streamlit, pandas, numpy, plotly, scikit-learn.
Optionnel GPU : cuML + CuPy, ou XGBoost >= 2 + CuPy.
Les 398 observations sont intégrées ci-dessous : aucun téléchargement au lancement.
"""
from __future__ import annotations

import hashlib
from io import StringIO
import json
import platform
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

SEED = 42
MPG_TO_L100 = 235.215
CO2_FACTOR = 23.5  # Hypothèse demandée : g/km par L/100 km (≈ essence).
NUMERIC_FEATURES = [
    "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"
]
FEATURES = NUMERIC_FEATURES + ["origin"]
ORIGINS = ["usa", "japan", "europe"]
PALETTE = {"usa": "#4F8EF7", "japan": "#18B68D", "europe": "#EFAC46"}
LABELS = {
    "name": "Modèle", "brand": "Marque", "mpg": "Efficacité (MPG US)",
    "l_100km": "Consommation (L/100 km)", "co2_g_km": "CO₂ estimé (g/km)",
    "horsepower": "Puissance (hp)", "weight": "Poids (lb)",
    "displacement": "Cylindrée (in³)", "cylinders": "Cylindres",
    "acceleration": "0–60 mph (s)", "model_year": "Année (code)",
    "year": "Année", "origin": "Origine", "horsepower_imputed": "Puissance imputée",
}

CSV_DATA = """mpg,cylinders,displacement,horsepower,weight,acceleration,model_year,origin,name
18.0,8,307.0,130.0,3504,12.0,70,usa,chevrolet chevelle malibu
15.0,8,350.0,165.0,3693,11.5,70,usa,buick skylark 320
18.0,8,318.0,150.0,3436,11.0,70,usa,plymouth satellite
16.0,8,304.0,150.0,3433,12.0,70,usa,amc rebel sst
17.0,8,302.0,140.0,3449,10.5,70,usa,ford torino
15.0,8,429.0,198.0,4341,10.0,70,usa,ford galaxie 500
14.0,8,454.0,220.0,4354,9.0,70,usa,chevrolet impala
14.0,8,440.0,215.0,4312,8.5,70,usa,plymouth fury iii
14.0,8,455.0,225.0,4425,10.0,70,usa,pontiac catalina
15.0,8,390.0,190.0,3850,8.5,70,usa,amc ambassador dpl
15.0,8,383.0,170.0,3563,10.0,70,usa,dodge challenger se
14.0,8,340.0,160.0,3609,8.0,70,usa,plymouth 'cuda 340
15.0,8,400.0,150.0,3761,9.5,70,usa,chevrolet monte carlo
14.0,8,455.0,225.0,3086,10.0,70,usa,buick estate wagon (sw)
24.0,4,113.0,95.0,2372,15.0,70,japan,toyota corona mark ii
22.0,6,198.0,95.0,2833,15.5,70,usa,plymouth duster
18.0,6,199.0,97.0,2774,15.5,70,usa,amc hornet
21.0,6,200.0,85.0,2587,16.0,70,usa,ford maverick
27.0,4,97.0,88.0,2130,14.5,70,japan,datsun pl510
26.0,4,97.0,46.0,1835,20.5,70,europe,volkswagen 1131 deluxe sedan
25.0,4,110.0,87.0,2672,17.5,70,europe,peugeot 504
24.0,4,107.0,90.0,2430,14.5,70,europe,audi 100 ls
25.0,4,104.0,95.0,2375,17.5,70,europe,saab 99e
26.0,4,121.0,113.0,2234,12.5,70,europe,bmw 2002
21.0,6,199.0,90.0,2648,15.0,70,usa,amc gremlin
10.0,8,360.0,215.0,4615,14.0,70,usa,ford f250
10.0,8,307.0,200.0,4376,15.0,70,usa,chevy c20
11.0,8,318.0,210.0,4382,13.5,70,usa,dodge d200
9.0,8,304.0,193.0,4732,18.5,70,usa,hi 1200d
27.0,4,97.0,88.0,2130,14.5,71,japan,datsun pl510
28.0,4,140.0,90.0,2264,15.5,71,usa,chevrolet vega 2300
25.0,4,113.0,95.0,2228,14.0,71,japan,toyota corona
25.0,4,98.0,,2046,19.0,71,usa,ford pinto
19.0,6,232.0,100.0,2634,13.0,71,usa,amc gremlin
16.0,6,225.0,105.0,3439,15.5,71,usa,plymouth satellite custom
17.0,6,250.0,100.0,3329,15.5,71,usa,chevrolet chevelle malibu
19.0,6,250.0,88.0,3302,15.5,71,usa,ford torino 500
18.0,6,232.0,100.0,3288,15.5,71,usa,amc matador
14.0,8,350.0,165.0,4209,12.0,71,usa,chevrolet impala
14.0,8,400.0,175.0,4464,11.5,71,usa,pontiac catalina brougham
14.0,8,351.0,153.0,4154,13.5,71,usa,ford galaxie 500
14.0,8,318.0,150.0,4096,13.0,71,usa,plymouth fury iii
12.0,8,383.0,180.0,4955,11.5,71,usa,dodge monaco (sw)
13.0,8,400.0,170.0,4746,12.0,71,usa,ford country squire (sw)
13.0,8,400.0,175.0,5140,12.0,71,usa,pontiac safari (sw)
18.0,6,258.0,110.0,2962,13.5,71,usa,amc hornet sportabout (sw)
22.0,4,140.0,72.0,2408,19.0,71,usa,chevrolet vega (sw)
19.0,6,250.0,100.0,3282,15.0,71,usa,pontiac firebird
18.0,6,250.0,88.0,3139,14.5,71,usa,ford mustang
23.0,4,122.0,86.0,2220,14.0,71,usa,mercury capri 2000
28.0,4,116.0,90.0,2123,14.0,71,europe,opel 1900
30.0,4,79.0,70.0,2074,19.5,71,europe,peugeot 304
30.0,4,88.0,76.0,2065,14.5,71,europe,fiat 124b
31.0,4,71.0,65.0,1773,19.0,71,japan,toyota corolla 1200
35.0,4,72.0,69.0,1613,18.0,71,japan,datsun 1200
27.0,4,97.0,60.0,1834,19.0,71,europe,volkswagen model 111
26.0,4,91.0,70.0,1955,20.5,71,usa,plymouth cricket
24.0,4,113.0,95.0,2278,15.5,72,japan,toyota corona hardtop
25.0,4,97.5,80.0,2126,17.0,72,usa,dodge colt hardtop
23.0,4,97.0,54.0,2254,23.5,72,europe,volkswagen type 3
20.0,4,140.0,90.0,2408,19.5,72,usa,chevrolet vega
21.0,4,122.0,86.0,2226,16.5,72,usa,ford pinto runabout
13.0,8,350.0,165.0,4274,12.0,72,usa,chevrolet impala
14.0,8,400.0,175.0,4385,12.0,72,usa,pontiac catalina
15.0,8,318.0,150.0,4135,13.5,72,usa,plymouth fury iii
14.0,8,351.0,153.0,4129,13.0,72,usa,ford galaxie 500
17.0,8,304.0,150.0,3672,11.5,72,usa,amc ambassador sst
11.0,8,429.0,208.0,4633,11.0,72,usa,mercury marquis
13.0,8,350.0,155.0,4502,13.5,72,usa,buick lesabre custom
12.0,8,350.0,160.0,4456,13.5,72,usa,oldsmobile delta 88 royale
13.0,8,400.0,190.0,4422,12.5,72,usa,chrysler newport royal
19.0,3,70.0,97.0,2330,13.5,72,japan,mazda rx2 coupe
15.0,8,304.0,150.0,3892,12.5,72,usa,amc matador (sw)
13.0,8,307.0,130.0,4098,14.0,72,usa,chevrolet chevelle concours (sw)
13.0,8,302.0,140.0,4294,16.0,72,usa,ford gran torino (sw)
14.0,8,318.0,150.0,4077,14.0,72,usa,plymouth satellite custom (sw)
18.0,4,121.0,112.0,2933,14.5,72,europe,volvo 145e (sw)
22.0,4,121.0,76.0,2511,18.0,72,europe,volkswagen 411 (sw)
21.0,4,120.0,87.0,2979,19.5,72,europe,peugeot 504 (sw)
26.0,4,96.0,69.0,2189,18.0,72,europe,renault 12 (sw)
22.0,4,122.0,86.0,2395,16.0,72,usa,ford pinto (sw)
28.0,4,97.0,92.0,2288,17.0,72,japan,datsun 510 (sw)
23.0,4,120.0,97.0,2506,14.5,72,japan,toyouta corona mark ii (sw)
28.0,4,98.0,80.0,2164,15.0,72,usa,dodge colt (sw)
27.0,4,97.0,88.0,2100,16.5,72,japan,toyota corolla 1600 (sw)
13.0,8,350.0,175.0,4100,13.0,73,usa,buick century 350
14.0,8,304.0,150.0,3672,11.5,73,usa,amc matador
13.0,8,350.0,145.0,3988,13.0,73,usa,chevrolet malibu
14.0,8,302.0,137.0,4042,14.5,73,usa,ford gran torino
15.0,8,318.0,150.0,3777,12.5,73,usa,dodge coronet custom
12.0,8,429.0,198.0,4952,11.5,73,usa,mercury marquis brougham
13.0,8,400.0,150.0,4464,12.0,73,usa,chevrolet caprice classic
13.0,8,351.0,158.0,4363,13.0,73,usa,ford ltd
14.0,8,318.0,150.0,4237,14.5,73,usa,plymouth fury gran sedan
13.0,8,440.0,215.0,4735,11.0,73,usa,chrysler new yorker brougham
12.0,8,455.0,225.0,4951,11.0,73,usa,buick electra 225 custom
13.0,8,360.0,175.0,3821,11.0,73,usa,amc ambassador brougham
18.0,6,225.0,105.0,3121,16.5,73,usa,plymouth valiant
16.0,6,250.0,100.0,3278,18.0,73,usa,chevrolet nova custom
18.0,6,232.0,100.0,2945,16.0,73,usa,amc hornet
18.0,6,250.0,88.0,3021,16.5,73,usa,ford maverick
23.0,6,198.0,95.0,2904,16.0,73,usa,plymouth duster
26.0,4,97.0,46.0,1950,21.0,73,europe,volkswagen super beetle
11.0,8,400.0,150.0,4997,14.0,73,usa,chevrolet impala
12.0,8,400.0,167.0,4906,12.5,73,usa,ford country
13.0,8,360.0,170.0,4654,13.0,73,usa,plymouth custom suburb
12.0,8,350.0,180.0,4499,12.5,73,usa,oldsmobile vista cruiser
18.0,6,232.0,100.0,2789,15.0,73,usa,amc gremlin
20.0,4,97.0,88.0,2279,19.0,73,japan,toyota carina
21.0,4,140.0,72.0,2401,19.5,73,usa,chevrolet vega
22.0,4,108.0,94.0,2379,16.5,73,japan,datsun 610
18.0,3,70.0,90.0,2124,13.5,73,japan,maxda rx3
19.0,4,122.0,85.0,2310,18.5,73,usa,ford pinto
21.0,6,155.0,107.0,2472,14.0,73,usa,mercury capri v6
26.0,4,98.0,90.0,2265,15.5,73,europe,fiat 124 sport coupe
15.0,8,350.0,145.0,4082,13.0,73,usa,chevrolet monte carlo s
16.0,8,400.0,230.0,4278,9.5,73,usa,pontiac grand prix
29.0,4,68.0,49.0,1867,19.5,73,europe,fiat 128
24.0,4,116.0,75.0,2158,15.5,73,europe,opel manta
20.0,4,114.0,91.0,2582,14.0,73,europe,audi 100ls
19.0,4,121.0,112.0,2868,15.5,73,europe,volvo 144ea
15.0,8,318.0,150.0,3399,11.0,73,usa,dodge dart custom
24.0,4,121.0,110.0,2660,14.0,73,europe,saab 99le
20.0,6,156.0,122.0,2807,13.5,73,japan,toyota mark ii
11.0,8,350.0,180.0,3664,11.0,73,usa,oldsmobile omega
20.0,6,198.0,95.0,3102,16.5,74,usa,plymouth duster
21.0,6,200.0,,2875,17.0,74,usa,ford maverick
19.0,6,232.0,100.0,2901,16.0,74,usa,amc hornet
15.0,6,250.0,100.0,3336,17.0,74,usa,chevrolet nova
31.0,4,79.0,67.0,1950,19.0,74,japan,datsun b210
26.0,4,122.0,80.0,2451,16.5,74,usa,ford pinto
32.0,4,71.0,65.0,1836,21.0,74,japan,toyota corolla 1200
25.0,4,140.0,75.0,2542,17.0,74,usa,chevrolet vega
16.0,6,250.0,100.0,3781,17.0,74,usa,chevrolet chevelle malibu classic
16.0,6,258.0,110.0,3632,18.0,74,usa,amc matador
18.0,6,225.0,105.0,3613,16.5,74,usa,plymouth satellite sebring
16.0,8,302.0,140.0,4141,14.0,74,usa,ford gran torino
13.0,8,350.0,150.0,4699,14.5,74,usa,buick century luxus (sw)
14.0,8,318.0,150.0,4457,13.5,74,usa,dodge coronet custom (sw)
14.0,8,302.0,140.0,4638,16.0,74,usa,ford gran torino (sw)
14.0,8,304.0,150.0,4257,15.5,74,usa,amc matador (sw)
29.0,4,98.0,83.0,2219,16.5,74,europe,audi fox
26.0,4,79.0,67.0,1963,15.5,74,europe,volkswagen dasher
26.0,4,97.0,78.0,2300,14.5,74,europe,opel manta
31.0,4,76.0,52.0,1649,16.5,74,japan,toyota corona
32.0,4,83.0,61.0,2003,19.0,74,japan,datsun 710
28.0,4,90.0,75.0,2125,14.5,74,usa,dodge colt
24.0,4,90.0,75.0,2108,15.5,74,europe,fiat 128
26.0,4,116.0,75.0,2246,14.0,74,europe,fiat 124 tc
24.0,4,120.0,97.0,2489,15.0,74,japan,honda civic
26.0,4,108.0,93.0,2391,15.5,74,japan,subaru
31.0,4,79.0,67.0,2000,16.0,74,europe,fiat x1.9
19.0,6,225.0,95.0,3264,16.0,75,usa,plymouth valiant custom
18.0,6,250.0,105.0,3459,16.0,75,usa,chevrolet nova
15.0,6,250.0,72.0,3432,21.0,75,usa,mercury monarch
15.0,6,250.0,72.0,3158,19.5,75,usa,ford maverick
16.0,8,400.0,170.0,4668,11.5,75,usa,pontiac catalina
15.0,8,350.0,145.0,4440,14.0,75,usa,chevrolet bel air
16.0,8,318.0,150.0,4498,14.5,75,usa,plymouth grand fury
14.0,8,351.0,148.0,4657,13.5,75,usa,ford ltd
17.0,6,231.0,110.0,3907,21.0,75,usa,buick century
16.0,6,250.0,105.0,3897,18.5,75,usa,chevroelt chevelle malibu
15.0,6,258.0,110.0,3730,19.0,75,usa,amc matador
18.0,6,225.0,95.0,3785,19.0,75,usa,plymouth fury
21.0,6,231.0,110.0,3039,15.0,75,usa,buick skyhawk
20.0,8,262.0,110.0,3221,13.5,75,usa,chevrolet monza 2+2
13.0,8,302.0,129.0,3169,12.0,75,usa,ford mustang ii
29.0,4,97.0,75.0,2171,16.0,75,japan,toyota corolla
23.0,4,140.0,83.0,2639,17.0,75,usa,ford pinto
20.0,6,232.0,100.0,2914,16.0,75,usa,amc gremlin
23.0,4,140.0,78.0,2592,18.5,75,usa,pontiac astro
24.0,4,134.0,96.0,2702,13.5,75,japan,toyota corona
25.0,4,90.0,71.0,2223,16.5,75,europe,volkswagen dasher
24.0,4,119.0,97.0,2545,17.0,75,japan,datsun 710
18.0,6,171.0,97.0,2984,14.5,75,usa,ford pinto
29.0,4,90.0,70.0,1937,14.0,75,europe,volkswagen rabbit
19.0,6,232.0,90.0,3211,17.0,75,usa,amc pacer
23.0,4,115.0,95.0,2694,15.0,75,europe,audi 100ls
23.0,4,120.0,88.0,2957,17.0,75,europe,peugeot 504
22.0,4,121.0,98.0,2945,14.5,75,europe,volvo 244dl
25.0,4,121.0,115.0,2671,13.5,75,europe,saab 99le
33.0,4,91.0,53.0,1795,17.5,75,japan,honda civic cvcc
28.0,4,107.0,86.0,2464,15.5,76,europe,fiat 131
25.0,4,116.0,81.0,2220,16.9,76,europe,opel 1900
25.0,4,140.0,92.0,2572,14.9,76,usa,capri ii
26.0,4,98.0,79.0,2255,17.7,76,usa,dodge colt
27.0,4,101.0,83.0,2202,15.3,76,europe,renault 12tl
17.5,8,305.0,140.0,4215,13.0,76,usa,chevrolet chevelle malibu classic
16.0,8,318.0,150.0,4190,13.0,76,usa,dodge coronet brougham
15.5,8,304.0,120.0,3962,13.9,76,usa,amc matador
14.5,8,351.0,152.0,4215,12.8,76,usa,ford gran torino
22.0,6,225.0,100.0,3233,15.4,76,usa,plymouth valiant
22.0,6,250.0,105.0,3353,14.5,76,usa,chevrolet nova
24.0,6,200.0,81.0,3012,17.6,76,usa,ford maverick
22.5,6,232.0,90.0,3085,17.6,76,usa,amc hornet
29.0,4,85.0,52.0,2035,22.2,76,usa,chevrolet chevette
24.5,4,98.0,60.0,2164,22.1,76,usa,chevrolet woody
29.0,4,90.0,70.0,1937,14.2,76,europe,vw rabbit
33.0,4,91.0,53.0,1795,17.4,76,japan,honda civic
20.0,6,225.0,100.0,3651,17.7,76,usa,dodge aspen se
18.0,6,250.0,78.0,3574,21.0,76,usa,ford granada ghia
18.5,6,250.0,110.0,3645,16.2,76,usa,pontiac ventura sj
17.5,6,258.0,95.0,3193,17.8,76,usa,amc pacer d/l
29.5,4,97.0,71.0,1825,12.2,76,europe,volkswagen rabbit
32.0,4,85.0,70.0,1990,17.0,76,japan,datsun b-210
28.0,4,97.0,75.0,2155,16.4,76,japan,toyota corolla
26.5,4,140.0,72.0,2565,13.6,76,usa,ford pinto
20.0,4,130.0,102.0,3150,15.7,76,europe,volvo 245
13.0,8,318.0,150.0,3940,13.2,76,usa,plymouth volare premier v8
19.0,4,120.0,88.0,3270,21.9,76,europe,peugeot 504
19.0,6,156.0,108.0,2930,15.5,76,japan,toyota mark ii
16.5,6,168.0,120.0,3820,16.7,76,europe,mercedes-benz 280s
16.5,8,350.0,180.0,4380,12.1,76,usa,cadillac seville
13.0,8,350.0,145.0,4055,12.0,76,usa,chevy c10
13.0,8,302.0,130.0,3870,15.0,76,usa,ford f108
13.0,8,318.0,150.0,3755,14.0,76,usa,dodge d100
31.5,4,98.0,68.0,2045,18.5,77,japan,honda accord cvcc
30.0,4,111.0,80.0,2155,14.8,77,usa,buick opel isuzu deluxe
36.0,4,79.0,58.0,1825,18.6,77,europe,renault 5 gtl
25.5,4,122.0,96.0,2300,15.5,77,usa,plymouth arrow gs
33.5,4,85.0,70.0,1945,16.8,77,japan,datsun f-10 hatchback
17.5,8,305.0,145.0,3880,12.5,77,usa,chevrolet caprice classic
17.0,8,260.0,110.0,4060,19.0,77,usa,oldsmobile cutlass supreme
15.5,8,318.0,145.0,4140,13.7,77,usa,dodge monaco brougham
15.0,8,302.0,130.0,4295,14.9,77,usa,mercury cougar brougham
17.5,6,250.0,110.0,3520,16.4,77,usa,chevrolet concours
20.5,6,231.0,105.0,3425,16.9,77,usa,buick skylark
19.0,6,225.0,100.0,3630,17.7,77,usa,plymouth volare custom
18.5,6,250.0,98.0,3525,19.0,77,usa,ford granada
16.0,8,400.0,180.0,4220,11.1,77,usa,pontiac grand prix lj
15.5,8,350.0,170.0,4165,11.4,77,usa,chevrolet monte carlo landau
15.5,8,400.0,190.0,4325,12.2,77,usa,chrysler cordoba
16.0,8,351.0,149.0,4335,14.5,77,usa,ford thunderbird
29.0,4,97.0,78.0,1940,14.5,77,europe,volkswagen rabbit custom
24.5,4,151.0,88.0,2740,16.0,77,usa,pontiac sunbird coupe
26.0,4,97.0,75.0,2265,18.2,77,japan,toyota corolla liftback
25.5,4,140.0,89.0,2755,15.8,77,usa,ford mustang ii 2+2
30.5,4,98.0,63.0,2051,17.0,77,usa,chevrolet chevette
33.5,4,98.0,83.0,2075,15.9,77,usa,dodge colt m/m
30.0,4,97.0,67.0,1985,16.4,77,japan,subaru dl
30.5,4,97.0,78.0,2190,14.1,77,europe,volkswagen dasher
22.0,6,146.0,97.0,2815,14.5,77,japan,datsun 810
21.5,4,121.0,110.0,2600,12.8,77,europe,bmw 320i
21.5,3,80.0,110.0,2720,13.5,77,japan,mazda rx-4
43.1,4,90.0,48.0,1985,21.5,78,europe,volkswagen rabbit custom diesel
36.1,4,98.0,66.0,1800,14.4,78,usa,ford fiesta
32.8,4,78.0,52.0,1985,19.4,78,japan,mazda glc deluxe
39.4,4,85.0,70.0,2070,18.6,78,japan,datsun b210 gx
36.1,4,91.0,60.0,1800,16.4,78,japan,honda civic cvcc
19.9,8,260.0,110.0,3365,15.5,78,usa,oldsmobile cutlass salon brougham
19.4,8,318.0,140.0,3735,13.2,78,usa,dodge diplomat
20.2,8,302.0,139.0,3570,12.8,78,usa,mercury monarch ghia
19.2,6,231.0,105.0,3535,19.2,78,usa,pontiac phoenix lj
20.5,6,200.0,95.0,3155,18.2,78,usa,chevrolet malibu
20.2,6,200.0,85.0,2965,15.8,78,usa,ford fairmont (auto)
25.1,4,140.0,88.0,2720,15.4,78,usa,ford fairmont (man)
20.5,6,225.0,100.0,3430,17.2,78,usa,plymouth volare
19.4,6,232.0,90.0,3210,17.2,78,usa,amc concord
20.6,6,231.0,105.0,3380,15.8,78,usa,buick century special
20.8,6,200.0,85.0,3070,16.7,78,usa,mercury zephyr
18.6,6,225.0,110.0,3620,18.7,78,usa,dodge aspen
18.1,6,258.0,120.0,3410,15.1,78,usa,amc concord d/l
19.2,8,305.0,145.0,3425,13.2,78,usa,chevrolet monte carlo landau
17.7,6,231.0,165.0,3445,13.4,78,usa,buick regal sport coupe (turbo)
18.1,8,302.0,139.0,3205,11.2,78,usa,ford futura
17.5,8,318.0,140.0,4080,13.7,78,usa,dodge magnum xe
30.0,4,98.0,68.0,2155,16.5,78,usa,chevrolet chevette
27.5,4,134.0,95.0,2560,14.2,78,japan,toyota corona
27.2,4,119.0,97.0,2300,14.7,78,japan,datsun 510
30.9,4,105.0,75.0,2230,14.5,78,usa,dodge omni
21.1,4,134.0,95.0,2515,14.8,78,japan,toyota celica gt liftback
23.2,4,156.0,105.0,2745,16.7,78,usa,plymouth sapporo
23.8,4,151.0,85.0,2855,17.6,78,usa,oldsmobile starfire sx
23.9,4,119.0,97.0,2405,14.9,78,japan,datsun 200-sx
20.3,5,131.0,103.0,2830,15.9,78,europe,audi 5000
17.0,6,163.0,125.0,3140,13.6,78,europe,volvo 264gl
21.6,4,121.0,115.0,2795,15.7,78,europe,saab 99gle
16.2,6,163.0,133.0,3410,15.8,78,europe,peugeot 604sl
31.5,4,89.0,71.0,1990,14.9,78,europe,volkswagen scirocco
29.5,4,98.0,68.0,2135,16.6,78,japan,honda accord lx
21.5,6,231.0,115.0,3245,15.4,79,usa,pontiac lemans v6
19.8,6,200.0,85.0,2990,18.2,79,usa,mercury zephyr 6
22.3,4,140.0,88.0,2890,17.3,79,usa,ford fairmont 4
20.2,6,232.0,90.0,3265,18.2,79,usa,amc concord dl 6
20.6,6,225.0,110.0,3360,16.6,79,usa,dodge aspen 6
17.0,8,305.0,130.0,3840,15.4,79,usa,chevrolet caprice classic
17.6,8,302.0,129.0,3725,13.4,79,usa,ford ltd landau
16.5,8,351.0,138.0,3955,13.2,79,usa,mercury grand marquis
18.2,8,318.0,135.0,3830,15.2,79,usa,dodge st. regis
16.9,8,350.0,155.0,4360,14.9,79,usa,buick estate wagon (sw)
15.5,8,351.0,142.0,4054,14.3,79,usa,ford country squire (sw)
19.2,8,267.0,125.0,3605,15.0,79,usa,chevrolet malibu classic (sw)
18.5,8,360.0,150.0,3940,13.0,79,usa,chrysler lebaron town @ country (sw)
31.9,4,89.0,71.0,1925,14.0,79,europe,vw rabbit custom
34.1,4,86.0,65.0,1975,15.2,79,japan,maxda glc deluxe
35.7,4,98.0,80.0,1915,14.4,79,usa,dodge colt hatchback custom
27.4,4,121.0,80.0,2670,15.0,79,usa,amc spirit dl
25.4,5,183.0,77.0,3530,20.1,79,europe,mercedes benz 300d
23.0,8,350.0,125.0,3900,17.4,79,usa,cadillac eldorado
27.2,4,141.0,71.0,3190,24.8,79,europe,peugeot 504
23.9,8,260.0,90.0,3420,22.2,79,usa,oldsmobile cutlass salon brougham
34.2,4,105.0,70.0,2200,13.2,79,usa,plymouth horizon
34.5,4,105.0,70.0,2150,14.9,79,usa,plymouth horizon tc3
31.8,4,85.0,65.0,2020,19.2,79,japan,datsun 210
37.3,4,91.0,69.0,2130,14.7,79,europe,fiat strada custom
28.4,4,151.0,90.0,2670,16.0,79,usa,buick skylark limited
28.8,6,173.0,115.0,2595,11.3,79,usa,chevrolet citation
26.8,6,173.0,115.0,2700,12.9,79,usa,oldsmobile omega brougham
33.5,4,151.0,90.0,2556,13.2,79,usa,pontiac phoenix
41.5,4,98.0,76.0,2144,14.7,80,europe,vw rabbit
38.1,4,89.0,60.0,1968,18.8,80,japan,toyota corolla tercel
32.1,4,98.0,70.0,2120,15.5,80,usa,chevrolet chevette
37.2,4,86.0,65.0,2019,16.4,80,japan,datsun 310
28.0,4,151.0,90.0,2678,16.5,80,usa,chevrolet citation
26.4,4,140.0,88.0,2870,18.1,80,usa,ford fairmont
24.3,4,151.0,90.0,3003,20.1,80,usa,amc concord
19.1,6,225.0,90.0,3381,18.7,80,usa,dodge aspen
34.3,4,97.0,78.0,2188,15.8,80,europe,audi 4000
29.8,4,134.0,90.0,2711,15.5,80,japan,toyota corona liftback
31.3,4,120.0,75.0,2542,17.5,80,japan,mazda 626
37.0,4,119.0,92.0,2434,15.0,80,japan,datsun 510 hatchback
32.2,4,108.0,75.0,2265,15.2,80,japan,toyota corolla
46.6,4,86.0,65.0,2110,17.9,80,japan,mazda glc
27.9,4,156.0,105.0,2800,14.4,80,usa,dodge colt
40.8,4,85.0,65.0,2110,19.2,80,japan,datsun 210
44.3,4,90.0,48.0,2085,21.7,80,europe,vw rabbit c (diesel)
43.4,4,90.0,48.0,2335,23.7,80,europe,vw dasher (diesel)
36.4,5,121.0,67.0,2950,19.9,80,europe,audi 5000s (diesel)
30.0,4,146.0,67.0,3250,21.8,80,europe,mercedes-benz 240d
44.6,4,91.0,67.0,1850,13.8,80,japan,honda civic 1500 gl
40.9,4,85.0,,1835,17.3,80,europe,renault lecar deluxe
33.8,4,97.0,67.0,2145,18.0,80,japan,subaru dl
29.8,4,89.0,62.0,1845,15.3,80,europe,vokswagen rabbit
32.7,6,168.0,132.0,2910,11.4,80,japan,datsun 280-zx
23.7,3,70.0,100.0,2420,12.5,80,japan,mazda rx-7 gs
35.0,4,122.0,88.0,2500,15.1,80,europe,triumph tr7 coupe
23.6,4,140.0,,2905,14.3,80,usa,ford mustang cobra
32.4,4,107.0,72.0,2290,17.0,80,japan,honda accord
27.2,4,135.0,84.0,2490,15.7,81,usa,plymouth reliant
26.6,4,151.0,84.0,2635,16.4,81,usa,buick skylark
25.8,4,156.0,92.0,2620,14.4,81,usa,dodge aries wagon (sw)
23.5,6,173.0,110.0,2725,12.6,81,usa,chevrolet citation
30.0,4,135.0,84.0,2385,12.9,81,usa,plymouth reliant
39.1,4,79.0,58.0,1755,16.9,81,japan,toyota starlet
39.0,4,86.0,64.0,1875,16.4,81,usa,plymouth champ
35.1,4,81.0,60.0,1760,16.1,81,japan,honda civic 1300
32.3,4,97.0,67.0,2065,17.8,81,japan,subaru
37.0,4,85.0,65.0,1975,19.4,81,japan,datsun 210 mpg
37.7,4,89.0,62.0,2050,17.3,81,japan,toyota tercel
34.1,4,91.0,68.0,1985,16.0,81,japan,mazda glc 4
34.7,4,105.0,63.0,2215,14.9,81,usa,plymouth horizon 4
34.4,4,98.0,65.0,2045,16.2,81,usa,ford escort 4w
29.9,4,98.0,65.0,2380,20.7,81,usa,ford escort 2h
33.0,4,105.0,74.0,2190,14.2,81,europe,volkswagen jetta
34.5,4,100.0,,2320,15.8,81,europe,renault 18i
33.7,4,107.0,75.0,2210,14.4,81,japan,honda prelude
32.4,4,108.0,75.0,2350,16.8,81,japan,toyota corolla
32.9,4,119.0,100.0,2615,14.8,81,japan,datsun 200sx
31.6,4,120.0,74.0,2635,18.3,81,japan,mazda 626
28.1,4,141.0,80.0,3230,20.4,81,europe,peugeot 505s turbo diesel
30.7,6,145.0,76.0,3160,19.6,81,europe,volvo diesel
25.4,6,168.0,116.0,2900,12.6,81,japan,toyota cressida
24.2,6,146.0,120.0,2930,13.8,81,japan,datsun 810 maxima
22.4,6,231.0,110.0,3415,15.8,81,usa,buick century
26.6,8,350.0,105.0,3725,19.0,81,usa,oldsmobile cutlass ls
20.2,6,200.0,88.0,3060,17.1,81,usa,ford granada gl
17.6,6,225.0,85.0,3465,16.6,81,usa,chrysler lebaron salon
28.0,4,112.0,88.0,2605,19.6,82,usa,chevrolet cavalier
27.0,4,112.0,88.0,2640,18.6,82,usa,chevrolet cavalier wagon
34.0,4,112.0,88.0,2395,18.0,82,usa,chevrolet cavalier 2-door
31.0,4,112.0,85.0,2575,16.2,82,usa,pontiac j2000 se hatchback
29.0,4,135.0,84.0,2525,16.0,82,usa,dodge aries se
27.0,4,151.0,90.0,2735,18.0,82,usa,pontiac phoenix
24.0,4,140.0,92.0,2865,16.4,82,usa,ford fairmont futura
23.0,4,151.0,,3035,20.5,82,usa,amc concord dl
36.0,4,105.0,74.0,1980,15.3,82,europe,volkswagen rabbit l
37.0,4,91.0,68.0,2025,18.2,82,japan,mazda glc custom l
31.0,4,91.0,68.0,1970,17.6,82,japan,mazda glc custom
38.0,4,105.0,63.0,2125,14.7,82,usa,plymouth horizon miser
36.0,4,98.0,70.0,2125,17.3,82,usa,mercury lynx l
36.0,4,120.0,88.0,2160,14.5,82,japan,nissan stanza xe
36.0,4,107.0,75.0,2205,14.5,82,japan,honda accord
34.0,4,108.0,70.0,2245,16.9,82,japan,toyota corolla
38.0,4,91.0,67.0,1965,15.0,82,japan,honda civic
32.0,4,91.0,67.0,1965,15.7,82,japan,honda civic (auto)
38.0,4,91.0,67.0,1995,16.2,82,japan,datsun 310 gx
25.0,6,181.0,110.0,2945,16.4,82,usa,buick century limited
38.0,6,262.0,85.0,3015,17.0,82,usa,oldsmobile cutlass ciera (diesel)
26.0,4,156.0,92.0,2585,14.5,82,usa,chrysler lebaron medallion
22.0,6,232.0,112.0,2835,14.7,82,usa,ford granada l
32.0,4,144.0,96.0,2665,13.9,82,japan,toyota celica gt
36.0,4,135.0,84.0,2370,13.0,82,usa,dodge charger 2.2
27.0,4,151.0,90.0,2950,17.3,82,usa,chevrolet camaro
27.0,4,140.0,86.0,2790,15.6,82,usa,ford mustang gl
44.0,4,97.0,52.0,2130,24.6,82,europe,vw pickup
32.0,4,135.0,84.0,2295,11.6,82,usa,dodge rampage
28.0,4,120.0,79.0,2625,18.6,82,usa,ford ranger
31.0,4,119.0,82.0,2720,19.4,82,usa,chevy s-10
"""


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Conserve le brut pour le ML, impute une copie pour l'analyse descriptive.

    La médiane descriptive est calculée sur la totalité du fichier. Elle n'est
    JAMAIS réutilisée pour le ML : son imputation est ajustée sur le train seul.
    """
    raw = pd.read_csv(StringIO(CSV_DATA), na_values=["?", "", "NA"])
    for column in ["mpg"] + NUMERIC_FEATURES:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["origin"] = raw["origin"].astype("string").str.strip().str.lower()
    raw["name"] = raw["name"].astype("string").str.strip()
    missing_before = raw.isna().sum().to_dict()
    valid = (
        raw["mpg"].gt(0) & raw["origin"].isin(ORIGINS)
        & raw["name"].notna() & raw["name"].ne("")
    )
    # Les identifiants d'observation restent stables même après filtrage.
    raw = raw.loc[valid].copy()
    raw.insert(0, "vehicle_id", raw.index + 1)
    for column in ["cylinders", "model_year"]:
        if raw[column].isna().any():
            raise ValueError(f"La colonne {column} contient des valeurs invalides.")
        raw[column] = raw[column].astype(int)
    raw["brand"] = raw["name"].str.split().str[0].str.title()
    raw["year"] = 1900 + raw["model_year"]
    raw["l_100km"] = MPG_TO_L100 / raw["mpg"]
    raw["co2_g_km"] = raw["l_100km"] * CO2_FACTOR
    raw["horsepower_imputed"] = raw["horsepower"].isna()
    descriptive = raw.copy()
    medians = descriptive[NUMERIC_FEATURES].median()
    descriptive[NUMERIC_FEATURES] = descriptive[NUMERIC_FEATURES].fillna(medians)
    audit = {
        "rows": len(raw), "discarded": len(pd.read_csv(StringIO(CSV_DATA))) - len(raw),
        "missing": missing_before, "horsepower_median": float(medians["horsepower"]),
    }
    return raw, descriptive, audit


def filter_data(
    data: pd.DataFrame, origins: list[str], years: tuple[int, int],
    cylinders: list[int], query: str,
) -> pd.DataFrame:
    """Recherche littérale (sans interpréter les caractères comme une regex)."""
    mask = (
        data["origin"].isin(origins) & data["year"].between(*years)
        & data["cylinders"].isin(cylinders)
    )
    if query.strip():
        mask &= (
            data["name"].str.contains(query.strip(), case=False, regex=False, na=False)
            | data["brand"].str.contains(query.strip(), case=False, regex=False, na=False)
        )
    return data.loc[mask].copy()


def get_cuda() -> tuple[Any, str, list[str]]:
    """Teste réellement l'allocation et une opération CUDA, pas seulement l'import."""
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None, "", ["CuPy ne détecte aucun périphérique CUDA."]
        cp.cuda.Device(0).use()
        _ = (cp.ones(8, dtype=cp.float32) * 2).sum().item()
        cp.cuda.runtime.deviceSynchronize()
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        return cp, str(name), []
    except Exception as exc:
        return None, "", [f"CUDA / CuPy indisponible : {type(exc).__name__}: {exc}"]


@dataclass
class ModelBundle:
    model: Any
    preprocessor: ColumnTransformer
    backend: str
    gpu_name: str
    cp: Any
    train_ms: float
    predict_ms: float
    r2: float
    rmse: float
    baseline_rmse: float
    train_n: int
    test_n: int
    X_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_rows: pd.DataFrame
    predictions: np.ndarray
    notices: list[str]
    training_warnings: list[str]
    importance: pd.DataFrame | None = None

    @property
    def is_gpu(self) -> bool:
        return self.cp is not None and self.backend in ("cuml", "xgboost")

    @property
    def engine_label(self) -> str:
        return {
            "cuml": "Random Forest • cuML / CUDA",
            "xgboost": "XGBoost • histogrammes / CUDA",
            "sklearn": "Random Forest • scikit-learn / CPU",
        }[self.backend]


def predict_encoded(bundle: ModelBundle, X: np.ndarray) -> np.ndarray:
    """Le moteur GPU reçoit des tableaux GPU, y compris pour UNE observation."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    if bundle.is_gpu:
        cp = bundle.cp
        cp.cuda.Device(0).use()
        device_X = cp.asarray(X)
        if bundle.backend == "xgboost":
            result = bundle.model.get_booster().inplace_predict(device_X)
        else:
            result = bundle.model.predict(device_X)
        # Un retour CPU invalide la promesse d'inférence CUDA dans cette application.
        if not isinstance(result, cp.ndarray):
            raise RuntimeError("Le moteur GPU n'a pas retourné un tableau CuPy.")
        cp.cuda.runtime.deviceSynchronize()
        result = cp.asnumpy(result).reshape(-1)
    else:
        result = np.asarray(bundle.model.predict(X)).reshape(-1)
    if not np.isfinite(result).all():
        raise RuntimeError("Le modèle a produit une valeur non finie.")
    return result


def train_model(
    rows: pd.DataFrame, n_estimators: int = 100, max_depth: int = 10,
    test_size: float = 0.2, force_cpu: bool = False,
) -> ModelBundle:
    """cuML → XGBoost CUDA → sklearn ; toute erreur GPU est expliquée.

    Split AVANT imputation et encodage : les statistiques du test ne sont pas
    utilisées pour entraîner le prétraitement. Aucun dérivé du MPG n'est une entrée.
    """
    if len(rows) < 30:
        raise ValueError("Il faut au moins 30 véhicules pour entraîner et évaluer le modèle.")
    if rows["mpg"].nunique() < 2:
        raise ValueError("La cible MPG doit contenir au moins deux valeurs différentes.")
    train, test = train_test_split(rows, test_size=test_size, random_state=SEED)
    if train[NUMERIC_FEATURES].notna().sum().eq(0).any():
        raise ValueError("Une caractéristique est entièrement manquante dans le train.")
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("origin", OneHotEncoder(
            categories=[ORIGINS], handle_unknown="ignore", sparse_output=False,
            dtype=np.float32,
        ), ["origin"]),
    ], remainder="drop")
    X_train = np.ascontiguousarray(preprocessor.fit_transform(train[FEATURES]), dtype=np.float32)
    X_test = np.ascontiguousarray(preprocessor.transform(test[FEATURES]), dtype=np.float32)
    y_train = train["mpg"].to_numpy(dtype=np.float32)
    y_test = test["mpg"].to_numpy(dtype=np.float32)
    cp, gpu_name, notices = (None, "", []) if force_cpu else get_cuda()
    if force_cpu:
        notices.append("Repli CPU demandé après un échec d'inférence GPU.")
    candidates = ["cuml", "xgboost", "sklearn"] if cp is not None else ["sklearn"]
    for backend in candidates:
        try:
            model_cp = cp if backend != "sklearn" else None
            if backend == "cuml":
                from cuml.ensemble import RandomForestRegressor as CuRandomForest
                model = CuRandomForest(
                    n_estimators=n_estimators, max_depth=max_depth,
                    n_bins=min(128, len(train)), max_features=1.0,
                    n_streams=1, random_state=SEED, output_type="cupy",
                )
            elif backend == "xgboost":
                import xgboost as xgb
                if int(xgb.__version__.split(".")[0]) < 2:
                    raise RuntimeError("XGBoost >= 2.0 requis pour device='cuda'.")
                model = xgb.XGBRegressor(
                    n_estimators=n_estimators, max_depth=max_depth,
                    tree_method="hist", device="cuda:0", learning_rate=0.05,
                    objective="reg:squarederror", random_state=SEED,
                    n_jobs=-1, verbosity=1,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth,
                    random_state=SEED, n_jobs=-1,
                )
            # Les transferts initiaux sont exclus du temps d'entraînement affiché.
            if model_cp is not None:
                device_X, device_y = cp.asarray(X_train), cp.asarray(y_train)
                cp.cuda.runtime.deviceSynchronize()
            else:
                device_X, device_y = X_train, y_train
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                started = time.perf_counter()
                model.fit(device_X, device_y)
                if model_cp is not None:
                    cp.cuda.runtime.deviceSynchronize()
                train_ms = (time.perf_counter() - started) * 1000
                if backend == "xgboost":
                    config = json.loads(model.get_booster().save_config())
                    device = config["learner"]["generic_param"]["device"]
                    if not str(device).startswith("cuda"):
                        raise RuntimeError(f"XGBoost a basculé sur {device} pendant fit.")
                bundle = ModelBundle(
                    model=model, preprocessor=preprocessor, backend=backend,
                    gpu_name=gpu_name if model_cp is not None else "", cp=model_cp,
                    train_ms=train_ms, predict_ms=0.0, r2=0.0, rmse=0.0,
                    baseline_rmse=float(np.sqrt(mean_squared_error(
                        y_test, np.full(len(y_test), float(y_train.mean()))))),
                    train_n=len(train), test_n=len(test), X_train=X_train, X_test=X_test,
                    y_test=y_test, test_rows=test.copy(), predictions=np.array([]),
                    notices=list(notices), training_warnings=[],
                )
                started = time.perf_counter()
                predictions = predict_encoded(bundle, X_test)
                bundle.predict_ms = (time.perf_counter() - started) * 1000
            messages = [str(w.message) for w in caught]
            # XGBoost peut émettre un warning de repli sans lever d'exception.
            if model_cp is not None and any(
                "falling back" in m.lower() or "not compiled with cuda" in m.lower()
                or "no visible gpu" in m.lower() for m in messages
            ):
                raise RuntimeError(" ; ".join(messages))
            bundle.training_warnings = messages
            bundle.predictions = predictions
            bundle.r2 = float(r2_score(y_test, predictions))
            bundle.rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
            return bundle
        except Exception as exc:
            if backend == "sklearn":
                raise RuntimeError(f"Échec du Random Forest CPU : {exc}") from exc
            notices.append(f"{backend} indisponible : {type(exc).__name__}: {exc}")
    raise RuntimeError("Aucun moteur de régression utilisable.")


def permutation_importance_gpu_aware(bundle: ModelBundle) -> pd.DataFrame:
    """Hausse de RMSE sur le test après permutation (3 répétitions).

    Les 3 colonnes one-hot d'origine sont permutées ensemble pour conserver des
    combinaisons valides. Les prédictions utilisent le moteur actif, dont CUDA.
    Les variables corrélées peuvent se partager l'importance.
    """
    rng = np.random.default_rng(SEED)
    groups = [(name, [i]) for i, name in enumerate(NUMERIC_FEATURES)]
    groups.append(("origin", list(range(len(NUMERIC_FEATURES), bundle.X_test.shape[1]))))
    records = []
    for name, indices in groups:
        changes = []
        for _ in range(3):
            permuted = bundle.X_test.copy()
            permuted[:, indices] = bundle.X_test[rng.permutation(len(permuted))][:, indices]
            predictions = predict_encoded(bundle, permuted)
            changes.append(float(np.sqrt(mean_squared_error(bundle.y_test, predictions))) - bundle.rmse)
        records.append({
            "Variable": LABELS[name], "Hausse de RMSE (MPG)": np.mean(changes),
            "Écart-type": np.std(changes),
        })
    return pd.DataFrame(records).sort_values("Hausse de RMSE (MPG)")


def show_plot(figure: go.Figure) -> None:
    figure.update_layout(
        margin=dict(l=12, r=12, t=45, b=15),
        legend_title_text="", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch")


def display_table(data: pd.DataFrame) -> None:
    st.dataframe(data.rename(columns=LABELS), width="stretch", hide_index=True)


def render_analysis(data: pd.DataFrame) -> None:
    st.subheader("Comprendre les véhicules sélectionnés")
    if data.empty:
        st.info("Aucun véhicule ne correspond aux filtres. Élargissez votre sélection.")
        return
    metrics = st.columns(5)
    metrics[0].metric("Puissance moyenne", f"{data.horsepower.mean():.1f} hp")
    metrics[1].metric("Efficacité moyenne", f"{data.mpg.mean():.2f} MPG")
    metrics[2].metric("Consommation moyenne", f"{data.l_100km.mean():.2f} L/100 km")
    metrics[3].metric("CO₂ moyen estimé", f"{data.co2_g_km.mean():.0f} g/km")
    metrics[4].metric("Véhicules", len(data))
    st.caption(
        "Moyennes calculées véhicule par véhicule. La moyenne des L/100 km n'est pas "
        "la conversion de la moyenne des MPG. Les puissances manquantes sont imputées."
    )
    c1, c2 = st.columns(2)
    with c1:
        unit = st.radio("Consommation à afficher", ["L/100 km", "MPG"], horizontal=True)
    with c2:
        color_choice = st.radio("Couleur des points", ["Origine", "Cylindres"], horizontal=True)
    target = "l_100km" if unit == "L/100 km" else "mpg"
    color = "origin" if color_choice == "Origine" else "cylinders_label"
    plot_data = data.assign(cylinders_label=data.cylinders.astype(str))
    show_plot(px.scatter(
        plot_data, x="weight", y=target, color=color,
        color_discrete_map=PALETTE if color == "origin" else None,
        hover_name="name", hover_data=["year", "horsepower", "origin"],
        labels={**LABELS, "cylinders_label": "Cylindres"}, opacity=0.8,
        title="Poids et consommation",
    ))
    left, right = st.columns(2)
    with left:
        annual = data.groupby("year", as_index=False).agg(
            mpg=("mpg", "mean"), l_100km=("l_100km", "mean"), count=("mpg", "size")
        )
        show_plot(px.line(
            annual, x="year", y=target, markers=True, hover_data=["count"],
            labels={**LABELS, "count": "Véhicules"}, title="Évolution selon l'année du modèle",
        ))
    with right:
        counts = data.groupby("origin", as_index=False).size()
        show_plot(px.bar(
            counts, x="origin", y="size", color="origin", color_discrete_map=PALETTE,
            labels={"origin": "Origine", "size": "Nombre de véhicules"},
            text_auto=True, title="Répartition par origine",
        ))
    left, right = st.columns(2)
    with left:
        show_plot(px.box(
            data, x="origin", y="horsepower", color="origin", points="outliers",
            color_discrete_map=PALETTE, labels=LABELS, title="Puissance par origine",
        ))
    with right:
        show_plot(px.box(
            data, x="origin", y=target, color="origin", points="outliers",
            color_discrete_map=PALETTE, labels=LABELS, title="Consommation par origine",
        ))
    cols = ["name", "year", "origin", "mpg", "l_100km", "co2_g_km"]
    left, right = st.columns(2)
    with left:
        st.markdown("#### 🏆 Les 5 plus économiques")
        display_table(data.nsmallest(5, "l_100km")[cols].round(2))
    with right:
        st.markdown("#### ⛽ Les 5 plus gourmands")
        display_table(data.nlargest(5, "l_100km")[cols].round(2))
    st.markdown("#### Explorer et télécharger")
    table = data.drop(columns="vehicle_id").round(3)
    display_table(table)
    st.download_button(
        "⬇ Télécharger la sélection en CSV", table.to_csv(index=False).encode("utf-8-sig"),
        "auto_mpg_selection.csv", "text/csv", key="download_filtered",
    )


def render_comparator(data: pd.DataFrame) -> None:
    st.subheader("Deux véhicules, une comparaison directe")
    st.caption("Catalogue complet : ce comparateur reste indépendant des filtres globaux.")
    catalog = data.set_index("vehicle_id", drop=False)
    ids = catalog.sort_values(["name", "year"]).index.tolist()

    def vehicle_label(identifier: int) -> str:
        row = catalog.loc[identifier]
        return f"{row['name'].title()} • {row['year']} • {row['origin']} • #{identifier}"

    left, right = st.columns(2)
    a_id = left.selectbox("Véhicule A", ids, format_func=vehicle_label, key="compare_a")
    b_ids = [i for i in ids if i != a_id]
    b_id = right.selectbox("Véhicule B", b_ids, format_func=vehicle_label, key="compare_b")
    a, b = catalog.loc[a_id], catalog.loc[b_id]
    columns = [
        "mpg", "l_100km", "cylinders", "displacement", "horsepower", "weight",
        "acceleration", "year", "origin", "co2_g_km",
    ]
    records = []
    for field in columns:
        av, bv = a[field], b[field]
        records.append({
            "Caractéristique": LABELS[field],
            "A — " + a["name"].title(): f"{av:.2f}" if isinstance(av, (float, np.floating)) else str(av),
            "B — " + b["name"].title(): f"{bv:.2f}" if isinstance(bv, (float, np.floating)) else str(bv),
        })
    st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
    saved = float(a.l_100km - b.l_100km)
    percentage = saved / float(a.l_100km) * 100
    left, middle, right = st.columns(3)
    left.metric("B : consommation", f"{b.l_100km:.2f} L/100 km", f"{b.l_100km-a.l_100km:+.2f} vs A", delta_color="inverse")
    middle.metric("B : puissance", f"{b.horsepower:.0f} hp", f"{b.horsepower-a.horsepower:+.0f} vs A")
    right.metric("B : CO₂ estimé", f"{b.co2_g_km:.0f} g/km", f"{b.co2_g_km-a.co2_g_km:+.0f} vs A", delta_color="inverse")
    if saved >= 0:
        st.success(f"B consomme {saved:.2f} L/100 km de moins que A, soit une économie de {percentage:.1f} % à distance égale.")
    else:
        st.info(f"B consomme {-saved:.2f} L/100 km de plus que A, soit une hausse de {-percentage:.1f} %.")
    if a.horsepower_imputed or b.horsepower_imputed:
        st.caption("La puissance d'au moins un véhicule comparé est imputée par la médiane descriptive.")


def render_hardware(slot: Any, bundle: ModelBundle | None, error: str = "") -> None:
    with slot.container():
        st.markdown("### ⚡ Statut du matériel")
        if bundle is None:
            st.info("⚪ Aucun modèle actif")
            if error:
                st.caption(error)
            return
        if bundle.is_gpu:
            st.success(f"🟢 GPU {bundle.gpu_name} activé (CUDA)")
        else:
            st.warning("🟠 CPU Fallback")
        st.caption(bundle.engine_label)
        with st.expander("Diagnostic des moteurs"):
            st.write(f"Python {platform.python_version()} • {platform.system()}")
            for message in bundle.notices:
                st.code(message, language=None)
            for message in bundle.training_warnings:
                st.caption(message)
            if not bundle.notices and not bundle.training_warnings:
                st.write("Entraînement et prédiction de test validés sur le moteur indiqué.")
            st.caption("Le GPU utilisé est celui du serveur qui exécute Python.")


def render_prediction(bundle: ModelBundle, reference: pd.DataFrame, config_key: str) -> None:
    st.subheader("Estimer la consommation d'un véhicule")
    st.caption(bundle.engine_label + " • Entraînement et inférence de test vérifiés")
    metrics = st.columns(4)
    metrics[0].metric("R² sur le test", f"{bundle.r2:.3f}")
    metrics[1].metric("RMSE sur le test", f"{bundle.rmse:.2f} MPG")
    metrics[2].metric("Entraînement", f"{bundle.train_ms:.1f} ms")
    metrics[3].metric("Prédiction du test", f"{bundle.predict_ms:.1f} ms")
    st.caption(
        f"{bundle.train_n} observations d'entraînement / {bundle.test_n} de test • graine {SEED}. "
        "Temps fit hors prétraitement et transferts initiaux ; synchronisation CUDA incluse. "
        "Temps de prédiction du test avec transferts. Valeurs conservées entre les interactions."
    )
    if bundle.backend == "xgboost":
        st.info("cuML n'est pas utilisable : le moteur actif est XGBoost CUDA (boosting d'arbres).")
    if bundle.importance is not None:
        with st.expander("Importance des variables", expanded=True):
            show_plot(px.bar(
                bundle.importance, x="Hausse de RMSE (MPG)", y="Variable",
                error_x="Écart-type", orientation="h", title="Importance par permutation sur le test",
            ))
            st.caption(
                "3 permutations par variable ; les colonnes d'origine sont permutées ensemble. "
                "Plus la hausse de RMSE est grande, plus le modèle utilise la variable. "
                "Une importance négative est possible. Les variables corrélées partagent l'information."
            )
    st.markdown("#### Configurer le véhicule")
    with st.form("vehicle_prediction"):
        left, right = st.columns(2)
        with left:
            cylinders = st.select_slider("Nombre de cylindres", options=[3, 4, 5, 6, 8], value=4)
            displacement = st.slider("Cylindrée (in³)", 68.0, 455.0, 140.0, 0.5)
            horsepower = st.slider("Puissance (hp)", 46.0, 230.0, 95.0, 1.0)
            weight = st.slider("Poids (lb)", 1613, 5140, 2500, 1)
        with right:
            acceleration = st.slider("Accélération 0–60 mph (secondes)", 8.0, 24.8, 15.0, 0.1)
            year = st.slider("Année du modèle", 1970, 1982, 1978)
            origin = st.selectbox("Origine du véhicule", ORIGINS)
            annual_km = st.number_input("Distance annuelle (km/an)", 0, 200000, 15000, 500)
        submitted = st.form_submit_button("Prédire la consommation", type="primary")
    pending = st.session_state.pop("pending_prediction", None)
    if submitted or (pending and pending["config_key"] == config_key):
        custom = pd.DataFrame([{
            "cylinders": cylinders, "displacement": displacement,
            "horsepower": horsepower, "weight": weight,
            "acceleration": acceleration, "model_year": year - 1900, "origin": origin,
        }])
        if pending and pending["config_key"] == config_key:
            custom = pd.DataFrame([pending["custom"]])
            annual_km = pending["annual_km"]
        outside = [
            LABELS[c] for c in NUMERIC_FEATURES
            if not reference[c].min() <= custom.iloc[0][c] <= reference[c].max()
        ]
        if origin not in reference.origin.unique():
            outside.append("Origine absente de l'entraînement")
        encoded = bundle.preprocessor.transform(custom[FEATURES])
        try:
            started = time.perf_counter()
            mpg = float(predict_encoded(bundle, encoded)[0])
            elapsed = (time.perf_counter() - started) * 1000
            if mpg <= 0:
                raise ValueError("Une prédiction MPG positive est nécessaire pour la conversion.")
            st.session_state["vehicle_result"] = {
                "config_key": config_key, "mpg": mpg, "annual_km": annual_km,
                "elapsed_ms": elapsed, "outside": outside,
            }
        except Exception as exc:
            st.session_state.pop("vehicle_result", None)
            if bundle.is_gpu:
                # Le prochain run réentraîne un vrai RandomForest CPU sur le même split.
                st.session_state["pending_prediction"] = {
                    "config_key": config_key, "custom": custom.iloc[0].to_dict(),
                    "annual_km": annual_km,
                }
                st.session_state["cpu_retry_key"] = config_key
                st.session_state["gpu_runtime_error"] = str(exc)
                st.session_state.pop("model_signature", None)
                st.rerun()
            st.error(f"La prédiction a échoué : {exc}")
    result = st.session_state.get("vehicle_result")
    if result and result["config_key"] == config_key:
        mpg = result["mpg"]
        liters = MPG_TO_L100 / mpg
        co2 = liters * CO2_FACTOR
        annual_tonnes = co2 * result["annual_km"] / 1_000_000
        metrics = st.columns(4)
        metrics[0].metric("Efficacité prédite", f"{mpg:.2f} MPG US")
        metrics[1].metric("Consommation prédite", f"{liters:.2f} L/100 km")
        metrics[2].metric("CO₂ estimé", f"{co2:.0f} g/km")
        metrics[3].metric("CO₂ annuel estimé", f"{annual_tonnes:.2f} t/an")
        st.caption(
            f"Distance : {result['annual_km']:,} km/an • "
            f"Carburant : {liters * result['annual_km'] / 100:,.0f} L/an • "
            f"Prédiction seule : {result['elapsed_ms']:.2f} ms ({bundle.engine_label})."
        )
        if result["outside"]:
            st.warning("Caractéristiques hors de la sélection d'apprentissage : " + ", ".join(result["outside"]))
    st.caption(
        "Estimation pédagogique sur des modèles 1970–1982. CO₂ = L/100 km × 23,5 ; "
        "facteur uniforme assimilé à l'essence, y compris pour les véhicules diesel du fichier. "
        "Ce calcul ne couvre ni le cycle de vie ni tous les carburants."
    )


def render_diagnostics(raw: pd.DataFrame, audit: dict[str, Any], bundle: ModelBundle | None) -> None:
    st.subheader("Données, méthode et diagnostics")
    metrics = st.columns(3)
    metrics[0].metric("Observations intégrées", audit["rows"])
    metrics[1].metric("Puissances manquantes", audit["missing"]["horsepower"])
    metrics[2].metric("Lignes écartées", audit["discarded"])
    quality = pd.DataFrame({"Colonne": audit["missing"].keys(), "Valeurs manquantes": audit["missing"].values()})
    display_table(quality)
    st.markdown(
        f"**Préparation.** Pour les graphiques et le comparateur, les puissances manquantes "
        f"sont remplacées par la médiane du fichier ({audit['horsepower_median']:.1f} hp). "
        "Pour le ML, les données brutes sont séparées en train/test, puis les médianes sont "
        "apprises sur le train uniquement. L'origine est encodée en one-hot ; le nom, la marque, "
        "le MPG observé et ses dérivés sont exclus des caractéristiques d'entrée."
    )
    st.markdown(
        "**Unités.** MPG US (plus élevé = plus économique) ; poids en livres (lb), "
        "cylindrée en pouces cubes (in³), puissance en hp et accélération 0–60 mph en secondes. "
        "Les origines sont des régions, pas trois pays. La marque est le premier mot du nom ; "
        "les variantes et fautes d'origine (vw, chevy, maxda…) sont conservées."
    )
    st.latex(r"C_{L/100\,km}=\frac{235.215}{MPG}\quad;\quad CO_{2,g/km}=23.5\,C\quad;\quad CO_{2,t/an}=\frac{CO_{2,g/km}\,d_{km/an}}{10^6}")
    st.markdown(
        "**Interprétation.** Le R² peut être négatif ; la RMSE s'exprime en MPG. "
        "Le test est une séparation aléatoire reproductible, pas une validation chronologique "
        "ou par famille de véhicules. Des versions proches peuvent donc se trouver des deux côtés. "
        "Le petit volume de données ne garantit aucun gain de temps avec le GPU."
    )
    st.download_button(
        "⬇ Télécharger le CSV brut intégré", CSV_DATA.encode("utf-8-sig"),
        "auto_mpg_brut.csv", "text/csv", key="download_raw",
    )
    if bundle is None:
        st.info("Les diagnostics de régression apparaîtront lorsqu'un modèle pourra être entraîné.")
        return
    st.markdown("#### Comparer les erreurs sur le test")
    left, right = st.columns(2)
    left.metric("RMSE du modèle", f"{bundle.rmse:.2f} MPG")
    right.metric("RMSE de référence", f"{bundle.baseline_rmse:.2f} MPG")
    st.caption("La référence prédit toujours le MPG moyen du train.")
    diagnostics = bundle.test_rows[["vehicle_id", "name", "year", "origin", "mpg"]].copy()
    diagnostics["prediction_mpg"] = bundle.predictions
    diagnostics["residual_mpg"] = bundle.y_test - bundle.predictions
    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            diagnostics, x="mpg", y="prediction_mpg", color="origin", hover_name="name",
            color_discrete_map=PALETTE,
            labels={"mpg": "MPG observé", "prediction_mpg": "MPG prédit", "origin": "Origine"},
            title="Prédictions et observations",
        )
        low = float(min(diagnostics.mpg.min(), diagnostics.prediction_mpg.min()))
        high = float(max(diagnostics.mpg.max(), diagnostics.prediction_mpg.max()))
        fig.add_shape(type="line", x0=low, x1=high, y0=low, y1=high, line=dict(dash="dash", color="gray"))
        show_plot(fig)
    with right:
        fig = px.scatter(
            diagnostics, x="prediction_mpg", y="residual_mpg", color="origin", hover_name="name",
            color_discrete_map=PALETTE,
            labels={"prediction_mpg": "MPG prédit", "residual_mpg": "Observé − prédit (MPG)", "origin": "Origine"},
            title="Résidus sur le test",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        show_plot(fig)
    st.dataframe(diagnostics.round(3), hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Auto MPG · Vehicle Lab", page_icon="🚘", layout="wide")
    st.markdown("""<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {border: 1px solid #8192a633; border-radius: 12px; padding: 14px;}
    div[data-testid="stMetricValue"] {font-size: 1.65rem;}
    </style>""", unsafe_allow_html=True)
    st.title("🚘 Auto MPG · Vehicle Lab")
    st.markdown("**Explorer, comparer et prédire la consommation automobile.**")
    raw, data, audit = load_data()
    with st.sidebar:
        st.header("Votre laboratoire automobile")
        st.markdown("### 1 · Filtres globaux")
        origins = st.multiselect("Origine", ORIGINS, default=ORIGINS)
        years = st.slider("Année du modèle", int(data.year.min()), int(data.year.max()), (1970, 1982))
        cylinders = st.multiselect("Cylindres", sorted(data.cylinders.unique().tolist()), default=sorted(data.cylinders.unique().tolist()))
        query = st.text_input("🔎 Modèle ou marque", placeholder="Toyota, Mustang…")
        selected = filter_data(data, origins, years, cylinders, query)
        st.caption(f"{len(selected)} / {len(data)} véhicules sélectionnés")
        st.markdown("### 2 · Hyperparamètres ML")
        n_estimators = st.slider("Nombre d'arbres", 10, 300, 100, 10)
        max_depth = st.slider("Profondeur maximale", 1, 20, 10)
        test_size = st.slider("Proportion de test", 0.1, 0.4, 0.2, 0.05)
        use_filtered = st.checkbox("Entraîner sur la sélection filtrée", value=True)
        st.caption("Décochez pour entraîner sur le catalogue complet. Le comparateur utilise toujours le catalogue complet.")
        retry = st.button("Réessayer les moteurs GPU", width="stretch")
        hardware_slot = st.empty()
    # Le sous-ensemble ML repart du brut : pas de médiane descriptive en entrée.
    training_rows = raw.loc[raw.vehicle_id.isin(selected.vehicle_id)] if use_filtered else raw
    signature = hashlib.sha256(
        repr((training_rows.vehicle_id.tolist(), n_estimators, max_depth, test_size)).encode()
    ).hexdigest()
    if retry:
        st.session_state.pop("model_signature", None)
        st.session_state.pop("cpu_retry_key", None)
        st.session_state.pop("gpu_runtime_error", None)
    bundle = None
    training_error = ""
    if len(training_rows) >= 30 and training_rows.mpg.nunique() >= 2:
        if st.session_state.get("model_signature") != signature:
            st.session_state.pop("vehicle_result", None)
            try:
                with st.spinner("Entraînement et vérification du moteur de calcul…"):
                    bundle = train_model(
                        training_rows, n_estimators, max_depth, test_size,
                        force_cpu=st.session_state.get("cpu_retry_key") == signature,
                    )
                    if st.session_state.get("cpu_retry_key") == signature:
                        bundle.notices.append("Échec GPU en cours d'utilisation : " + st.session_state.get("gpu_runtime_error", ""))
                    try:
                        bundle.importance = permutation_importance_gpu_aware(bundle)
                    except Exception as exc:
                        if bundle.is_gpu:
                            failure = f"Échec de l'inférence GPU pendant les permutations : {exc}"
                            bundle = train_model(training_rows, n_estimators, max_depth, test_size, force_cpu=True)
                            bundle.notices.append(failure)
                            bundle.importance = permutation_importance_gpu_aware(bundle)
                        else:
                            raise
                # Stockage par session : pas de modèle mutable partagé entre utilisateurs.
                st.session_state["model_bundle"] = bundle
                st.session_state["model_signature"] = signature
            except Exception as exc:
                st.session_state.pop("model_bundle", None)
                st.session_state.pop("model_signature", None)
                training_error = str(exc)
        else:
            bundle = st.session_state.get("model_bundle")
    else:
        training_error = "Sélection insuffisante : au moins 30 véhicules et deux valeurs MPG distinctes sont nécessaires."
    render_hardware(hardware_slot, bundle, training_error)
    if st.session_state.get("cpu_retry_key") == signature and bundle and not bundle.is_gpu:
        st.warning("Le GPU a rencontré une erreur. Le modèle a été réentraîné sur CPU ; la prédiction demandée est recalculée automatiquement.")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analyse descriptive", "⚡ Comparateur", "🤖 Prédiction & carbone", "🔬 Données & diagnostics"
    ])
    with tab1:
        render_analysis(selected)
    with tab2:
        render_comparator(data)
    with tab3:
        if bundle is None:
            st.info(training_error or "Aucun modèle actif. Modifiez les filtres ou les paramètres.")
        else:
            render_prediction(bundle, training_rows, signature)
    with tab4:
        render_diagnostics(raw, audit, bundle)
    st.caption("Auto MPG • 398 observations historiques • Les émissions sont estimées avec le facteur fourni.")


if __name__ == "__main__":
    main()
