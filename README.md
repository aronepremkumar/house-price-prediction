# House Price Prediction

## Problem Description
Homeowners, buyers, and real estate agents often struggle with accurate house valuations due to varying market conditions, property features, and visual appeal. Delays in pricing can lead to over/under-selling. The goal is to predict house prices to aid in listings, negotiations, or investments, reducing financial risks in urban markets like San Francisco or Southern California.

## Dataset
- Tabular: California Housing Prices (Kaggle: camnugent/california-housing-prices) – ~20k samples with features like sq_ft (derived), bedrooms, income, etc., and target median_house_value.
- Images: Extended with CubiCasa5k floor plans (Kaggle: qmarva/cubicasa5k) – Randomly paired as interiors.

## Setup
1. Create virtual env: python -m venv env; source env/bin/activate
2. pip install -r requirements.txt
3. Run python data_prep.py to download/prepare data.

## Usage
- EDA: jupyter notebook eda.ipynb
- Train: jupyter notebook train.ipynb or python train.py
- API: uvicorn main:app --reload
- Docker: docker build -t house-price-predictor .; docker run -p 8000:8000 house-price-predictor
- EKS: Follow deployment steps below.

## Deployment to AWS EKS
1. Create ECR repo: aws ecr create-repository --repository-name house-price-predictor
2. Push Docker image: (auth, tag, push as in guide)
3. Create EKS cluster: eksctl create cluster --name price-cluster --region us-west-1 --nodes 2
4. kubectl apply -f deployment.yaml
5. Test URL from kubectl get svc