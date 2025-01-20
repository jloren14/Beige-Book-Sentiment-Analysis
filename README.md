# Predicting U.S. Economic Recessions Using Sentiment Analysis of the Federal Reserve’s Beige Book
The repository contains the codes and files I used in the final project for the course "Transforming Business with AI: The Power of the Large Language Models" at NHH Norwegian School of Economics. The data scraping was done by Håkon Molnes and the rest of coding for the project by Julia Lorenc. 

## Table of Contents
* [Project Overview](#project-overview)
* [Data Information](#data-information)
* [Methodology and Analysis](#methodology-and-analysis)
* [Key Findings](#key-findings)
* [Future Work](#future-work)

## Project Overview
The Beige Book is a qualitative economic report providing insights into regional economic conditions across 12 Federal Reserve districts in United States. In this project, we investigate whether sentiment scores derived from the Federal Reserve’s Beige Book can predict U.S. economic recessions, by applying advanced sentiment analysis methodologies and machine learning models. Additionally, we want to measure to what extent the sentiment scores enhance the accuracy of traditional forecasting models.

## Data Information
Beige Book reports from years 1997-2014 were scraped from the [Federal Reserve Bank of Minneapolis](https://www.minneapolisfed.org/region-and-community/regional-economic-indicators/beige-book-archive) website. 

## Methodology and Analysis
The reports collected from years 1997-2024 cover three major recessions:
* Dot-com bubble (2001)
* Great Recession (2007-2009)
* COVID-19 Recession (2020)

1. Scraping the reports and storing them as a structured Data Frame
2. Sentiment Analysis with [finBERT](https://huggingface.co/ProsusAI/finbert) model
     * Splitting every report into sentences
     * Processing each sentence through `finBERT` for sentiment classification (positive = 1, neutral = 0, negative = -1)
     * Calculating the average quantitative sentiment score for each report out of sentence scores
     * Averaging the sentiment scores at the district level
  
3. Predictive modelling with XGBoost model on raw `finBERT` scores
     * SMOTE application to address class imbalance (rare recession events)
     * Model training
     * Model evaluation with precision, recall, F1-score and ROC-AUC
     * SHAP analysis to determine district-level importance
  
4. `finBERT` fine-tuning on Beige Book specific language
     * Collecting Beige Book reports outside of the sentiment analysis scope
     * Splitting the reports into sentences
     * Processing each sentence through the `OpenAI GPT-4` model with API to label the sentiment of the sentences for training and validation data
     * Using [ProsusAI finBERT GitHub repository](https://github.com/ProsusAI/finBERT) for fine-tuning on the collected and labeled data
  
5. Sentiment Analysis with fine-tuned `finBERT` (the same procedure as step 2)
6. Predictive modelling with XGBoost on fine-tuned `finBERT` scores (the same procedure as step 3)

## Key Findings
* XGBoost model on raw `finBERT` scores:
    * Recall for recession prediction: 50%
    * Most predictive districts: New York, Boston, Dallas
 
* XGBoost model on ine-tuned `finBERT` scores:
    * Recall improved from 50% to 75%
    * Most predictive districts: San Francisco, Boston, Cleveland
 
## Future work
Future research should emphasize fine-tuning hyperparameters of predictive models to further enhance their performance, particularly in improving the model's ability to accurately identify recessions. Additionally, exploring alternative predictive algorithms could broaden the scope of this project, potentially uncovering models better suited to sentiment-based data. Finally, refining the calculation of sentiment scores through advanced techniques – such as manipulating probabilities and logits within BERT-based models or testing the abilities of the GPT-4 model – could significantly enhance the granularity and precision of the sentiment scores. This, in turn, could improve the predictive power of the models, elevating their overall performance and reliability.
