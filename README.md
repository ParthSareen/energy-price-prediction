# Energy Price Prediction

This project is dedicated to exploring and implementing time series forecasting models, focusing on energy demand and capacity prediction. The project utilizes various data exploration techniques and machine learning models, including RNNs and Autoformers, to analyze and predict energy demand and capacity based on historical data.

## Project Structure

- **Data Exploration**: Initial data analysis and visualization are performed in Jupyter notebooks (`eda.ipynb`). This includes loading data, cleaning, and preparing it for modeling.
  
- **Modeling**: The project explores two main types of models for time series forecasting:
  - **RNN Model**: Implemented in `rnn_model.ipynb`, this model uses Recurrent Neural Networks to predict energy demand and capacity.


## Setup

To run the notebooks and scripts in this project, ensure you have the following installed:
- Python 3.11.7 or higher
- Jupyter Notebook
- Required Python packages: `pandas`, `matplotlib`, `tensorflow`, `torch`, `transformers`

You can install the necessary packages using `pip`:
`pip install pandas matplotlib tensorflow torch transformers`


## Data Cleaning and Exploration

The data was pulled from the [IESO](https://www.ieso.ca/Power-Data/Data-Directory) and a scraper (`scrape.py`) was used to pull the data. The data was then cleaned and combined into 
The project uses energy demand and capacity data, which is processed and analyzed in the `eda.ipynb` notebook. The data includes hourly energy demand, capacity, and prices, which are used to train and evaluate the forecasting models.



## Models

### RNN Model

The RNN model is detailed in `rnn_model.ipynb`, where it's trained and evaluated on the energy data. The model's architecture and training logs can be found within the notebook. It utilizes LSTM layers to capture the temporal dependencies in the energy data. The model is compiled with the Adam optimizer and mean squared error loss function. Training involves 100 epochs with a batch size of 32 and includes a validation split to monitor performance on unseen data.

### Linear Regression Model

In addition to the RNN model, a Linear Regression model is explored in `linear_regression.ipynb` for comparison. This model serves as a baseline to evaluate the performance of more complex models like the RNN. It uses features such as demand, capacity, and time to predict energy prices. The model is evaluated using metrics like RMSE (Root Mean Squared Error) and R^2 score to assess its accuracy and fit.
