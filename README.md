# LSTM_Stock_Predictor
Deep Learning Recurrent Neural Networks

![deep-learning.jpg](Images/deep-learning.jpg)

# Background
Due to the volatility of cryptocurrency speculation, investors will often try to incorporate sentiment from social media and news articles to help guide their trading strategies. One such indicator is the [Crypto Fear and Greed Index (FNG)](https://alternative.me/crypto/fear-and-greed-index/) which attempts to use a variety of data sources to produce a daily FNG value for cryptocurrency. I have been asked to help build and evaluate deep learning models using both the FNG values and simple closing prices to determine if the FNG indicator provides a better signal for cryptocurrencies than the normal closing price data.

In this project, I will use deep learning recurrent neural networks to model bitcoin closing prices. One model will use the FNG indicators to predict the closing price while the second model will use a window of closing prices to predict the nth closing price.

The broad tasks I will perform to complete this analysis:

1. [Prepare the data for training and testing](#prepare-the-data-for-training-and-testing)
2. [Build and train custom LSTM RNNs](#build-and-train-custom-lstm-rnns)
3. [Evaluate the performance of each model](#evaluate-the-performance-of-each-model)

- - -

### Files

[Closing Prices Code Notebook](LSTM_Code/lstm_stock_predictor_closing.ipynb)

[FNG Code Notebook](LSTM_Code/lstm_stock_predictor_fng.ipynb)

- - -

# Tasks Performed

### Prepare the data for training and testing

Created a Jupyter Notebook for each RNN. The code contains a function to create the window of time for the data in each dataset.

For the Fear and Greed model, I used the FNG values to try and predict the closing price. A function was created in the notebook to assist with this.

For the closing price model, I used the previous closing prices to try and predict the next closing price. A function was created in the notebook to help with this.

Each model used 70% of the data for training and 30% of the data for testing.

A MinMaxScaler was used to the X and y values to scale the data for the model.

Finally, a reshape of the X_train and X_test values to fit the model's requirement of samples, time steps, and features. (*example:* `X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))`)

### Build and train custom LSTM RNNs

In each Jupyter Notebook, the same custom LSTM RNN architecture was used. In one notebook, I fit the data using the FNG values. In the second notebook, I fit the data using only closing prices.

I used the same parameters and training steps for each model. This is necessary to compare each model accurately.

# Performance Evaluation of each model

## Analysis:
* Which model had the lowest loss - The Predictive Model using Closing Price had the least "Loss" for both the training set, and more importantly the Test/Predictive set @ Training Loss = 0.0133 / Test Loss = .0487 versus the Fear & Greed Predictive Model @ Training Loss = 0.0428 / Test Loss = .2472
* Determine which model tracks the actual values best over time - The Predictive Model using Closing Price tracks the actual values best over time, which is evident when looking at the two charts showing Real versus Predicitate data¶
* Determine the appropriate Window Size for the model - The optimum Window Size for both the Closing Price and Fear & Greed, with the lease amount of loss, was a seven (7) day window

[Closing Prices Chart](LSTM_Code/Closing_Prices.png)
[Fear & Greed Chart](LSTM_Code/Fear_&_Greed.png)

- - -

### Resources

[Keras Sequential Model Guide](https://keras.io/getting-started/sequential-model-guide/)

[Illustrated Guide to LSTMs](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

[Stanford's RNN Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

