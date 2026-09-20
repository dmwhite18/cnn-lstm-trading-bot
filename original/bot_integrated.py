# -*- coding: utf-8 -*-
# ARCHIVED PROTOTYPE (2022-2025) — original code, preserved for portfolio/reference.
# Hardcoded API credentials have been removed and replaced with environment-variable
# reads (see .env.example). This file is not maintained and is shown as-is.

# -*- coding: utf-8 -*-
"""
Created on Sun May 15 20:09:38 2022

@author: dmwhi
"""

from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score 
import alpaca_trade_api
import datetime
import numpy
# import pandas_datareader
import pickle
import time
import multiprocessing
from multiprocessing import Manager
import random
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)
from datetime import timedelta
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.layers import concatenate, Conv1D, Dense, Embedding, GlobalMaxPooling1D, Input, LeakyReLU, LSTM, SpatialDropout1D, Dropout
from keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.wrappers import Bidirectional
import matplotlib.pyplot as plt
# from pandas_datareader import data as pdr
import yfinance as yf
import math
import copy

# yf.pdr_override()

import os
# Credentials come from the environment (see .env.example). Never hardcode API keys.
key_id = os.environ["ALPACA_KEY_ID"]
secret_key = os.environ["ALPACA_SECRET_KEY"]
# Defaults to the PAPER endpoint; override via env for live.
base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
api_version = "v2"

class trading_bot:
    def __init__(self):
        self.filepath_conv='./conv_model_beta'
        self.filepath_lstm='./lstm_model_beta'
        self.data_source = 'yahoo'
        self.include_content = True
        self.limit = 50
        self.limit2 = 500
        self.maxlen = 244
        self.date_string_1 = '2015-01-01 00:00:00.000000'
        self.date_string_2 = '1970-01-01 00:00:00.000000'
        self.padding = self.truncating = 'pre'
        self.random_state = 42
        self.test_size = 0.15
        self.threshold_1 = 0.6
        self.threshold_2 = 0.9
        self.train_size = .7
        self.validation_split = (1-self.train_size-self.test_size)/(1-self.test_size)
        self.window_1 = 75
        self.window_2 = 94
        self.window_3 = 27
        self.rest = alpaca_trade_api.REST(key_id=key_id, secret_key=secret_key, 
                                          base_url=base_url, 
                                          api_version=api_version)
        self.start_1 = datetime.datetime.fromisoformat(self.date_string_1)
        self.start_2 = datetime.datetime.fromisoformat(self.date_string_2)
        with open('tokenizer2_beta.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.window_init = 1
        self.units_3 = 1
        self.save_best_only = True
        self.monitor_1 = 'val_accuracy'
        self.mode_1 = 'max'
        self.monitor_2 = 'val_loss'
        self.mode_2 = 'min'
        self.shuffle = True
       
    def get_clock(self, self_bought, analyze_flag, first_count, need_selling):
        while analyze_flag.value == 1:
            while True:
                try:
                    is_open = self.rest.get_clock().is_open
                    time.sleep(1)
                    break

                except Exception as error:
                    print(error)
                    time.sleep(1)           

            if is_open:
                if first_count.value == 1:
                    first_count.value = 0

            else:
                if first_count.value == 0:
                    first_count.value = 1
                    self_bought[:] = []
                    need_selling.value = 1
                
    def analyze(self, self_bought, self_buy, self_flag_1, analyze_flag, stocks, thres, bracket_buy, quantity):
        while True:
            try:
                is_open = self.rest.get_clock().is_open
                time.sleep(20)
                break
            
            except Exception as error:
                print(error)
                time.sleep(1)
        
        try:
            print('Open Market:', is_open, '.')

        except Exception as error:
            print(error)

        if is_open:
            conv_model = load_model(filepath=self.filepath_conv)
            lstm_model = load_model(filepath=self.filepath_lstm)
            if self_flag_1.value == 1:
                self_flag_1.value = 0
                end_utc = datetime.datetime.now(datetime.timezone.utc)
                end_date = end_utc.date()
                start = end_date - datetime.timedelta(days=7)
                while True:
                    try:
                        news = self.rest.get_news(symbol='AMZN', start=start, end=end_date)
                        break
                    
                    except Exception as error:
                        print(error)
                        time.sleep(1)
                
                for article in news:
                    symbols = article.symbols
                    for symbol in symbols:
                        if symbol not in stocks:
                            stocks.append(symbol)
                
                stock_temp = stocks[:]
                random.shuffle(stock_temp)
                stocks[:] = stock_temp[:100]
            
            else:
                symbols = stocks[:]
                stocks[:] = []
                for symbol in symbols:
                    end_utc = datetime.datetime.now(datetime.timezone.utc)
                    end_date = end_utc.date()
                    start = end_date - datetime.timedelta(days=7)
                    while True:
                        try:
                            news = self.rest.get_news(symbol=symbol, start=start, end=end_date)
                            break
                        
                        except Exception as error:
                            print(error)
                            time.sleep(1)
                    
                    for article in news:
                        symbols = article.symbols
                        for symbol in symbols:
                            if symbol not in stocks:
                                stocks.append(symbol)
                
                stock_temp = stocks[:]
                random.shuffle(stock_temp)
                stocks[:] = stock_temp[:100]
        
            for stock in stocks:
                while True:
                    try:
                        is_open = self.rest.get_clock().is_open
                        break
        
                    except Exception as error:
                        print(error)
                        time.sleep(1)    
                
                if not is_open:
                    break
        
                if stock.find('USD') >= 0:
                    name_pdr = stock[:stock.find('USD')] + '-' + stock[stock.find('USD'):]
        
                else:
                    name_pdr = stock
        
                while True:
                    try:
                        buying_power = float(self.rest.get_account().regt_buying_power)
                        equity = float(self.rest.get_account().equity)
                        break
        
                    except Exception as error:
                        print(error)
                        time.sleep(1)
        
                attempt = 10
                while attempt:
                    try:
                        # close = pdr.get_data_yahoo(name_pdr)['Close'][-1]
                        close = yf.download(name_pdr)['Close'].iloc[-1][0]
                        proceed = True
                        break
        
                    except Exception as error:
                        print(error)
                        time.sleep(1)
                        attempt -= 1
                        if attempt == 0:
                            proceed = False
                            close = 0
                
                try:
                    print('The current stock being analyzed is', stock, '.  Buying power is', buying_power, 'and the current price of the stock is', close, '.')

                except Exception as error:
                    print(error)
                                        
                if stock not in self_bought and 0.9 * buying_power >= quantity.value * close and proceed and stock != 'MOG' and quantity.value * close >= 1 and close <= 0.01 * equity:
                    end = datetime.datetime.now()
                    start = end - datetime.timedelta(days=int(365*4))
                    attempt = 10
                    while attempt:
                        try:
                            # datareader = pdr.get_data_yahoo(name_pdr, start=start, end=end)
                            datareader = yf.download(name_pdr, start=start, end=end)
                            proceed = True
                            break
        
                        except Exception as error:
                            print(error)
                            time.sleep(1)
                            attempt -= 1
                            if attempt == 0:
                                proceed = False
        
                    if proceed:
                        index = datareader.index            
                        texts = []
                        y = []
                        for element_1 in range(len(index)-self.window_1-self.window_2+1):
                            start = index[element_1].isoformat("T") + "Z"
                            end = index[element_1+self.window_1].isoformat("T") + "Z"
                            while True:
                                try:
                                    news = self.rest.get_news(symbol=stock, start=start, end=end, limit=self.limit, include_content=self.include_content)
                                    break
        
                                except Exception as error:
                                    print(error)
                                    time.sleep(1)
        
                            if datareader.loc[index[element_1+self.window_1+self.window_2-1]]['Open'][0] > datareader.loc[index[element_1+self.window_1-1]]['Close'][0]:
                                for article in news:
                                    if hasattr(article, 'content'):
                                        texts.append(article.content)
                                        y.append(1)
        
                            else:
                                for article in news:
                                    if hasattr(article, 'content'):
                                        texts.append(article.content)
                                        y.append(0)    
        
                        X_1 = self.tokenizer.texts_to_sequences(texts=texts)            
                        if X_1 and y:
                            y = numpy.array(y)      
                            X_1 = pad_sequences(sequences=X_1, maxlen=self.maxlen, padding=self.padding, truncating=self.truncating)
                            y_predict = conv_model.predict(X_1)
                            if len(numpy.unique(y)) == 1:
                                if numpy.unique(y) == 0:
                                    y = numpy.append(y, 1)
                                    y_predict = numpy.append(y_predict, 1)
        
                                elif numpy.unique(y) == 1:
                                    y = numpy.append(y, 0)
                                    y_predict = numpy.append(y_predict, 0)
        
                            try:
                                score = roc_auc_score(y, y_predict)
        
                            except Exception as error:
                                print(error)
                                score = 0
                            
                            try:
                                print('The news accuracy score of', stock, 'is', score, '.  The threshold to pass is', self.threshold_1, '.')
                                
                            except Exception as error:
                                print(error)
                                    
                            if score >= self.threshold_1:
                                index = datareader.index[-self.window_1:]
                                if not index.empty:
                                    start = index[0] + datetime.timedelta(days=1)
                                    start = start.isoformat("T") + "Z"
                                    while True:
                                        try:
                                            news = self.rest.get_news(symbol=stock, start=start, end=end_date+datetime.timedelta(days=1), limit=self.limit, include_content=self.include_content)
                                            break
        
                                        except Exception as error:
                                            print(error)
                                            time.sleep(1)
        
                                    texts = []
                                    for element_2 in news:
                                        if hasattr(element_2, 'content'):
                                            texts.append(element_2.content)
        
                                    X_1 = self.tokenizer.texts_to_sequences(texts=texts) 
                                    X_1 = pad_sequences(sequences=X_1, maxlen=self.maxlen, padding=self.padding, truncating=self.truncating)
                                    try:
                                        y_predicts = conv_model.predict(X_1)
                                        y_predition = True
        
                                    except Exception:
                                        y_predition = False
        
                                    if y_predition:
                                        y_predicts_ave = sum(y_predicts) / len(y_predicts)
                                        try:
                                            print('The news score of', stock, 'is', y_predicts_ave[0], '.  The threshold to pass is', thres.value, '.')

                                        except Exception as error:
                                            print(error)

                                        if y_predicts_ave >= thres.value:
                                            scaler_x = MinMaxScaler()
                                            scaler_y = MinMaxScaler()
                                            if not datareader.values.size == 0:
                                                scaled_x_data = scaler_x.fit_transform(datareader.values[:-1])
                                                scaled_y_data = scaler_y.fit_transform(datareader['Open'].values.reshape(-1, 1)[:-1])            
                                                X_2 = []
                                                y_1 = []
                                                for element_1 in range(len(datareader.index)-self.window_3-2):
                                                    X_2.append(scaled_x_data[element_1:element_1+self.window_3])
                                                    y_1.append(scaled_y_data[element_1+self.window_3+1])                  
                              
                                                X_2, y_1 = numpy.array(X_2), numpy.array(y_1)        
                                                X_2 = X_2.reshape(X_2.shape[0], X_2.shape[1]*X_2.shape[2], 1)
                                                y_1_predict = lstm_model.predict(X_2)
                                                score = r2_score(y_true=y_1, y_pred=y_1_predict)
                                                try:
                                                    print('The price accuracy score of', stock, 'is', score, '.  The threshold to pass is', self.threshold_2, '.')

                                                except Exception as error:
                                                    print(error)

                                                if score >= self.threshold_2:
                                                    scaler_x = MinMaxScaler()
                                                    scaler_y = MinMaxScaler()
                                                    scaled_x_data = scaler_x.fit_transform(datareader[-self.window_3-1:-1].values)
                                                    scaled_y_data = scaler_y.fit_transform(datareader[-self.window_3-1:-1]['Open'].values.reshape(-1, 1))            
                                                    X_2 = numpy.array(scaled_x_data)        
                                                    X_2 = X_2.reshape(1, X_2.shape[0]*X_2.shape[1], 1)
                                                    y_1_predict_2 = lstm_model.predict(X_2)
                                                    y_1_predict_2 = scaler_y.inverse_transform(y_1_predict_2.reshape(-1, 1))
                                                    attempt = 10
                                                    while attempt:
                                                        try:
                                                            # close = pdr.get_data_yahoo(name_pdr)['Close'][-1]
                                                            close = yf.download(name_pdr)['Close'].iloc[-1][0]
                                                            proceed = True
                                                            break
        
                                                        except Exception as error:
                                                            print(error)
                                                            time.sleep(1)
                                                            attempt -= 1
                                                            if attempt == 0:
                                                                proceed = False
                                                                close = 0
                                                    
                                                    try:
                                                        print('Predicted next day opening price of', stock, 'is', y_1_predict_2[0][0], '.  The current price is', close, '.')

                                                    except Exception as error:
                                                        print(error)
                                                        
                                                    if y_1_predict_2 > close and proceed:
                                                        while True:
                                                            try:
                                                                is_open = self.rest.get_clock().is_open
                                                                break
        
                                                            except Exception as error:
                                                                print(error)
                                                                time.sleep(1)
                                                        
                                                        if is_open:
                                                            while True:
                                                                try:
                                                                    account = self.rest.get_account()
                                                                    break

                                                                except Exception as error:
                                                                    print(error)
                                                                    time.sleep(1)

                                                            if float(account.equity)>=26000:
                                                                bracket_buy[stock] = y_1_predict_2[0,0]

                                                            else:
                                                                self_buy.append(stock)

                                                            try:
                                                                print('The stock', stock, 'has been queued for procurement!')

                                                            except Exception as error:
                                                                print(error)
        
        analyze_flag.value = 0
                                    
    def selling(self, need_selling, pause_buying, analyze_flag, quantity, blacklist, pending_sell):
        while analyze_flag.value == 1:
            if int(datetime.datetime.now().strftime("%H"))==0 and need_selling.value == 1 or pending_sell.value:
                need_selling.value = 0
                allow = 1
                if pending_sell.value:
                    while True:
                        try:
                            clock = self.rest.get_clock()
                            time.sleep(1)
                            break
        
                        except Exception as error:
                            print(error)
                            time.sleep(1)
                            
                    if (clock.next_open - clock.timestamp).total_seconds() < 3600:
                        pending_sell.value = 0
                        with open('pending_sell.txt', 'w') as handle:
                            handle.write(str(numpy.sum(pending_sell.value)))                
                        
                    else:
                        allow = 0

                if allow:                    
                    while True:
                        try:
                            positions = self.rest.list_positions()
                            break
    
                        except Exception as error:
                            print(error)
                            time.sleep(1)
    
                    symbols = []
                    avg_entry_prices = []
                    qtys = []
                    exchanges = []
                    for position in positions:
                        symbols.append(position.symbol)
                        avg_entry_prices.append(position.avg_entry_price)
                        qtys.append(position.qty)
                        exchanges.append(position.exchange)
    
                    pause_buying.value = 1
                    while symbols:
                        if len(symbols) > self.limit2:
                            symbols_short = symbols[:self.limit2]
                            avg_entry_prices_short = avg_entry_prices[:self.limit2]
                            qtys_short = qtys[:self.limit2]
                            exchanges_short = exchanges[:self.limit2]
                            while True:
                                try:
                                    open_orders = self.rest.list_orders(limit=self.limit2, symbols=symbols_short)
                                    break
    
                                except Exception as error:
                                    print(error)
                                    time.sleep(1)
    
                            order_symbol = []
                            for open_order in open_orders:
                                order_symbol.append(open_order.symbol)
    
                            for index in range(len(symbols_short)):
                                if symbols_short[index] not in order_symbol and symbols_short[index] not in blacklist:
                                    if exchanges_short[index] == 'OTC':
                                        attempt = 10
                                        while attempt:
                                            try:
                                                self.rest.submit_order(symbol=symbols_short[index], qty=str(math.floor(float(qtys_short[index]))), side="sell", type="limit", time_in_force="day", limit_price=math.ceil(float(avg_entry_prices_short[index])*100)/100)
                                                break
        
                                            except Exception as error:
                                                print(error)
                                                time.sleep(1)
                                                attempt -= 1
    
                                    else:
                                        attempt = 10
                                        while attempt:
                                            try:
                                                self.rest.submit_order(symbol=symbols_short[index], qty=str(math.floor(float(qtys_short[index]))), side="sell", type="limit", time_in_force="gtc", limit_price=math.ceil(float(avg_entry_prices_short[index])*100)/100)
                                                break
        
                                            except Exception as error:
                                                print(error)
                                                time.sleep(1)
                                                attempt -= 1
    
                            del symbols[:self.limit2]
                            del avg_entry_prices[:self.limit2]
                            del qtys[:self.limit2]
                            del exchanges[:self.limit2]
    
                        elif symbols:
                            while True:
                                try:
                                    open_orders = self.rest.list_orders(limit=self.limit2, symbols=symbols)
                                    break
    
                                except Exception as error:
                                    print(error)
                                    time.sleep(1)
    
                            order_symbol = []
                            for open_order in open_orders:
                                order_symbol.append(open_order.symbol)
    
                            for index in range(len(symbols)):
                                if symbols[index] not in order_symbol and symbols[index] not in blacklist:
                                    if exchanges[index] == 'OTC':
                                        attempt = 10
                                        while attempt:
                                            try:
                                                self.rest.submit_order(symbol=symbols[index], qty=str(math.floor(float(qtys[index]))), side="sell", type="limit", time_in_force="day", limit_price=math.ceil(float(avg_entry_prices[index])*100)/100)
                                                break
        
                                            except Exception as error:
                                                print(error)
                                                time.sleep(1)
                                                attempt -= 1
                                                
                                    else:
                                        attempt = 10
                                        while attempt:
                                            try:
                                                self.rest.submit_order(symbol=symbols[index], qty=str(math.floor(float(qtys[index]))), side="sell", type="limit", time_in_force="gtc", limit_price=math.ceil(float(avg_entry_prices[index])*100)/100)
                                                break
        
                                            except Exception as error:
                                                print(error)
                                                time.sleep(1)
                                                attempt -= 1
                                        
                            symbols = []
                    
                    while True:
                        try:
                            cash = float(self.rest.get_account().cash)
                            break
                        
                        except Exception as error:
                            print(error)
                            time.sleep(1)
                    
                    if cash < 0 and quantity.value != 1:
                        quantity.value = 1
                        with open('quantity.txt', 'w') as handle:
                            handle.write(str(numpy.sum(quantity.value)))                
                        
                    pause_buying.value = 0

    def buying(self, self_bought, self_buy, analyze_flag, pause_buying, bracket_buy, quantity):
        while analyze_flag.value == 1:
            if self_buy and pause_buying.value == 0:
                to_buy = self_buy[:]
                self_buy[:] = []
                for symbol in to_buy:
                    while True:
                        try:
                            open_orders = self.rest.list_orders(limit=self.limit2, symbols=[symbol])
                            break

                        except Exception as error:
                            print(error)
                            time.sleep(1)

                    for open_order in open_orders:
                        while True:
                            try:
                                self.rest.cancel_order(order_id=open_order.id)
                                break

                            except Exception as error:
                                print(error)
                                time.sleep(1)

                    attempt = 10
                    while attempt:
                        try:
                            self.rest.submit_order(symbol=symbol, qty=quantity.value, time_in_force='gtc')
                            self_bought.append(symbol)
                            break

                        except Exception as error:
                            print(error)
                            time.sleep(1)
                            attempt -= 1

            if bracket_buy and pause_buying.value == 0:
                to_bracket_buy = copy.deepcopy(bracket_buy)
                bracket_buy.clear()
                for key,value in to_bracket_buy.items():
                    attempt = 10
                    while attempt:
                        try:
                            self.rest.submit_order(symbol=key, qty=quantity.value, time_in_force='gtc', side='buy', type='market', order_class='bracket', stop_loss={'stop_price':0.02, 'limit_price':0.01}, take_profit={'limit_price':math.floor(value*100)/100})
                            break

                        except Exception as error:
                            print(error)
                            time.sleep(1)
                            attempt -= 1
                        
    def nn_cl_bo1(self, activation_1, activation_2, batch_size, epochs, filters_1, filters_2, filters_3, kernel_size_1, kernel_size_2, kernel_size_3, 
                  learning_rate, maxlen, num_words, optimizer, output_dim, 
                  padding_truncating, units_1, window, window2, rate, rate_2, names, thres):
        optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 
                      'Nadam', 'Ftrl','SGD']
        optimizerD= {'Adam':Adam(learning_rate=learning_rate), 
                      'SGD':SGD(learning_rate=learning_rate), 
                      'RMSprop':RMSprop(learning_rate=learning_rate), 
                      'Adadelta':Adadelta(learning_rate=learning_rate), 
                      'Adagrad':Adagrad(learning_rate=learning_rate), 
                      'Adamax':Adamax(learning_rate=learning_rate), 
                      'Nadam':Nadam(learning_rate=learning_rate), 
                      'Ftrl':Ftrl(learning_rate=learning_rate)}
        activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu', 
                       'elu', 'exponential', LeakyReLU]
        padding_truncatingL = ['pre', 'post']        
        batch_size = 2 ** round(batch_size)
        end_1 = datetime.datetime.now()
        epochs = round(epochs)
        filters_1 = 2 ** round(filters_1)
        filters_2 = 2 ** round(filters_2)
        filters_3 = 2 ** round(filters_3)
        kernel_size_1 = round(kernel_size_1)
        kernel_size_2 = round(kernel_size_2)
        kernel_size_3 = round(kernel_size_3)
        maxlen = round(maxlen)
        output_dim = 2 ** round(output_dim)
        start_1 = end_1 - timedelta(days=int(365*4))
        units_1 = 2 ** round(units_1)
        window = round(window)
        window2 = round(window2)    
        activation_1 = activationL[round(activation_1)]
        activation_2 = activationL[round(activation_2)]
        optimizer = optimizerD[optimizerL[round(optimizer)]]
        padding = truncating = padding_truncatingL[round(padding_truncating)]
        tokenizer = Tokenizer()        
        texts_init = []
        for name in names:
            attempt = 10
            while attempt:
                try:
                    if name.find('USD') >= 0:
                        name_446 = name[:name.find('USD')] + '-' + name[name.find('USD'):]
                    else:
                        name_446 = name
                    # pandas_datareader_datareader = pdr.get_data_yahoo(name_446, 
                    #                                                             start=start_1, 
                    #                                                             end=end_1)
                    pandas_datareader_datareader = yf.download(name_446, 
                                                                                start=start_1, 
                                                                                end=end_1)
                    # pandas_datareader_datareader = pandas_datareader.DataReader(name=name_446, 
                    #                                                             data_source=self.data_source, 
                    #                                                             start=start_1, 
                    #                                                             end=end_1)
                    break
                except Exception as error:
                    print(error)
                    time.sleep(1)
                    attempt -= 1
            pandas_datareader_datareader_index = pandas_datareader_datareader.index        
            for element_1 in range(len(pandas_datareader_datareader.index)-self.window_init-1):
                start_2 = pandas_datareader_datareader_index[element_1].isoformat("T") + "Z"
                end_2 = pandas_datareader_datareader_index[element_1+self.window_init].isoformat("T") + "Z"
                while True:
                    try:
                        alpaca_trade_api_rest_get_news = self.rest.get_news(
                            symbol=name, start=start_2, end=end_2, limit=self.limit, 
                            include_content=self.include_content)
                        break
                    except Exception as error:
                        print(error)
                        time.sleep(1)
                for element_2 in alpaca_trade_api_rest_get_news:
                    if hasattr(element_2, 'content'):
                        texts_init.append(element_2.content)
        tokenizer.fit_on_texts(texts=texts_init)
        num_words_init = len(tokenizer.word_index) + 1
        num_words = round(num_words_init * num_words)
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(texts=texts_init)       
        flag_1 = True
        for name in names:
            attempt = 10
            while attempt:
                try:
                    if name.find('USD') >= 0:
                        name_483 = name[:name.find('USD')] + '-' + name[name.find('USD'):]
                    else:
                        name_483 = name
                    # pandas_datareader_datareader = pdr.get_data_yahoo(name_483, 
                    #                                                             start=start_1, 
                    #                                                             end=end_1)
                    pandas_datareader_datareader = yf.download(name_483, 
                                                                                start=start_1, 
                                                                                end=end_1)
                    # pandas_datareader_datareader = pandas_datareader.DataReader(name=name_483, 
                    #                                                             data_source=self.data_source, 
                    #                                                             start=start_1, 
                    #                                                             end=end_1)
                    break
                except Exception as error:
                    print(error)
                    time.sleep(1)
                    attempt -= 1    
            pandas_datareader_datareader_index = pandas_datareader_datareader.index
            texts = []
            y = []
            for element_1 in range(len(pandas_datareader_datareader.index)-window-window2+1):
                start_2 = pandas_datareader_datareader_index[element_1].isoformat("T") + "Z"
                end_2 = pandas_datareader_datareader_index[element_1+window].isoformat("T") + "Z"
                while True:
                    try:
                        alpaca_trade_api_rest_get_news = self.rest.get_news(
                            symbol=name, start=start_2, end=end_2, limit=self.limit, 
                            include_content=self.include_content)
                        break
                    except Exception as error:
                        print(error)
                        time.sleep(1)
                if pandas_datareader_datareader.loc[pandas_datareader_datareader_index[element_1+window+window2-1]]['Open'][0] > pandas_datareader_datareader.loc[pandas_datareader_datareader_index[element_1+window-1]]['Close'][0]:
                    for article in alpaca_trade_api_rest_get_news:
                        if hasattr(article, 'content'):
                            texts.append(article.content)
                            y.append(1)
                else:
                    for article in alpaca_trade_api_rest_get_news:
                        if hasattr(article, 'content'):
                            texts.append(article.content)
                            y.append(0)
            X_1 = tokenizer.texts_to_sequences(texts=texts)            
            if X_1 and y:
                X_train, X_valid, y_train, y_valid = train_test_split(X_1, y, 
                                                                                 test_size=self.test_size, 
                                                                                 random_state=self.random_state)
                y_train, y_valid = numpy.array(y_train), numpy.array(y_valid)        
            
                X_train = pad_sequences(sequences=X_train, maxlen=maxlen, padding=padding, truncating=truncating)
                X_valid = pad_sequences(sequences=X_valid, maxlen=maxlen, padding=padding, truncating=truncating)
                if flag_1:
                    flag_1 = False
                    temp_1 = X_train
                    temp_2 = X_valid
                    temp_5 = y_train
                    temp_6 = y_valid
                else:
                    temp_1 = numpy.append(temp_1, X_train, axis=0)
                    temp_2 = numpy.append(temp_2, X_valid, axis=0)
                    temp_5 = numpy.append(temp_5, y_train, axis=0)
                    temp_6 = numpy.append(temp_6, y_valid, axis=0)        
        X_train = temp_1
        X_valid = temp_2
        y_train = temp_5
        y_valid = temp_6    
        def nn_cl_fun():
            shape=(X_train.shape[1],)
            input_1 = Input(shape=shape)         
            embedding = Embedding(input_dim=num_words, 
                                  output_dim=output_dim)(input_1)
            spatialdropout1d = SpatialDropout1D(rate=rate)(embedding)
            conv1d_1 = Conv1D(filters=filters_1, kernel_size=kernel_size_1, 
                              activation=activation_1)(spatialdropout1d)
            globalmaxpooling1d_1 = GlobalMaxPooling1D()(conv1d_1)        
            conv1d_2 = Conv1D(filters=filters_2, kernel_size=kernel_size_2, 
                              activation=activation_1)(spatialdropout1d)
            globalmaxpooling1d_2 = GlobalMaxPooling1D()(conv1d_2)        
            conv1d_3 = Conv1D(filters=filters_3, kernel_size=kernel_size_3, 
                              activation=activation_1)(spatialdropout1d)
            globalmaxpooling1d_3 = GlobalMaxPooling1D()(conv1d_3)        
            concatenate_2 = concatenate([globalmaxpooling1d_1, 
                                         globalmaxpooling1d_2,
                                         globalmaxpooling1d_3])        
            dense_1 = Dense(units=units_1, activation=activation_1)(concatenate_2)
            dropout_1 = Dropout(rate=rate_2)(dense_1)
            dense_2 = Dense(units=units_1/4, activation=activation_1)(dropout_1)
            dropout_2 = Dropout(rate=rate_2)(dense_2)
            dense_3 = Dense(units=self.units_3, activation=activation_2)(dropout_2)
            nn = Model(inputs=input_1, outputs=dense_3)                
            nn.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            return nn    
        model = nn_cl_fun()
        callbacks = [ModelCheckpoint(filepath=self.filepath_conv, 
                                     save_best_only=self.save_best_only, 
                                     monitor=self.monitor_1, mode=self.mode_1)]
        model.fit(x=X_train, y=y_train, callbacks=callbacks, 
                  validation_split=self.validation_split, shuffle=self.shuffle, 
                  batch_size=batch_size, epochs=epochs)    
        model = load_model(filepath=self.filepath_conv)
        y_hat = model.predict(X_valid)
        x = plt.hist(y_hat)
        x_sum = sum(x[0])
        num = 0
        for ind in x[0]:
            num += ind
            if num/x_sum >= 0.5:
                thres.value = x[1][numpy.where(x[0]==ind)] * 0.8
                with open('thres.txt', 'w') as handle:
                    handle.write(str(numpy.sum(thres.value)))                
                break
        with open('tokenizer2_beta.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def nn_cl_bo2(self, activation_1, 
                  batch_size, epochs, 
                  learning_rate, 
                  optimizer, 
                  units_1, units_2, window, dropout, names):
        activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                        'elu', 'exponential', LeakyReLU]
        batch_size = 2 ** round(batch_size)
        end_1 = datetime.datetime.now()
        epochs = round(epochs)
        optimizerD= {'Adam':Adam(learning_rate=learning_rate), 'SGD':SGD(learning_rate=learning_rate),
                      'RMSprop':RMSprop(learning_rate=learning_rate), 'Adadelta':Adadelta(learning_rate=learning_rate),
                      'Adagrad':Adagrad(learning_rate=learning_rate), 'Adamax':Adamax(learning_rate=learning_rate),
                      'Nadam':Nadam(learning_rate=learning_rate), 'Ftrl':Ftrl(learning_rate=learning_rate)}
        optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
        start_1 = end_1 - timedelta(days=int(365*4))
        units_1 = 2 ** round(units_1)
        units_2 = 2 ** round(units_2)
        window = round(window)
        activation_1 = activationL[round(activation_1)]
        optimizer = optimizerD[optimizerL[round(optimizer)]]
        flag_1 = True
        for name in names:
            attempt = 10
            while attempt:
                try:
                    if name.find('USD') >= 0:
                        name_621 = name[:name.find('USD')] + '-' + name[name.find('USD'):]
                    else:
                        name_621 = name
                    # pandas_datareader_datareader = pdr.get_data_yahoo(name_621, 
                    #                                                             start=start_1, 
                    #                                                             end=end_1)
                    pandas_datareader_datareader = yf.download(name_621, 
                                                                                start=start_1, 
                                                                                end=end_1)
                    # pandas_datareader_datareader = pandas_datareader.DataReader(name=name_621, 
                    #                                                             data_source=self.data_source, 
                    #                                                             start=start_1, 
                    #                                                             end=end_1)
                    break
                except Exception as error:
                    attempt -= 1
                    print(error)
                    time.sleep(1)
            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()
            if pandas_datareader_datareader.values.size != 0:
                scaled_x_data = scaler_x.fit_transform(pandas_datareader_datareader.values)
                scaled_y_data = scaler_y.fit_transform(
                    pandas_datareader_datareader['Open'].values.reshape(-1, 1))            
                X_2 = []
                y_1 = []
                for element_1 in range(len(pandas_datareader_datareader.index)-window-1):
                    X_2.append(scaled_x_data[element_1:element_1+window])
                    y_1.append(scaled_y_data[element_1+window+1])                  
                X_2, y_1 = numpy.array(X_2), numpy.array(y_1)        
                X_train_2, X_test_2, y_train_2, y_true = train_test_split(X_2, y_1, 
                                                                          test_size=self.test_size, 
                                                                          random_state=self.random_state)
                X_test_2 = X_test_2.reshape(X_test_2.shape[0], 
                                            X_test_2.shape[1]*X_test_2.shape[2], 1)
                X_train_2 = X_train_2.reshape(X_train_2.shape[0], 
                                              X_train_2.shape[1]*X_train_2.shape[2], 1)
                if flag_1:
                    flag_1 = False
                    temp_3 = X_test_2
                    temp_4 = X_train_2
                    temp_5 = y_train_2
                    temp_6 = y_true
                else:
                    temp_3 = numpy.append(temp_3, X_test_2, axis=0)
                    temp_4 = numpy.append(temp_4, X_train_2, axis=0)
                    temp_5 = numpy.append(temp_5, y_train_2, axis=0)
                    temp_6 = numpy.append(temp_6, y_true, axis=0)
        X_test_2 = temp_3
        X_train_2 = temp_4
        y_train_2 = temp_5
        y_true = temp_6
        def nn_cl_fun():
            nn = Sequential()
            nn.add(Input(shape=(X_train_2.shape[1], 1)))
            nn.add(Bidirectional(LSTM(units=units_1, dropout=dropout, return_sequences=True)))           
            nn.add(Bidirectional(LSTM(units=units_2, dropout=dropout)))
            nn.add(Dense(self.units_3, activation=activation_1))           
            nn.compile(loss='mean_absolute_error', optimizer=optimizer)
            return nn
        model = nn_cl_fun()
        callbacks = [ModelCheckpoint(filepath=self.filepath_lstm, 
                                     save_best_only=self.save_best_only, 
                                     monitor=self.monitor_2, mode=self.mode_2)]
        model.fit(x=X_train_2, y=y_train_2, callbacks=callbacks, 
                  validation_split=self.validation_split, shuffle=self.shuffle, 
                  batch_size=batch_size, epochs=epochs)    
        model = load_model(filepath=self.filepath_lstm)

if __name__ == '__main__':   
    manager = Manager()             
    self_bought = manager.list()
    self_buy = manager.list()
    stocks = manager.list()
    blacklist = manager.list(['CUROQ', 'NOGNQ'])
    self_flag_1 = manager.Value('i', 1)
    analyze_flag = manager.Value('i', 0)
    first_count = manager.Value('i', 1)
    need_selling = manager.Value('i', 1)
    pause_buying = manager.Value('i', 0)
    with open('thres.txt', 'r') as handle:
        temp_num = float(handle.read())    
    with open('quantity.txt', 'r') as handle:
        quantity = float(handle.read())    
    with open('pending_sell.txt', 'r') as handle:
        pending_sell = float(handle.read())    
    thres = manager.Value('f', temp_num)
    quantity = manager.Value('f', quantity)
    pending_sell = manager.Value('f', pending_sell)
    bracket_buy = manager.dict()
    bot = trading_bot()
    need_build = False
    activation_1 = 3.6255964994681715
    activation_2 = 1.9712091728092112
    batch_size = 4.704323427845867
    epochs = 100
    filters_1 = 4.848595944960158
    filters_2 = 4.848595944960158
    filters_3 = 4.848595944960158
    kernel_size_1 = 3.601169794991689
    kernel_size_2 = 7.22585944241003
    kernel_size_3 = 4.689705536453457
    learning_rate = 0.1263623710979251
    num_words = 0.6253804126545683
    maxlen = 243.62737912616177
    optimizer = 4.05527115171985
    output_dim = 3.924736431609251
    padding_truncating = 0.8440738232752986
    units_1 = 3.6132466326907346
    window = 75.11363934368919
    window2 = 94.1993060347709
    rate = 0.17395415736095649
    rate_2 = 0.13862255760779651
    activation_3 = 4.987266138202758
    batch_size_2 = 6.4267546517985075
    learning_rate_2 = 1.0
    optimizer_2 = 6.833145804158623
    window_3 = 26.69199633603636
    units_3 = 3.4330423695981533
    units_2 = 2.7739760044284583
    dropout = 0.0
    modeling_not_needed = True
    while True:
        if modeling_not_needed:
            analyze_flag.value = 1
            analysis = multiprocessing.Process(target=bot.analyze, args=(self_bought, self_buy, self_flag_1, analyze_flag, stocks, thres, bracket_buy, quantity))
            clock = multiprocessing.Process(target=bot.get_clock, args=(self_bought, analyze_flag, first_count, need_selling))
            buy = multiprocessing.Process(target=bot.buying, args=(self_bought, self_buy, analyze_flag, pause_buying, bracket_buy, quantity))
            sell = multiprocessing.Process(target=bot.selling, args=(need_selling, pause_buying, analyze_flag, quantity, blacklist, pending_sell))
            analysis.start()
            clock.start()
            buy.start()
            sell.start()
            analysis.join()
            clock.join()
            buy.join()
            sell.join()
        if (datetime.datetime.today().weekday() == 5 and need_build) or not modeling_not_needed:
            modeling_not_needed = True
            need_build = False
            now = datetime.datetime.now(datetime.timezone.utc)
            end = now.date()
            start = end - datetime.timedelta(days=7)
            while True:
                try:
                    get_news = bot.rest.get_news(symbol='AMZN', start=start, end=end)
                    break
                except Exception as error:
                    print(error)
                    time.sleep(1)
            make_break = False
            names = []
            for news in get_news:
                symbols = news.symbols
                for symbol in symbols:
                    if symbol not in names:
                        names.append(symbol)
                #     if len(names) >= 100:
                #         make_break = True
                #         break
                # if make_break:
                #     break
            news_model = multiprocessing.Process(target=bot.nn_cl_bo1, args=(activation_1, activation_2, batch_size, epochs, filters_1, filters_2, filters_3, kernel_size_1, kernel_size_2, kernel_size_3, learning_rate, maxlen, num_words, optimizer, output_dim, padding_truncating, units_1, window, window2, rate, rate_2, names, thres))
            news_model.start()
            news_model.join()
            with open('tokenizer2_beta.pickle', 'rb') as handle:
                bot.tokenizer = pickle.load(handle)
            price_model = multiprocessing.Process(target=bot.nn_cl_bo2, args=(activation_3, batch_size_2, epochs, learning_rate_2, optimizer_2, units_3, units_2, window_3, dropout, names))
            price_model.start()
            price_model.join()
            while True:
                try:
                    buying_power = float(bot.rest.get_account().regt_buying_power)
                    equity = float(bot.rest.get_account().equity)
                    multiplier = float(bot.rest.get_account().multiplier)
                    if multiplier == 4:
                        multiplier = 2
                    break
    
                except Exception as error:
                    print(error)
                    time.sleep(1)
            
            qty_temp = equity * multiplier // (equity * multiplier - buying_power)
            
            if qty_temp == 1:
                quantity.value = 1
                while True:
                    try:
                        bot.rest.cancel_all_orders()
                        break
                    
                    except Exception as error:
                        print(error)
                        time.sleep(1)
                        
                need_selling.value = 1
                pending_sell.value = 1
                with open('pending_sell.txt', 'w') as handle:
                    handle.write(str(numpy.sum(pending_sell.value)))                

            elif qty_temp > quantity.value:
                quantity.value += 1

            elif qty_temp < quantity.value:
                quantity.value -= 1

            else:
                quantity.value = quantity.value
                
            with open('quantity.txt', 'w') as handle:
                handle.write(str(numpy.sum(quantity.value)))                
            
        elif datetime.datetime.today().weekday() != 5 and not need_build:
            need_build = True
            