# Instructions
The code can be run either by running python main.py or by executing the cells in main.ipynb.


# Setup
There are two main configuration files: data/api_data.json and data/cryptos_to_analyse.json.
The first one contains the set up API account for reddit, it is required in order to access
their API. The second one contains a json-like representation of the three coins that will be
analysed. For both files, I have provided some default values: for the api_data.json, I have 
already created an account and an API instance and provided it's data, so you can directly use 
it. for the cryptos_to_analyse.json, I have provided three cryptocurrencies: bitcoin, ethereum and
cardano. You can always change them if you want analysis for different ones.

# Result interpretation
Most of the results are represented visually, and all the relevant explanations can be found in the
report. In short, the code produces some graphs in the visiualization folder. The graphs are mainly 
about sentiment, word frequency, and word count.
