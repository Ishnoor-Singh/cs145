# Read me

## Instructions for running
```sh
# assumes you have a folder called data with test.csv, submission.csv, train_trendency.csv and train_vaccine.csv
# should have python3
pip install pandas
pip install sklearn
pip install numpy 
python submission.py
# will creat file called submission.csv
```
I have set hist_days to 2 by default since that way it takes about 20 mins to complete, our final submission was with it being 5 which took about 2.5 hours. I also ran with it being 7 which took upwards of 5 hours.

## Explanation

Model is an Autoregression that use a parameter called hist_days to set how many previous days of data to use to make predictions.

The general mechanism is as follows:
- once we get a date and state pair to return results for check if models exist for the state
- if they do, run models to make prediction
  - this includes creating a row entry with columns for all the data from the `hist_days` number of days
  - run models to get all the data for current day. this includes deaths, confirmed, vaccine numbers, all number columns provided to us essentially
- if the models are not trained, train the models:
  - this includes creating models capable of predicting all the paramaters for that day based on all the parameters from the last `hist_days` days

### Model
The model itself uses Polynomial Regression which is implemented using Linear regression from sklearn.
> I had implemented linear regression myself and used the close-form solution however, I seem to running into singular matrices or some of the models. Thus i swapped it out with linear regression from sklearn. I also experimented with the gradient decent formation, however, I was facing some issues.

The paramaters for the model are all the numerical parameters from the two csvs, each parameter appears `hist_days` times and occupies its own column

by default my polynomial regression class uses degrees=4 and all models run with that

## Major issues:
### Linear Regression
I had a lot of trouble first with singular matrices and then with normalization in linear regression

### New York
New York does not seem to appear in the vaccine data file and that causes a lot of errors. I do a left join on the data and that lead to some NaN's I wasnt expecting.