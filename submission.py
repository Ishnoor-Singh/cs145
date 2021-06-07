# imports
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, metrics


def main():

    train_trendency = pd.read_csv('./data/train_trendency.csv')
    train_vaccine = pd.read_csv('./data/train_vaccine.csv')
    test = pd.read_csv('./data/test.csv')
    submission = pd.read_csv('./data/submission.csv')

    class LinearRegression(object):
        def __init__(self):
            self.coefficients = []

        def fit(self, X, y):
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            X = self._concatenate_ones(X)
            self.coefficients = np.linalg.inv(
                X.transpose().dot(X)).dot(X.transpose()).dot(y)

        def _pred(self, entry):
            #         print(entry.shape)

            b0 = self.coefficients[0]
            beta = self.coefficients[1:]
            prediction = b0
    #         prediction = 0
    #         print(prediction)
            for xi, bi in zip(entry, beta):
                #             print(xi)
                prediction += bi*xi
    #         print(prediction)
            return prediction

        def predict(self, entries):
            return [self._pred(entry) for entry in entries]

        def _concatenate_ones(self, X):
            ones = np.ones(shape=X.shape[0]).reshape(-1, 1)
            return np.concatenate((ones, X), 1)

    class PolynomialRegression(object):

        def __init__(self, degrees=4):

            self._lin = linear_model.LinearRegression()
            self._degrees = degrees

        def fit(self, X, y):
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            self._scaler = MinMaxScaler()
            self._scaler.fit(X)
            X = self._scaler.transform(X)
            poly = PolynomialFeatures(degree=self._degrees)
            X_poly = poly.fit_transform(X)
            self._lin.fit(X_poly, y)

        def predict(self, entry):
            X = entry
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            X = self._scaler.transform(X)
            poly = PolynomialFeatures(degree=self._degrees)
            X_poly = poly.fit_transform(X)
            return self._lin.predict(X_poly)

    class State(object):
        def __init__(self, state, train_trendency, train_vaccine):
            self.name = state
            self._hist_days = 2

            print('creating data for ' + state)
            self._populate_data(state, train_trendency, train_vaccine)
            print('done creating data for ' + state)
            print('training models for ' + state)
            self._train_models()
            print('done training models for ' + state)

        def add_row(self, date):
            print('adding row with date ' + date)

            new_row = []

            prev_data = self.get_prev_data(date)
            X = np.reshape(np.asarray(prev_data, dtype=float), (1, -1))
            new_row.append(self._confirmed_model.predict(X)[0])
            new_row.append(self._deaths_model.predict(X)[0])
            new_row.append(self._recovered_model.predict(X)[0])
            new_row.append(self._active_model.predict(X)[0])
            new_row.append(self._incident_model.predict(X)[0])
            new_row.append(self._ttr_model.predict(X)[0])
            new_row.append(self._cfr_model.predict(X)[0])
            new_row.append(self._tr_model.predict(X)[0])

            for i in range(len(prev_data) - 2*self._hist_days):
                new_row.append(prev_data[i])
            new_row.append(self._tv_model.predict(X)[0])
            new_row.append(self._pfv_model.predict(X)[0])
            for i in range(2*self._hist_days):
                new_row.append(len(prev_data) - 2*self._hist_days + i)
    #         print(new_row)
    #         [confirmed] = self._confirmed_model.predict(np.reshape(self._train_numpy[-1], (1,-1)))
    #         print(self._train_numpy[-1])
            self._label_data.append(new_row)
            return new_row[0], new_row[1]

        def get_prev_data(self, date):
            data = []
            cols = ['Confirmed', 'Deaths', 'Recovered', 'Active', 'Incident_Rate', 'Total_Test_Results',
                    'Case_Fatality_Ratio', 'Testing_Rate', 'total_vaccinations', 'people_fully_vaccinated']
    #         np.reshape(train[-1], (1,-1))
            for index in range(len(cols)):
                col = cols[index]
                for i in range(self._hist_days):
                    data.append(
                        self._label_data[col][self._label_data.shape[0]-1*(i + 1)])
            return data

        def _train_models(self):
            #         self._confirmed = train_data['Confirmed']
            #         self._deaths = train_data['Deaths']
            #         self._recovered = train_data['Recovered']
            #         self._active = train_data['Active']
            #         self._incident = train_data['Incident_Rate']
            #         self._ttr = train_data['Total_Test_Results']
            #         self._cfr = train_data['Case_Fatality_Ratio']
            #         self._tr = train_data['Testing_Rate']
            #         self._tv = train_data['total_vaccinations']
            #         self._pfv = train_data['people_fully_vaccinated']
            self._confirmed_model = PolynomialRegression()
            self._confirmed_model.fit(self._train_numpy, self._confirmed)

            self._deaths_model = PolynomialRegression()
            self._deaths_model.fit(self._train_numpy, self._deaths)

            self._recovered_model = PolynomialRegression()
            self._recovered_model.fit(self._train_numpy, self._recovered)

            self._active_model = PolynomialRegression()
            self._active_model.fit(self._train_numpy, self._active)

            self._incident_model = PolynomialRegression()
            self._incident_model.fit(self._train_numpy, self._incident)

            self._ttr_model = PolynomialRegression()
            self._ttr_model.fit(self._train_numpy, self._ttr)

            self._cfr_model = PolynomialRegression()
            self._cfr_model.fit(self._train_numpy, self._cfr)

            self._tr_model = PolynomialRegression()
            self._tr_model.fit(self._train_numpy, self._tr)

            self._tv_model = PolynomialRegression()
            self._tv_model.fit(self._train_numpy, self._tv)

            self._pfv_model = PolynomialRegression()
            self._pfv_model.fit(self._train_numpy, self._pfv)

        def _populate_data(self, state, train_trendency, train_vaccine):
            trendency_state = train_trendency.loc[train_trendency['Province_State'] == state].copy(
            )
            vaccine_state = train_vaccine.loc[train_vaccine['location'] == state].copy(
            )

            # date conversion
            trendency_state['Date'] = pd.to_datetime(
                trendency_state["Date"], format='%m-%d-%Y')
            vaccine_state['date'] = pd.to_datetime(
                vaccine_state["date"], format='%Y-%m-%d')

            # drop some columns
            trendency_state.drop(
                columns=['Unnamed: 0', 'Province_State'], axis=1, inplace=True)
            vaccine_state.drop(
                columns=['Unnamed: 0', 'location'], axis=1, inplace=True)
            # add shifted columns for past data
            for i in range(self._hist_days):
                vaccine_state['total_vaccinations - ' +
                              str(i+1)] = vaccine_state['total_vaccinations'].shift(i+1)
            for i in range(self._hist_days):

                vaccine_state['people_fully_vaccinated - ' +
                              str(i+1)] = vaccine_state['people_fully_vaccinated'].shift(i+1)

            columns_to_shift = ['Confirmed', 'Deaths', 'Recovered', 'Active',
                                'Incident_Rate', 'Total_Test_Results', 'Case_Fatality_Ratio', 'Testing_Rate']

            for column in columns_to_shift:
                for i in range(self._hist_days):
                    trendency_state[column + ' - ' +
                                    str(i+1)] = trendency_state[column].shift(i+1)
            # replace Nan's with 0
            vaccine_state.fillna(0, inplace=True)
            trendency_state.fillna(0, inplace=True)

            # create training data

        #     print(train_vaccine_data.dtypes)
        #     print(train_data.dtypes)
            train_data = trendency_state
            train_vaccine_data = vaccine_state
            # rename Date to date
            train_data['date'] = train_data['Date']
            train_data.drop(['Date'], axis=1, inplace=True)
            train_data = pd.merge(train_data, train_vaccine_data, how='left')
            train_data.fillna(0, inplace=True)
            self._label_data = train_data.copy()
        #     print(train_data.dtypes)
        #     print (train_data)

            # we will drop date here

            # take out the target columns
            self._confirmed = train_data['Confirmed']
            self._deaths = train_data['Deaths']
            self._recovered = train_data['Recovered']
            self._active = train_data['Active']
            self._incident = train_data['Incident_Rate']
            self._ttr = train_data['Total_Test_Results']
            self._cfr = train_data['Case_Fatality_Ratio']
            self._tr = train_data['Testing_Rate']
            self._tv = train_data['total_vaccinations']
            self._pfv = train_data['people_fully_vaccinated']

            # drop stuff we can't use
        #     train_data = train_data.drop(columns= ['Confirmed', 'Deaths', 'Recovered', 'Active', 'Incident_Rate', 'Total_Test_Results', 'Case_Fatality_Ratio', 'Testing_Rate']
        # , axis= 1)
        #     train_vaccine_data = train_vaccine_data.drop(columns=['total_vaccinations', 'people_fully_vaccinated'], axis=1)
        #     print(train_data)

            train_data.drop(columns=['Confirmed', 'Deaths', 'Recovered', 'Active', 'Incident_Rate', 'Total_Test_Results',
                                     'Case_Fatality_Ratio', 'Testing_Rate', 'total_vaccinations', 'people_fully_vaccinated'], axis=1, inplace=True)
            # keep a copy with date
            self._train_dated = train_data

            train_data.drop(['date'], axis=1, inplace=True)
            self._train_data = train_data
    #         print(train_data.dtypes)
            self._train_numpy = train_data.to_numpy()
            return

    class StatesHandler(object):

        def __init__(self, train_trendency, train_vaccine):
            self._state_dict = {}
            self._vaccine = train_vaccine
            self._trendency = train_trendency
            self._submission = submission

        def new_entry(self, state, date, ID):
            if state not in self._state_dict:
                print(state + ' not found, initiating')
                self._state_dict[state] = State(
                    state, self._trendency, self._vaccine)
            else:
                print(state + ' was found')
            state_obj = self._state_dict[state]
            confirmed, deaths = state_obj.add_row(date)
            print(state, date, str(confirmed), str(deaths))
            submission.at[ID, 'Confirmed'] = int(confirmed)
            submission.at[ID, 'Deaths'] = int(deaths)
            print('\n\n')
            # modify this row

    states = StatesHandler(train_trendency, train_vaccine)

    i = 0
    for row in test.iterrows():

        # if row[1][1] == 'New York':
        #     i = i + 1
        #     print(str(i) + ' of 1500 done. \n\n')
        #     continue
        states.new_entry(row[1][1], row[1][2], row[1][0])
        i = i + 1
        print(str(i) + ' of 1500 done. \n\n')
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
