import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing





class MyModel:
    def __init__(self):
        self.final_enoding={}
        self.model=0
        

        
    def fit(self,files):
        def extract_batters_bowlers(row):
            batters = row['batters']
            batters=str(batters)
            batters=batters[1:]
            batters=batters[:-1]
            batters=batters[1:]
            batters=batters[:-1]
            batters=batters.split("' '")
            for i in range(len(batters)):
                if i >= 5:
                    break
                row[f'batter{i+1}'] = batters[i]
            for i in range(len(batters), 5):
                row[f'batter{i+1}'] = 'NA'
            bowlers = row['bowlers']
            bowlers=str(bowlers)
            bowlers=bowlers[1:]
            bowlers=bowlers[:-1]
            bowlers=bowlers[1:]
            bowlers=bowlers[:-1]
            bowlers=bowlers.split("' '")
            for i in range(len(bowlers)):
                if i >= 4:
                    break
                row[f'bowler{i+1}'] = bowlers[i]
            for i in range(len(bowlers), 4):
                row[f'bowler{i+1}'] = 'NA'
            return row
        df=files[0]
        df2=files[1]
        df=df.loc[df['overs']<6]
        df=df.loc[df['innings']<3]
        df=df.loc[df['ID']!=829763]
        df=df.loc[df['ID']!=501265]
        dt_2 = pd.DataFrame()
        dt_2['total_run'] = df.groupby(['ID', 'innings'])['total_run'].sum().values
        dt_2['batters'] = df.groupby(['ID', 'innings'])['batter'].unique().values
        dt_2['bowlers'] = df.groupby(['ID', 'innings'])['bowler'].unique().values
        dt_3 = pd.DataFrame({
        'ID': df['ID'].repeat(2),
        'innings': np.tile([1, 2], len(df))
        })
        dt_3 = dt_3.drop_duplicates(subset=['ID', 'innings'])
        # reset the index for both dataframes
        dt_2 = dt_2.reset_index(drop=True)
        dt_3 = dt_3.reset_index(drop=True)

        # concatenate the two dataframes horizontally
        merged_df = pd.concat([dt_3, dt_2], axis=1)
        sample_mohit = pd.merge(merged_df, df2[['ID', 'Venue']], on='ID')
        sample_mohit = sample_mohit.apply(extract_batters_bowlers, axis=1)
        sample_mohit = sample_mohit.drop(["batters", "bowlers"], axis=1)
        df = sample_mohit
        df = df.drop(["ID"], axis=1)
        cat_cols = ['Venue', 'batter1', 'batter2', 'batter3', 'batter4', 'batter5', 'bowler1', 'bowler2', 'bowler3', 'bowler4']
        dicti={}
        for i in cat_cols:
            le = preprocessing.LabelEncoder()
            le.fit(df[i])
            df[i]=le.transform(df[i])
            s="mapping_"+i
            s=dict(zip(le.classes_, le.transform(le.classes_)))
            dicti.update(s)

        X = df.drop(["total_run"], axis=1)
        y = df["total_run"]
        # Define a grid of hyperparameters to search over
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}

        # Create a Lasso regression object
        lasso = Lasso()

        # Create a GridSearchCV object
        grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_absolute_error')

        # Fit the GridSearchCV object on the training set
        grid_search.fit(X, y)

        # Get the best hyperparameters and the best model
        best_alpha = grid_search.best_params_['alpha']
        best_model = grid_search.best_estimator_

        self.model=best_model
        self.final_enoding=dicti






        
    def predict(self,test:pd.DataFrame):
        def extract_batters_bowlers2(row):
            batters = row['batsmen']
            batters=batters.split(", ")
            for i in range(len(batters)):
                if i >= 5:
                    break
                row[f'batter{i+1}'] = batters[i]
            for i in range(len(batters), 5):
                row[f'batter{i+1}'] = 'NA'
            bowlers = row['bowlers']
            bowlers=bowlers.split(", ")
            for i in range(len(bowlers)):
                if i >= 4:
                    break
                row[f'bowler{i+1}'] = bowlers[i]
            for i in range(len(bowlers), 4):
                row[f'bowler{i+1}'] = 'NA'
            return row
        test=test.apply(extract_batters_bowlers2,axis=1)
        test=test.drop(['batting_team','bowling_team'],axis=1)
        a=test[0:1]
        b=test[1:2]
        test1=[]
        test1.append(a['innings'][0])
        test1.append(a['venue'][0])
        test1.append(a['batter1'][0])
        test1.append(a['batter2'][0])
        test1.append(a['batter3'][0])
        test1.append(a['batter4'][0])
        test1.append(a['batter5'][0])
        test1.append(a['bowler1'][0])
        test1.append(a['bowler2'][0])
        test1.append(a['bowler3'][0])
        test1.append(a['bowler4'][0])

        test2=[]
        test2.append(b['innings'][1])
        test2.append(b['venue'][1])
        test2.append(b['batter1'][1])
        test2.append(b['batter2'][1])
        test2.append(b['batter3'][1])
        test2.append(b['batter4'][1])
        test2.append(b['batter5'][1])
        test2.append(b['bowler1'][1])
        test2.append(b['bowler2'][1])
        test2.append(b['bowler3'][1])
        test2.append(b['bowler4'][1])
        final_t1=[]
        final_t2=[]
        final_t1.append(test1[0])
        final_t2.append(test2[0])
        for i in test1[1:]:
            try:
                final_t1.append(self.final_enoding[i])
            except:
                final_t1.append(141)
        for i in test2[1:]:
            try:
                final_t2.append(self.final_enoding[i])
            except:
                final_t2.append(141)
        print(final_t1)
        print(final_t2)
        test1=np.array(final_t1)
        test2=np.array(final_t2)



        pred1=self.model.predict([test1])
        pred2=self.model.predict([test2])
        predictions=[pred1[0],pred2[0]]
        print(predictions)
        return predictions
        
        
        







