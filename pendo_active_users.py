import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeClassifier

infile_turnboard = '/Users/bradley.webb/Desktop/pendo_active_users/pendo_turn_page_view_03-09-21.csv'
infile_turndetails = '/Users/bradley.webb/Desktop/pendo_active_users/pendo_turn_detail_page_view_03-09-21.csv'
infile_exp_group = '/Users/bradley.webb/Desktop/pendo_active_users/experiment_group_assignment.csv'
infile_activated_users = '/Users/bradley.webb/Desktop/pendo_active_users/Turn_Board_Requests_&_Access_Status_2021_03_09.csv'

df_board = pd.read_csv(infile_turnboard)
df_details = pd.read_csv(infile_turndetails)
df_exp_group = pd.read_csv(infile_exp_group)
df_activated_users  = pd.read_csv(infile_activated_users)

df_board.rename(columns={'Page Views for Maintenance > Unit Turns': 'PAGE_VIEWS',
                         'Time On Page (minutes) for Maintenance > Unit Turns': 'TIME_SPENT'},
                inplace=True)
df_details.rename(columns={'Page Views for Maintenance > Unit Turn Details': 'PAGE_VIEWS',
                           'Time On Page (minutes) for Maintenance > Unit Turn Details': "TIME_SPENT"},
                  inplace=True)
df_board = df_board.join(df_exp_group.set_index('VHOST'), on='name')
df_board = df_board.join(df_activated_users.set_index('EMAIL'), on='email', how='right')


# print(df_exp_group['GROUP'].unique())
# ['control' 'no-onboarding' 'DIY-onboarding']

df_board['GROUP_CATEGORICAL'] = None
df_board.loc[(df_board['GROUP'] == 'no-onboarding'), ['GROUP_CATEGORICAL']] = 1
# df_board.loc[(~df_board['GROUP'].isna()) & (df_board['GROUP'] == 'DIY-onboarding'), ['GROUP_CATEGORICAL']] = 0
df_board.loc[(~df_board['GROUP'].isna()) & (df_board['GROUP'] != 'no-onboarding'), ['GROUP_CATEGORICAL']] = 0
# df_board.loc[(~df_board['GROUP'].isna()) & (df_board['GROUP'] != 'control'), ['GROUP_CATEGORICAL']] = 0

df_board_no_nan = df_board[(df_board['GROUP_CATEGORICAL'] == 1) | (df_board['GROUP_CATEGORICAL'] == 0)]

x = df_board_no_nan[['TIME_SPENT']].to_numpy()
y = df_board_no_nan['GROUP_CATEGORICAL'].to_numpy()

model = LinearRegression().fit(x,y)
r_sq = model.score(x, y)
pd.set_option('display.max_columns', 500)
print(df_board[['GROUP','PAGE_VIEWS', 'TIME_SPENT']].groupby('GROUP').agg(['count', 'mean', 'median', 'std']))
print('coefficient of determination', r_sq)
