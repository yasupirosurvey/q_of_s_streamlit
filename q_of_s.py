#基本ライブラリ
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#データセットの読み込み
df = pd.read_csv('./sample.csv')

#目標値を数値にラベルづけ
df.loc[df['quality of sleep'] == 0, 'quality of sleep'] = 'いまいち'
df.loc[df['quality of sleep'] == 1, 'quality of sleep'] = 'わるくない'
df.loc[df['quality of sleep'] == 2, 'quality of sleep'] = 'そこそこよい'

#学習用データとテストデータを切り分け
x = df.drop('quality of sleep', axis=1).values
t = df['quality of sleep'].values
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

#入力変数の標準化
std_scaler = StandardScaler()
std_scaler.fit(x_train)

#データセットの変換
x_train_std = std_scaler.transform(x_train)
x_test_std = std_scaler.transform(x_test)

#予測モデル
svc = SVC(C=2)
svc.fit(x_train_std, t_train)
#
st.header('Quality of Sleep')
st.write('\n')
#入力画面
n_of_steps = st.slider('歩数（歩）', min_value=0, max_value=13000, step=100)
screen_time = st.selectbox('スマホ試聴時間（時間）', list(range(0,13)))
time_from_meal_to_bedtime = st.selectbox('食事から睡眠までの時間（時間）', list(range(0,13)))
drinking = st.selectbox('お酒を飲んだ？（yes=1, no=0）', list(range(0,2)))
calories_of_dinner = st.slider('夕食のカロリー（kcal）', min_value=0, max_value=2000, step=100)
time_of_sleeping = st.selectbox('睡眠予定時間は？（時間）', list(range(0,13)))

#入力データ
value_df = pd.DataFrame([], columns=[ '歩数（歩）', 'スマホ視聴時間（時間）', '食事から睡眠までの時間（時間）', '飲酒（yes=1, no=0）', '夕食のカロリー（kcal）', '睡眠予想時間（時間）'])
record = pd.DataFrame([[n_of_steps, screen_time, time_from_meal_to_bedtime, drinking, calories_of_dinner,time_of_sleeping]], columns=value_df.columns)
value_df = pd.concat([value_df, record], ignore_index=True)


#入力値の表示
st.write('### 入力値')
st.write(value_df)

#予測値のデータフレーム
#入力値の標準化
x_input = value_df.values
x_input_std = std_scaler.transform(x_input)
pred_probs = svc.predict(x_input_std)

#予測結果の出力
st.write('### 結果')
#st.write(f'あなたの今日の睡眠の質は{pred_probs}でしょう。おやすみなさい。')
if pred_probs[0] == 'いまいち':
    st.markdown(f'<span style="color: red;">あなたの睡眠の質は{pred_probs[0]}です。</span>', unsafe_allow_html=True)
else:
    st.write(f'あなたの睡眠の質は{pred_probs[0]}です。')