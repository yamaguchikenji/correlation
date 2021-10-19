import streamlit as st
import data as d
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

X_COLS = ['身長', '体重', '座高', '握力', '上体起こし', '長座体前屈', '反復横跳び', 'シャトルラン', '50ｍ走', '立ち幅跳び', 'ハンドボール投げ']
TEST_START_INDEX = 400
TEST_END_INDEX = 420

st.set_page_config(
    page_title="PE Score Analysis App",
    layout="wide",
    # collapsed: デフォルトを閉じる
    # expanded: デフォルトを開く
    initial_sidebar_state="expanded",
    )

@st.cache
def load_full_data():
    data = pd.read_csv(d.DATA_SOURCE)
    return data

@st.cache 
def load_num_data():
    data = pd.read_csv(d.DATA_SOURCE)
    rows = ['学年', '性別']
    data = data.drop(rows, axis=1)
    return data


def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'vis'

    st.sidebar.markdown('## ページ切り替え')
    # --- page選択ラジオボタン
    page = st.sidebar.radio('ページ選択', ('データ可視化', '単回帰分析'))

    # --- page振り分け
    if page == 'データ可視化':
        st.session_state.page = 'vis'
        vis()
    elif page == '単回帰分析':
        st.session_state.page = 'lr'
        lr()

# ---------------- グラフで可視化 :  各グラフを選択する ----------------------------------
def vis():
    st.title("体力測定 データ")
    score = load_num_data()
    full_data = load_full_data()

    st.sidebar.markdown('## いろんなグラフを試してみよう')

    # sidebar でグラフを選択
    graph = st.sidebar.radio(
        'グラフの種類',
        ('散布図', 'ヒストグラム', '箱ひげ図')
    )

    if  graph  == '散布図':
        left, right = st.beta_columns(2)

        with left: # 散布図の表示 
            x_label = st.selectbox('横軸を選択',X_COLS)
            y_label = st.selectbox('縦軸を選択',X_COLS)

        with right: # 色分けオプション            
            coloring = st.radio(
                "グラフの色分け",
                ('なし', '学年', '性別')
            )

        if coloring == '学年':
            fig = px.scatter(
            full_data,
            x=x_label,
            y=y_label,
            color="学年"
            )   
        elif coloring == "性別":
            fig = px.scatter(
                full_data,
                x=x_label,
                y=y_label,
                color="性別",
                )
        else:
            fig = px.scatter(
                full_data,
                x=x_label,
                y=y_label,
                )
        st.plotly_chart(fig, use_container_width=True)

        cor = d.get_corrcoef(score, x_label, y_label)
        st.write('相関係数：' + str(cor))

    # ヒストグラム
    elif graph == "ヒストグラム":
        hist_val = st.selectbox('変数を選択',X_COLS)
        fig = px.histogram(score, x=hist_val)
        st.plotly_chart(fig, use_container_width=True)
    
    # 箱ひげ図
    elif graph == '箱ひげ図':
        box_val_y = st.selectbox('箱ひげ図にする変数を選択',X_COLS)

        left, right = st.beta_columns(2)
        with left: # 散布図の表示 
            fig = px.box(full_data, x='学年', y=box_val_y, )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.box(full_data, x='性別', y=box_val_y)
            st.plotly_chart(fig, use_container_width=True)


# ---------------- 単回帰分析 ----------------------------------
def  lr():
    st.title('単回帰分析を使って予測してみよう！')
    df = load_full_data()

    st.sidebar.markdown('## まずはタイプ 1から！')

    # sidebar でグラフを選択
    df_type = st.sidebar.radio(
        '',
        ('タイプ 1', 'タイプ 2', 'タイプ 3')
    )

    # タイプ 1; フルデータ
    if df_type == "タイプ 1":
        filtered_df = load_num_data()
    # タイプ 2: 女子のみのデータ
    elif df_type == "タイプ 2":
        filtered_df = d.load_filtered_data(df, "女子")
    # タイプ 3: 高1女子のみのデータ
    else:
        filtered_df = d.load_filtered_data(df, "高1女子")

    # 変数を取得してから、単回帰したい
    with st.form('get_lr_data'):
        y_label = st.selectbox('予測したい変数(目的変数)', X_COLS)
        x_label = st.selectbox('予測に使いたい変数(説明変数)', X_COLS)

        # trainとtestをsplit
        df_train = pd.concat([filtered_df[filtered_df.no < TEST_START_INDEX] , filtered_df[filtered_df.no > TEST_END_INDEX]])
        df_test = pd.concat([filtered_df[TEST_START_INDEX <= filtered_df.no] , filtered_df[filtered_df.no <= TEST_END_INDEX]])

        y_train = df_train[[y_label]]
        y_test = df_test[[y_label]]
        X_train = df_train[[x_label]]
        X_test = df_test[[x_label]]
        # テストデータなし
        # y = df[[y_label]]
        # X = df[[x_label]]

        submitted = st.form_submit_button("分析スタート")

        if submitted:
            # モデルの構築
            model_lr = LinearRegression()
            # model_lr.fit(X, y)
            model_lr.fit(X_train, y_train)
            y_pred = model_lr.predict(X_test)

            # 結果の出力
            if model_lr.intercept_ < 0:
                st.write('y= %.3fx - %.3f' % (model_lr.coef_ , -1*(model_lr.intercept_)))
            else:
                st.write('y= %.3fx + %.3f' % (model_lr.coef_ , model_lr.intercept_))

            st.write('平均平方二乗誤差： %.2f' % np.sqrt(mean_squared_error(y_test, y_pred)))

            # グラフの描画
            fig = px.scatter(
            x=filtered_df[x_label].values, y=filtered_df[y_label].values,
            labels={'x':x_label, 'y':y_label},trendline='ols', trendline_color_override='red')
            st.plotly_chart(fig, use_container_width=True)

main()
