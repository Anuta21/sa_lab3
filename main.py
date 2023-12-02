import streamlit as st
import pandas as pd
import numpy as np
from sa_lab3.GridSearch import *

st.set_page_config(page_title='ЛР3', layout='wide')

st.markdown("""
    <style>
    .stProgress .st-ey {
        background-color: #5fe0de;
    }
    </style>
    """, unsafe_allow_html=True)
st.header('Відновлення функціональних залежностей у вигляді ієрархічної багаторівневої системи моделей у класі мультиплікативних функцій')
st.write('Виконала **бригада 3 з КА-01** у складі Магаріної Анни, Стожок Анастасії, Захарової Єлизавети')

st.markdown("""
   <style>
            .st-emotion-cache-133e3c0  {background-color: transparent; padding: 0}
            .st-emotion-cache-6qob1r {background-color: #E6AFAF; }
            .main  {background-color: #FEECEF; }
            .st-emotion-cache-18ni7ap  {background-color: #FEECEF; }
            .st-emotion-cache-1fttcpj {display: none}
            .st-emotion-cache-1v7f65g .e1b2p2ww14 {margin: 0}
            .st-emotion-cache-3qrmye {background-color: #E81F64;}
            .st-emotion-cache-16txtl3 {padding: 20px 20px 0px 20px}
            .st-emotion-cache-1629p8f .h1 {padding-bottom: 20px}
            .st-emotion-cache-z5fcl4 {padding: 20px 20px }
}
            
    </style>

    """, unsafe_allow_html=True)

st.sidebar.title("Дані")
input_file = st.sidebar.file_uploader('Оберіть файл вхідних даних', type=['csv', 'txt'], key='input_file')
st.sidebar.markdown('<div style="border-bottom: 1px solid #AAA1A5; height: 1px;"/>', unsafe_allow_html=True)

st.sidebar.title("Розмірність")
col1, col2, col3, col4 = st.sidebar.columns(4)
x1_dim = col1.number_input('X1', value=2, step=1, min_value=0, key='x1_dim')
x2_dim = col2.number_input('X2', value=2, step=1, min_value=0, key='x2_dim')
x3_dim = col3.number_input('X3', value=3, step=1, min_value=0, key='x3_dim')
y_dim = col4.number_input('Y', value=4, step=1, min_value=0, key='y_dim')
st.sidebar.markdown('<div style="border-bottom: 1px solid #AAA1A5; height: 1px;"/>', unsafe_allow_html=True)


st.sidebar.title('Поліноми')
poly_type = ['Поліноми Лежандра', 'Зсунуті поліноми Лежандра',  'Синус']
selected_poly_type = st.sidebar.selectbox("Оберіть тип", poly_type)

custom_structure = st.sidebar.checkbox('Власна структура функцій')

st.sidebar.write('Оберіть степені поліномів')
col1, col2, col3 = st.sidebar.columns(3)
x1_deg = col1.number_input('X1', value=1, step=1, min_value=0, key='x1_deg')
x2_deg = col2.number_input('X2', value=1, step=1, min_value=0, key='x2_deg')
x3_deg = col3.number_input('X3', value=1, step=1, min_value=0, key='x3_deg')
st.sidebar.markdown('<div style="border-bottom: 1px solid #AAA1A5; height: 1px;"/>', unsafe_allow_html=True)

st.sidebar.title('Додатково')
col1, col2= st.sidebar.columns(2)
weight_method = col1.radio('Ваги цільових функцій', ['Нормоване значення', 'Середнє арифметичне'], key='select_func')
lambda_option = col2.checkbox('Визначати λ з трьох систем рівнянь')
normed_plots = col2.checkbox('Графіки для нормованих значень')

col1, col2 = st.columns(2)
if col1.button('Виконати обчислення', key='run1'):
    input_file_text = input_file.getvalue().decode()

    params = {
                'dimensions': [x1_dim, x2_dim, x3_dim, y_dim],
                'input_file': input_file_text,
                'degrees': [x1_deg, x2_deg, x3_deg],
                'weights': weight_method,
                'poly_type': selected_poly_type,
                'lambda_multiblock': lambda_option,
                'custom_structure': custom_structure
            }
    with st.spinner():
        solver, solution, degrees = getSolution(params, pbar_container=col3, max_deg=15)
    if degrees != params['degrees']:
        col1.write(f'**Підібрані степені поліномів:**  \tX1 : {degrees[0]}, X2 : {degrees[1]}, X3 : {degrees[2]}')

    error_cols = st.columns(2)
    for ind, info in enumerate(solver.show_streamlit()[-2:]):
        error_cols[ind].subheader(info[0])
        error_cols[ind].dataframe(info[1])

    matrices = solver.show_streamlit()[:-2]
    if normed_plots:
        st.subheader(matrices[1][0])
        st.dataframe(matrices[1][1])
    else:
        st.subheader(matrices[0][0])
        st.dataframe(matrices[0][1])

    st.write(solution.get_results())

    matr_cols = st.columns(3)
    for ind, info in enumerate(matrices[2:5]):
        matr_cols[ind].subheader(info[0])
        matr_cols[ind].dataframe(info[1])

if col2.button('Побудувати графіки', key='run2'):
    input_file_text = input_file.getvalue().decode()
    params = {
            'dimensions': [x1_dim, x2_dim, x3_dim, y_dim],
            'input_file': input_file_text,
            'degrees': [x1_deg, x2_deg, x3_deg],
            'weights': weight_method,
            'poly_type': selected_poly_type,
            'lambda_multiblock': lambda_option,
            'custom_structure': custom_structure
        }
    with st.spinner():
        solver, solution, degrees = getSolution(params, pbar_container=col3, max_deg=15)

    if normed_plots:
        Y_values = solution._solution.Y
        F_values = solution._solution.F
    else:
        Y_values = solution._solution.Y_
        F_values = solution._solution.F_

    plot_n_cols = Y_values.shape[1]
        
    st.subheader('Графіки')
    plot_cols = st.columns(plot_n_cols)

    for n in range(plot_n_cols):
        df = pd.DataFrame(
                np.array([Y_values[:, n], F_values[:, n]]).T,
                columns=[f'Y{n+1}', f'F{n+1}']
            )
        # print(df)
        plot_cols[n].write(f'Координата {n+1}')
        plot_cols[n].line_chart(df)
        plot_cols[n].write(f'Похибка координати {n+1}')
        df = pd.DataFrame(
                np.abs(Y_values[:, n] - F_values[:, n]).T,
                columns=[f'E{n+1}']
            )
        plot_cols[n].line_chart(df)
       
