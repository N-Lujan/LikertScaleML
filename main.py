from streamlit_option_menu import option_menu
import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="Encuesta de Salud Mental",
    page_icon="./images/logo-naal.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

items_estres = ['Nunca', 'Casi nunca', 'De vez en cuando', 'Casi siempre', 'Siempre']
items_ansiedad = ['Nunca', 'Varios días', 'La mitad de los días', 'Casi cada día']

values1 = {'Nunca': 0, 'Casi nunca': 1, 'De vez en cuando': 2, 'Casi siempre': 3, 'Siempre': 4}
values2 = {'Nunca': 4, 'Casi nunca': 3, 'De vez en cuando': 2, 'Casi siempre': 1, 'Siempre': 0}
values3 = {'Nunca': 0, 'Varios días': 1, 'La mitad de los días': 2, 'Casi cada día': 3}

model_stress = pickle.load(open('models/svclassifier_stress.pkl', 'rb'))
model_anxiety = pickle.load(open('models/rl_model_anxiety.pkl', 'rb'))

labels_estres = {'ESTRES_ALTO': 'ALTO',
                 'ESTRES_MEDIO': 'MEDIO',
                 'ESTRES_BAJO': 'BAJO',
                 'SIN_ESTRES': 'SIN ESTRÉS'}

labels_anxiety = {'ANSIEDAD GRAVE': 'GRAVE',
                  'ANSIEDAD MODERADA': 'MODERADA',
                  'ANSIEDAD LEVE': 'LEVE',
                  'SIN ANSIEDAD': 'SIN ANSIEDAD'}

data = []

with st.sidebar:
    selected = option_menu(
        menu_title="Formularios",
        options=["Ansiedad", "Estrés"],
        icons=["heart-pulse", "heart-pulse-fill"],
        menu_icon="pencil-square"
    )

if selected == "Estrés":
    st.header('Evaluación del nivel de estrés')
    estres_question_1 = st.radio(
        label="1. ¿Con qué frecuencia ha estado afectado por algo que ha ocurrido inesperadamente?",
        options=items_estres,
    )

    data.append(values1.get(estres_question_1))

    estres_question_2 = st.radio(
        label="2. ¿Con qué frecuencia se ha sentido incapaz de controlar las cosas importantes en su vida?",
        options=items_estres,
    )

    data.append(values1.get(estres_question_2))

    estres_question_3 = st.radio(
        label="3. ¿Con qué frecuencia se ha sentido nervioso o estresado?",
        options=items_estres,
    )

    data.append(values1.get(estres_question_3))

    estres_question_4 = st.radio(
        label="4. ¿Con qué frecuencia ha estado seguro sobre su capacidad "
              "para manejar sus problemas personales?",
        options=items_estres,
    )

    data.append(values2.get(estres_question_4))

    estres_question_5 = st.radio(
        label="5. ¿Con qué frecuencia ha sentido que las cosas le van bien?",
        options=items_estres,
    )

    data.append(values2.get(estres_question_5))

    estres_question_6 = st.radio(
        label="6. ¿Con qué frecuencia ha sentido que no podía afrontar todas las cosas que tenía que hacer?",
        options=items_estres,
    )

    data.append(values1.get(estres_question_6))

    estres_question_7 = st.radio(
        label="7. ¿Con qué frecuencia ha podido controlar las dificultades de su vida?",
        options=items_estres,
    )

    data.append(values2.get(estres_question_7))

    estres_question_8 = st.radio(
        label="8. ¿Con qué frecuencia se ha sentido al control de todo?",
        options=items_estres,
    )

    data.append(values2.get(estres_question_8))

    estres_question_9 = st.radio(
        label="9. ¿Con qué frecuencia ha estado enfadado porque las cosas que le han ocurrido "
              "estaban fuera de su control?",
        options=items_estres,
    )

    data.append(values1.get(estres_question_9))

    estres_question_10 = st.radio(
        label="10. ¿Con qué frecuencia ha sentido que las dificultades se acumulan tanto que "
              "no puede superarlas?",
        options=items_estres,
    )

    data.append(values1.get(estres_question_10))

    if st.button("Evaluar"):
        data_evaluate = np.array([data])
        pred = model_stress.predict(data_evaluate)
        st.success('NIVEL DE ESTRÉS: {}'.format(labels_estres.get(pred[0])))

if selected == "Ansiedad":
    st.header('Evaluación del nivel de ansiedad')
    ansiedad_question_1 = st.radio(
        label="1. Sentirse nervioso, ansioso y/o notar que se le ponen los nervios de punta.",
        options=items_ansiedad,
    )

    data.append(values3.get(ansiedad_question_1))

    ansiedad_question_2 = st.radio(
        label="2. No ser capaz de parar o controlar sus preocupaciones.",
        options=items_ansiedad,
    )

    data.append(values3.get(ansiedad_question_2))

    ansiedad_question_3 = st.radio(
        label="3. Preocuparse demasiado sobre diferentes cosas.",
        options=items_ansiedad,
    )

    data.append(values3.get(ansiedad_question_3))

    ansiedad_question_4 = st.radio(
        label="4. Dificultad para relajarse.",
        options=items_ansiedad,
    )

    data.append(values3.get(ansiedad_question_4))

    ansiedad_question_5 = st.radio(
        label="5. Estar tan desasosegado que le resulta difícil parar quieto.",
        options=items_ansiedad,
    )

    data.append(values3.get(ansiedad_question_5))

    ansiedad_question_6 = st.radio(
        label="6. Sentirse fácilmente disgustado o irritable.",
        options=items_ansiedad,
    )

    data.append(values3.get(ansiedad_question_6))

    ansiedad_question_7 = st.radio(
        label="7. Sentirse asustado como si algo horrible pudiese pasar.",
        options=items_ansiedad,
    )

    data.append(values3.get(ansiedad_question_7))

    if st.button("Evaluar"):
        data_evaluate = np.array([data])
        pred = model_anxiety.predict(data_evaluate)
        st.success('NIVEL DE ANSIEDAD: {}'.format(labels_anxiety.get(pred[0])))
