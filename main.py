import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from methods.svc_self_training import predict_tsvm_cccp
from methods.LS_SVM_help import predict_ls_help
from methods.Laplacian_SVM import predict_lapsvm

def apply_predictions(df, predictions, target_column, y_unlabeled, full=False):
    df['Предсказанный диагноз'] = df[target_column]
    df['Источник'] = 'ручное'
    if full and len(predictions) == len(df):
        df['Предсказанный диагноз'] = predictions
        df.loc[y_unlabeled.index, 'Источник'] = 'предсказание'
    elif len(predictions) == len(y_unlabeled):
        df.loc[y_unlabeled.index, 'Предсказанный диагноз'] = predictions
        df.loc[y_unlabeled.index, 'Источник'] = 'предсказание'
    else:
        raise ValueError("Неправильная длина предсказаний")
    return df

st.set_page_config(page_title="SVM Медицинский помощник", layout="wide")
st.title("🩺 Диагностический помощник на S3VM")

st.markdown("""
Загрузите CSV-файл с медицинскими данными (частично размеченными), укажите, какая колонка отвечает за диагноз — и модель на основе **полуконтролируемого обучения** обучится и предскажет диагноз для остальных записей 🧠✨
""")

uploaded_file = st.file_uploader("📁 Загрузите ваш .csv файл", type=["csv"])

MODEL_PARAM_WIDGETS = {
    "LapSVM": lambda: {
        "gamma": st.number_input("Gamma", value=0.5),
        "lamA": st.number_input("λA (размеченные)", value=10.0),
        "lamI": st.number_input("λI (неразмеченные)", value=10.0),
        "k": st.number_input("k (соседи)", value=10, min_value=1, step=1)
    },
    "Gradient S3VM": lambda: {
        "C": st.number_input("C (регуляризация)", value=1.0),
        "C_star_final": st.number_input("C* (неразмеченные)", value=0.1),
        "gamma": st.number_input("Gamma (ядро)", value=1.0)
    },
    "LS-SVM + Help-Training Classifier": lambda: {
        "gamma": st.number_input("Gamma", value=30.0),
        "sigma": st.number_input("Sigma", value=13.5)
    }
}

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Данные")
    st.dataframe(df.head())

    target_column = st.selectbox("🎯 Выберите колонку с целевой меткой (диагнозом)", df.columns)
    possible_features = [col for col in df.columns if col != target_column]

    if 'feature_columns' not in st.session_state:
        st.session_state.feature_columns = []

    if st.button("📌 Выбрать все признаки"):
        st.session_state.feature_columns = possible_features

    st.session_state.feature_columns = st.multiselect(
        "🧬 Выберите признаки для анализа",
        possible_features,
        default=st.session_state.feature_columns
    )

    feature_columns = st.session_state.feature_columns

    selected_method = st.selectbox("🤖 Выберите метод обучения",
        ["Gradient S3VM",
        "LS-SVM + Help-Training Classifier",
        "Laplacian SVM",
        ])

    method_functions = {
        "Gradient S3VM": predict_tsvm_cccp,
        "LS-SVM + Help-Training Classifier": predict_ls_help,
        "Laplacian SVM": predict_lapsvm,
    }

    with st.expander("⚙ Параметры модели"):
        get_params = MODEL_PARAM_WIDGETS.get(selected_method)
        model_params = get_params() if get_params else {}

    if st.button("🚀 Обучить модель"):
        if not feature_columns:
            st.error("❌ Пожалуйста, выберите хотя бы один признак для обучения.")
        else:
            X = df[feature_columns]
            y = df[target_column]

            y_clean = y.fillna(-1).astype(int)
            X_scaled = StandardScaler().fit_transform(X)

            X_unlabeled = X_scaled[y_clean == -1]
            y_unlabeled = y[y_clean == -1]

            X_labeled = X_scaled[y_clean != -1]
            y_labeled = y[y_clean != -1]

            st.info(f"Обучение на {len(X_labeled)} размеченных и {len(X_unlabeled)} неразмеченных записях")
            
            method_fn = method_functions.get(selected_method)

            if method_fn:
                predictions = method_fn(
                    X_labeled, y_labeled, X_unlabeled, **model_params
                )
                df = apply_predictions(df, predictions, target_column, y_unlabeled, full=True)

                st.success("Модель обучена! Вот предсказания:")
                st.dataframe(df[[target_column, 'Предсказанный диагноз']])

                with st.expander("📉 Визуализация результатов (PCA)"):
                    pca = PCA(n_components=2)
                    X_vis = pca.fit_transform(X_scaled)
                    plt.figure(figsize=(8, 6))
                    is_labeled = df['Источник'] == 'ручное'
                    is_predicted = df['Источник'] == 'предсказание'

                    plt.scatter(
                        X_vis[is_labeled, 0], X_vis[is_labeled, 1],
                        c=pd.Categorical(df.loc[is_labeled, 'Предсказанный диагноз']).codes,
                        cmap='viridis', marker='o', alpha=0.6, label='Лейблы'
                    )
                    plt.scatter(
                        X_vis[is_predicted, 0], X_vis[is_predicted, 1],
                        c=pd.Categorical(df.loc[is_predicted, 'Предсказанный диагноз']).codes,
                        cmap='viridis', marker='x', alpha=0.9, label='Предсказания'
                    )
                    plt.xlabel("PCA 1")
                    plt.ylabel("PCA 2")
                    plt.title("Карта предсказаний (PCA)")
                    plt.colorbar(label='Диагноз')
                    plt.legend()
                    st.pyplot(plt.gcf())

                st.download_button("💾 Скачать результат с диагнозами", data=df.to_csv(index=False).encode('utf-8'), file_name= selected_method + " result.csv")
            else:
                st.warning("Выбранный метод не реализован.")
        
        

        
