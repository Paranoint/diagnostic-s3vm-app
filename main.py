import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier
#from dummy_lapsvm import LapSVM  # Временно, пока нет настоящей реализации

st.set_page_config(page_title="SVM Медицинский помощник", layout="wide")
st.title("🩺 Диагностический помощник на S3VM")

st.markdown("""
Загрузите CSV-файл с медицинскими данными (частично размеченными), укажите, какая колонка отвечает за диагноз — и модель на основе **полуконтролируемого обучения** обучится и предскажет диагноз для остальных записей 🧠✨
""")

uploaded_file = st.file_uploader("📁 Загрузите ваш .csv файл", type=["csv"])

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
    if st.button("🚀 Обучить модель"):
        df[target_column] = df[target_column].replace(-1, np.nan)
        X = df[feature_columns]
        y = df[target_column]

        X_unlabeled = X[y.isnull()]
        y_unlabeled = y[y.isnull()]

        X_labeled = X[~y.isnull()]
        y_labeled = y[~y.isnull()].astype(int)

        st.info(f"Обучение на {len(X_labeled)} размеченных и {len(X_unlabeled)} неразмеченных записях")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        y_semi = y.copy()
        y_semi[y_semi.isnull()] = -1
        y_semi = y_semi.astype(int)

        base_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        model = SelfTrainingClassifier(base_model, threshold=0.95)

        model.fit(X_scaled, y_semi)

        predictions = model.predict(X_scaled)
        df['Предсказанный диагноз'] = df[target_column]
        df.loc[y_unlabeled.index, 'Предсказанный диагноз'] = pd.Series(predictions, index=df.index)[y_unlabeled.index]

        st.success("Модель обучена! Вот предсказания:")
        st.dataframe(df[[target_column, 'Предсказанный диагноз']])

        with st.expander("📉 Визуализация результатов (PCA)"):
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(scaler.transform(X))
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=pd.Categorical(df['Предсказанный диагноз']).codes, cmap='viridis')
            plt.title("Карта предсказаний (PCA)")
            plt.colorbar(scatter, ticks=range(len(df['Предсказанный диагноз'].unique())), label='Диагноз')
            st.pyplot(plt.gcf())

        st.download_button("💾 Скачать результат с диагнозами", data=df.to_csv(index=False).encode('utf-8'), file_name="s3vm_diagnosis_result.csv")
