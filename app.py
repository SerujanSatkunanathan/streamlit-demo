from PIL import Image
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px

    dataset = st.sidebar.file_uploader("DataSet", type='.csv')

    if dataset is not None:
        data = pd.read_csv(dataset)
        x = data['Hours']
        y = data['Scores']
        import numpy as np
        x = np.array(list(x))

        from sklearn.model_selection import train_test_split
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=False)
        model = LinearRegression()
        model.fit(x_train.reshape(-1, 1), y_train)

        st.success('Model Trained Successfully...')
        from sklearn.metrics import mean_absolute_error

        pred = model.predict(x_test.reshape(-1, 1))

        show = pd.DataFrame({"Actual Score": y_test, "Predicted Score": pred})
        st.dataframe(show)

        mse = mean_absolute_error(y_test, pred)
        st.info(f"Mean absolute value {mse}")

    else:
        st.warning("Upload your dataset.")


