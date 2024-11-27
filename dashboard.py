import mlflow.pyfunc
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

import functions.model_functions as ModelFunctions

DATASET = "datasets/dataset_nome.csv"
MODEL = "mlflow/mlartifacts/id_experimento/id_modelo/artifacts/model"
TARGET = "target"

@st.cache_resource 
def load_model():
    """
    Carrega o modelo treinado a partir do caminho especificado e o armazena em cache para otimizar o desempenho.

    A função usa o `mlflow.pyfunc.load_model` para carregar o modelo treinado armazenado no caminho especificado pela variável `MODEL`.
    A funcionalidade de cache é ativada pelo decorador `@st.cache_resource`, o que garante que o modelo seja carregado apenas uma vez e reutilizado em execuções subsequentes.

    Retorna:
    --------
    mlflow.pyfunc.PythonModel
        O modelo carregado e pronto para fazer previsões.
    """

    return mlflow.pyfunc.load_model(MODEL)

def model_metrics(y, y_pred, avg="weighted"):
    """
    Calcula e retorna as métricas de avaliação para o modelo, incluindo precisão, recall, f1-score, acurácia e a matriz de confusão.

    A função recebe os valores reais (`y`) e os valores preditos (`y_pred`) e calcula as seguintes métricas:
    - Acurácia (accuracy)
    - F1-score (com a média especificada por `avg`)
    - Precisão (precision)
    - Recall
    - Matriz de confusão (confusion matrix)

    Parâmetros:
    -----------
    y : array-like, shape (n_samples,)
        Os valores reais das classes de destino (verdadeiros).

    y_pred : array-like, shape (n_samples,)
        Os valores preditos pelo modelo para as classes de destino.

    avg : str, opcional, default="weighted"
        A estratégia de média a ser usada para calcular o f1-score, precisão e recall. Pode ser:
        - "micro" : Calcula métricas globalmente, contando o total de verdadeiros positivos, falsos negativos e falsos positivos.
        - "macro" : Calcula métricas para cada rótulo, e depois faz a média.
        - "weighted" : Calcula métricas para cada rótulo, e depois faz a média ponderada pelos suportes (número de verdadeiros rótulos).
        - "samples" : Calcula métricas para cada instância e depois faz a média.

    Retorna:
    --------
    tuple
        Um tupla contendo as métricas calculadas na seguinte ordem:
        - Acurácia (accuracy)
        - F1-score (f1score)
        - Precisão (precision)
        - Recall
        - Matriz de confusão (cm)
    """

    accuracy = accuracy_score(y, y_pred)
    f1score = f1_score(y, y_pred, average=avg)
    precision = precision_score(y, y_pred, average=avg)
    recall = recall_score(y, y_pred, average=avg)
    cm = confusion_matrix(y, y_pred)

    return accuracy, f1score, precision, recall, cm

def bar_metrics_plot(df_metrics: pd.DataFrame):
    """
    Exibe um gráfico de barras interativo usando Plotly para visualizar as métricas de desempenho de um modelo.

    A função recebe um DataFrame contendo as métricas e suas respectivas pontuações e gera um gráfico de barras, 
    onde o eixo x contém as métricas e o eixo y mostra os valores (pontuações) associadas a cada métrica.

    Parâmetros:
    -----------
    df_metrics : pd.DataFrame
        Um DataFrame contendo as métricas e suas respectivas pontuações.
        O DataFrame deve ter as seguintes colunas:
        - 'Metrics' : Os nomes das métricas (ex: 'Acurácia', 'F1-score', etc.).
        - 'Score' : Os valores ou pontuações das métricas.
    """

    st.subheader('Metrics')
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_metrics['Metrics'],
        y=df_metrics['Score'],
        name='Metrics',
        marker_color='green'
    ))

    fig.update_layout(barmode='group', title='Metrics')
    st.plotly_chart(fig)


def conf_matrix_test_plot(cm_test):
    """
    Exibe uma matriz de confusão interativa usando Plotly para visualizar o desempenho do modelo no conjunto de teste.

    A função recebe uma matriz de confusão (geralmente gerada por `confusion_matrix`) e exibe um gráfico interativo
    com as células representando as contagens de cada classe predita versus a classe real.

    Parâmetros:
    -----------
    cm_test : np.ndarray ou pd.DataFrame
        A matriz de confusão para o conjunto de teste.
        A matriz deve ser quadrada e conter as contagens de predições vs rótulos reais.
        Normalmente, é obtida utilizando `confusion_matrix(y_true, y_pred)`.
    """

    st.subheader('Confusion Matrix - Test Side')
    fig = px.imshow(cm_test)
    
    st.plotly_chart(fig)


def conf_matrix_train_plot(cm_train):
    """
    Exibe uma matriz de confusão interativa usando Plotly para visualizar o desempenho do modelo no conjunto de treino.

    A função recebe uma matriz de confusão (geralmente gerada por `confusion_matrix`) e exibe um gráfico interativo
    com as células representando as contagens de cada classe predita versus a classe real para o conjunto de treino.

    Parâmetros:
    -----------
    cm_train : np.ndarray ou pd.DataFrame
        A matriz de confusão para o conjunto de treino.
        A matriz deve ser quadrada e conter as contagens de predições vs rótulos reais.
        Normalmente, é obtida utilizando `confusion_matrix(y_true, y_pred)`.
    """

    st.subheader('Confusion Matrix - Train Side')
    fig = px.imshow(cm_train)
    
    st.plotly_chart(fig)


def trp_fpr(cm_test):
    """
    Calcula e exibe as taxas de Verdadeiro Positivo (TPR) e Falso Positivo (FPR) com base na matriz de confusão
    fornecida, e apresenta esses valores em um gráfico de barras interativo.

    A função extrai os valores de TP (True Positives), FN (False Negatives), FP (False Positives) e TN (True Negatives)
    da matriz de confusão, calcula o TPR e o FPR, e então exibe essas métricas em um gráfico de barras usando Plotly.

    Parâmetros:
    -----------
    cm_test : np.ndarray
        Matriz de confusão gerada para o conjunto de teste, onde:
        - cm_test[0, 0] é o número de Verdadeiros Positivos (TP),
        - cm_test[0, 1] é o número de Falsos Negativos (FN),
        - cm_test[1, 0] é o número de Falsos Positivos (FP),
        - cm_test[1, 1] é o número de Verdadeiros Negativos (TN).
    """
    
    tp,  fn , fp, tn = cm_test[0, 0], cm_test[0, 1], cm_test[1, 0], cm_test[1, 1]

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    df_tpr_fpr = pd.DataFrame(
        {
            'Type': ['TPR', 'FPR'],
            'Rate': [tpr, fpr]
        }
    )

    st.subheader('True Positive Rate x False Positive Rate')
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_tpr_fpr['Type'],
        y=df_tpr_fpr['Rate'],
        name='TPR x FPR',
        marker_color='blue'
    ))

    fig.update_layout(barmode='group', title='Metrics')
    st.plotly_chart(fig)


def class_report_test(y_test, y_pred_test):
    """
    Exibe o relatório de classificação para o conjunto de teste, incluindo métricas como precisão, recall, 
    f1-score e suporte para cada classe, em formato de tabela no Streamlit.

    A função utiliza o `classification_report` do scikit-learn para gerar as métricas de avaliação do modelo 
    no conjunto de teste, e exibe essas métricas em uma tabela formatada com pandas.

    Parâmetros:
    -----------
    y_test : array-like
        Rótulos verdadeiros para o conjunto de teste.

    y_pred_test : array-like
        Rótulos previstos pelo modelo para o conjunto de teste.
    """

    st.subheader('Classification Report - Test Side')
    classif_report = classification_report(y_test ,y_pred_test, output_dict=True)
    df_report = pd.DataFrame(classif_report).transpose()
    st.write(df_report)


def class_report_train(y_train, y_pred_train):
    """
    Exibe o relatório de classificação para o conjunto de treinamento, incluindo métricas como precisão, recall, 
    f1-score e suporte para cada classe, em formato de tabela no Streamlit.

    A função utiliza o `classification_report` do scikit-learn para gerar as métricas de avaliação do modelo 
    no conjunto de treinamento, e exibe essas métricas em uma tabela formatada com pandas.

    Parâmetros:
    -----------
    y_train : array-like
        Rótulos verdadeiros para o conjunto de treinamento.

    y_pred_train : array-like
        Rótulos previstos pelo modelo para o conjunto de treinamento.
    """
    
    st.subheader('Classification Report - Train Side')
    classif_report = classification_report(y_train ,y_pred_train, output_dict=True)
    df_report = pd.DataFrame(classif_report).transpose()
    st.write(df_report)

def main():
    df = pd.read_csv(DATASET)

    X_train, X_test, y_train, y_test = ModelFunctions.model_train_test(df, TARGET)

    X_train_processed, X_test_processed = ModelFunctions.model_pre_process(df, TARGET, X_train, X_test)

    model = load_model()

    y_pred_test = model.predict(X_test_processed)
    y_pred_train = model.predict(X_train_processed)

    accuracy_test, f1score_test, precision_test, recall_test, cm_test = model_metrics(y_test, y_pred_test)

    accuracy_train, f1score_train, precision_train, recall_train, cm_train = model_metrics(y_train, y_pred_train)

    df_metrics = pd.DataFrame(
        {
            'Metrics': [
                'Accuracy Score Test', 'F1 Score Test',
                'Precision Score Test', 'Recall Score Test',
                'Accuracy Score Train', 'F1 Score Train',
                'Precision Score Train', 'Recall Score Train'
            ],
            'Score': [
                accuracy_test, f1score_test, precision_test, 
                recall_test, accuracy_train, f1score_train, 
                precision_train, recall_train
            ]
        }
    )

    st.title('Metrics Monitor')

    st.sidebar.title('Choose here your visualization:')
    option = st.sidebar.selectbox(
        'Choose Visualization:',
        ['Metrics', 'Confusion Matrix - Test', 'Confusion Matrix - Train',
        'True Positive Rate x False Positive Rate', 'Classification Report - Test',
        'Classification Report - Train']
    )

    match option:
        case 'Metrics':
            bar_metrics_plot(df_metrics)
        case 'Confusion Matrix - Test':
            conf_matrix_test_plot(cm_test)
        case 'Confusion Matrix - Train':
            conf_matrix_train_plot(cm_train)
        case 'True Positive Rate x False Positive Rate':
            trp_fpr(cm_test)
        case 'Classification Report - Test':
            class_report_test(y_test, y_pred_test)
        case 'Classification Report - Train':
            class_report_train(y_train, y_pred_train)
        case _:
            st.error("Opção inválida!")

if __name__ == "__main__":
    main()