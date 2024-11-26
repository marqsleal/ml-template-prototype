import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline 
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay 

PALETTE = 'YlGnBu'
COLOR = 'gray'
EDGE = 'black'
DPI = 120

def qnt_obs_per_class(df: pd.DataFrame, title: str, target_column: str, labels_df: list[str]):
    sns.set_context('paper')
    sns.set_style('white')

    plt.figure(figsize=(4, 4))
    plt.title('Quantidade de observações por Classes', y=1.1)
    plt.figtext(0.5, 0.93, f'{title} - {target_column}', ha='center', color=COLOR, fontsize=9)

    dados_target = df[target_column].value_counts()

    ax= sns.barplot(data = dados_target, palette = PALETTE)
    ax.bar_label(ax.containers[0], fontsize=10)
    ax.bar_label(ax.containers[1], fontsize=10)
    plt.ylabel('Nº de observações')
    plt.xticks(ticks=[0, 1], labels=labels_df)
    sns.despine(left = True)
    plt.figure(dpi=DPI)
    plt.show()

def cont_features_mean_per_target_class(df: pd.DataFrame, title: str, continum_columns: list[str],target_column: str, labels_df: list[str]):
    plt.figure(figsize = (12, 12))
    plt.suptitle('Estatísticas de médias das features contínuas por classes ', y=1.02, fontsize=16)
    plt.figtext(0.5, 0.99, f'{title} - {target_column}', ha='center', fontsize=10, color=COLOR)

    for i, q in enumerate(continum_columns, 1):
        plt.subplot(4, 3, i)
        estatisticas_media = df.groupby([target_column])[q].mean().reset_index()
        ax = sns.barplot(data = estatisticas_media, x = target_column, y = q, palette = PALETTE)
        ax.bar_label(ax.containers[0], fontsize=9, padding = -15)
        ax.bar_label(ax.containers[1], fontsize=9, padding = -15)
        sns.despine(left = True)
        plt.xticks(ticks=[0, 1], labels=labels_df)
        plt.title(f'Média da variável {q}', fontsize=10)

    plt.tight_layout()
    plt.figure(dpi=DPI)
    plt.show()

def dist_cont_per_target_class(df: pd.DataFrame, title: str, continum_columns: list[str], target_column: str, labels_df: list[str]):
    n_subplots = len(continum_columns)

    fig, ax = plt.subplots(1, n_subplots, figsize=(4 * n_subplots, 4))

    if n_subplots == 1:
        ax = [ax]

    for i, column in enumerate(continum_columns):
        sns.histplot(
            data=df, 
            x=column, 
            hue=target_column, 
            palette = PALETTE, 
            kde = True, 
            linewidth = .1, 
            ax = ax[i], 
            edgecolor = EDGE
        )

        ax[i].set_title(f'Distribuição da {column} por \n Status de {target_column}', y = 1.1)
        ax[i].set_ylabel('Nº Observações')
        ax[i].set_xlabel(f'{column}')
        ax[i].legend(title=f'{title} - {target_column}', labels=labels_df)
        ax[i].grid(axis = 'y', alpha = 0.5)
        ax[i].grid(axis = 'x', alpha = 0)

    sns.despine(left = True)
    plt.tight_layout()
    plt.figure(dpi=DPI)
    plt.show()

def correlation_plot(df: pd.DataFrame, methods: list[str]):
    for method in methods:
        correlation_matrix = df.corr(method=method)
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        plt.figure(figsize=(8, 4))
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap=PALETTE, 
            fmt='.2f', 
            mask=mask, 
            xticklabels=correlation_matrix.columns, 
            yticklabels=correlation_matrix.columns
        )
        plt.title(f"Correlação ({method.capitalize()})")
    
    sns.despine(left=True)
    plt.tight_layout()
    plt.figure(dpi=DPI)
    plt.show()

def outliers_plot(df: pd.DataFrame, colunas_num_continuas: list[str]):
    plt.figure(figsize = (10, 6))
    plt.suptitle('Boxplots ', fontsize=16)

    for i, q in enumerate(colunas_num_continuas, 1):
        paleta = sns.color_palette(PALETTE)
        plt.subplot(2, 2, i)
        ax = sns.boxplot(data = df[q], color = paleta[0], linecolor = paleta[4], orient='h', width=0.5)
        sns.despine(left = True)
        plt.title(f'Boxplot da variável {q}', fontsize=10)
        plt.grid(alpha=0.5)

    plt.tight_layout()
    plt.figure(dpi=DPI)
    plt.show()

def cm_roc_recall_plot(model_name, model, X_train, X_test, y_train, y_test) -> None:
    paleta = sns.color_palette(PALETTE)

    pipeline = Pipeline(steps=[
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  

    print(f'{model_name} \n{classification_report(y_test, y_pred)}')
    print(f'ROC AUC score: {roc_auc_score(y_test, y_pred_proba):.4f}')

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    plt.suptitle(f'Métricas para o modelo {model_name}')

    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=PALETTE, colorbar=False, ax=ax[0])
    ax[0].set_title('Matriz de Confusão - Dados de Teste', fontsize=10)
    ax[0].set_xlabel('Rótulos Preditos')
    ax[0].set_ylabel('Rótulos Reais')

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    ax[1].set_title('Curva ROC-AUC', fontsize=10)
    ax[1].plot(fpr, tpr, label=f'Curva ROC (AUC = {roc_auc:.2f})', color=paleta[4])
    ax[1].plot([0, 1], [0, 1], 'k--', label='Linha aleatória (AUC = 0.5)', color = 'gray')
    ax[1].set_xlabel('Taxa de Falsos Positivos (FPR)')
    ax[1].set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
    ax[1].legend(loc='lower right')
    ax[1].grid(alpha =.25)

    precision, recall, _ = precision_recall_curve(y_pred, y_pred_proba)
    ax[2].set_title('Curva Precisão-recall', fontsize=10)
    ax[2].plot(recall, precision, label='Curva de precision-recall', color=paleta[4])
    ax[2].set_xlabel('Recall')
    ax[2].set_ylabel('Precisão')
    ax[2].legend(loc='best')
    ax[2].grid(alpha =.25)

    plt.tight_layout()
    plt.figure(dpi=DPI)
    plt.show()

def model_metrics(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    resumo_metricas = []

    for model_name, model in models.items():
        pipeline = Pipeline(steps=[
        ('classifier', model)])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        resumo_metricas.append({
            'Modelo': model_name,
            'Precisão': precision_score(y_test, y_pred),
            'Acurácia': accuracy_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'f1-score (weighted)': f1_score(y_test, y_pred, average = "weighted"),
            'f1-score (macro avg)': f1_score(y_test, y_pred, average = "macro")
        })
    
    resumo_metricas_df = pd.DataFrame(resumo_metricas)
    resumo_metricas_df.sort_values(by = 'f1-score (macro avg)', ascending = False)

    return resumo_metricas_df
