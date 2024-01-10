import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_metrics(resultados_totales, title, height=500, width=1000):

    identificadores = [resultado['identificador'] for resultado in resultados_totales]
    precisiones = [resultado['presicion'] for resultado in resultados_totales]
    recalls = [resultado['recall'] for resultado in resultados_totales]
    f1_scores = [resultado['f1_score'] for resultado in resultados_totales]

    # Crear figuras
    fig = go.Figure()

    # Añadir barras para cada métrica
    fig.add_trace(go.Bar(x=identificadores, y=precisiones, text=precisiones, name='Precision'))
    fig.add_trace(go.Bar(x=identificadores, y=recalls, text=recalls, name='Recall'))
    fig.add_trace(go.Bar(x=identificadores, y=f1_scores, text=f1_scores, name='F1 Score'))

    # Configurar diseño del gráfico
    fig.update_layout(barmode='group', xaxis_tickangle=-45, title=f'Métricas por estación {title}',
                      xaxis=dict(title='Estación'), yaxis=dict(title='Métricas'), height=height, width=width)

    # Ajustar el color y tamaño del texto
    fig.update_traces(textfont=dict(color='black', size=15))

    # Mostrar el gráfico
    fig.show()

# Graficar los verdaderos positivos, falsos positivos y falsos negativos
def plot_bar(resultados_totales, title, height=500, width=1000):

    identificadores = [resultado['identificador'] for resultado in resultados_totales]

    # Crear figuras
    fig = go.Figure()

    # Añadir barras para verdadero positivo, falso positivo y falso negativo
    fig.add_trace(go.Bar(x=identificadores, y=[resultado['resultados']['Verdaderos Positivos'] for resultado in resultados_totales], 
                         text=[resultado['resultados']['Verdaderos Positivos'] for resultado in resultados_totales],
                         name='Verdaderos Positivos'))
    
    fig.add_trace(go.Bar(x=identificadores, y=[resultado['resultados']['Falsos Positivos'] for resultado in resultados_totales], 
                         text=[resultado['resultados']['Falsos Positivos'] for resultado in resultados_totales],
                         name='Falsos Positivos'))
    
    fig.add_trace(go.Bar(x=identificadores, y=[resultado['resultados']['Falsos Negativos'] for resultado in resultados_totales], 
                         text=[resultado['resultados']['Falsos Negativos'] for resultado in resultados_totales],
                         name='Falsos Negativos'))

    # Configurar diseño del gráfico
    fig.update_layout(barmode='group', xaxis_tickangle=-45, title= f'TP, FP y FN por estación {title}',
                      xaxis=dict(title='Estación'), yaxis=dict(title=f'Métricas'), height=height, width=width)

    # Ajustar el color y tamaño del texto
    fig.update_traces(textfont=dict(color='black', size=15))

    # Mostrar el gráfico
    fig.show()

def plot_differences_amount(resultados_before, resultados_after,title ,plot_metrics = True, height=500, width=1000):
    identificadores = [resultado['identificador'] for resultado in resultados_before]  # assuming same order

    if plot_metrics:
        # Plot precision, recall, and F1 score
        precisiones_diff = [round(after['presicion'] - before['presicion'], 2) for before, after in zip(resultados_before, resultados_after)]
        recalls_diff = [round(after['recall'] - before['recall'], 2) for before, after in zip(resultados_before, resultados_after)]
        f1_scores_diff = [round(after['f1_score'] - before['f1_score'], 2) for before, after in zip(resultados_before, resultados_after)]

        metrics = [precisiones_diff, recalls_diff, f1_scores_diff]
        metric_names = ['Cambio de precision', 'Cambio de recall', 'Cambio de F1 Score']
    else:
        # Plot true positives, false negatives, and false positives
        tp_diff = [after['resultados']['Verdaderos Positivos'] - before['resultados']['Verdaderos Positivos'] for before, after in zip(resultados_before, resultados_after)]
        fn_diff = [after['resultados']['Falsos Negativos'] - before['resultados']['Falsos Negativos'] for before, after in zip(resultados_before, resultados_after)]
        fp_diff = [after['resultados']['Falsos Positivos'] - before['resultados']['Falsos Positivos'] for before, after in zip(resultados_before, resultados_after)]

        metrics = [tp_diff, fn_diff, fp_diff]
        metric_names = ['Cambio de Verdaderos Positivos', 'Cambio de Falsos Negativos', 'Cambio de Falsos Positivos']

    # Create figures
    fig = go.Figure()

    # Add bars for each metric difference
    for metric_diff, metric_name in zip(metrics, metric_names):
        fig.add_trace(go.Bar(x=identificadores, y=metric_diff, text=metric_diff, name=metric_name))

    # Configure the chart layout
    fig.update_layout(barmode='group', xaxis_tickangle=-45, title=f"Cambio por estación {title}",
                      xaxis=dict(title='Estación'), yaxis=dict(title='Cambio'), height=height, width=width)

    # Adjust the color and size of the text
    fig.update_traces(textfont=dict(color='black', size=15))

    # Show the chart
    fig.show()



def plot_differences_bar(resultados_before, resultados_after, title, plot_metrics=True, height=500, width=1000):
    identificadores = [resultado['identificador'] for resultado in resultados_before]  # assuming same order

    if plot_metrics:
        # Plot precision, recall, and F1 score
        metrics_before = [[round(before['presicion'], 2) for before in resultados_before],
                          [round(before['recall'], 2) for before in resultados_before],
                          [round(before['f1_score'], 2) for before in resultados_before]]
        metrics_after = [[round(after['presicion'], 2) for after in resultados_after],
                         [round(after['recall'], 2) for after in resultados_after],
                         [round(after['f1_score'], 2) for after in resultados_after]]
        metric_names = ['Precision', 'Recall', 'F1 Score']
    else:
        # Plot true positives, false negatives, and false positives
        metrics_before = [[before['resultados']['Verdaderos Positivos'] for before in resultados_before],
                          [before['resultados']['Falsos Negativos'] for before in resultados_before],
                          [before['resultados']['Falsos Positivos'] for before in resultados_before]]
        metrics_after = [[after['resultados']['Verdaderos Positivos'] for after in resultados_after],
                         [after['resultados']['Falsos Negativos'] for after in resultados_after],
                         [after['resultados']['Falsos Positivos'] for after in resultados_after]]
        metric_names = ['Verdaderos Positivos', 'Falsos Negativos', 'Falsos Positivos']

    # Create subplots: one row for each metric
    fig = make_subplots(rows=3, cols=1)

    offset_y = -0.2

    for i, metric in enumerate(metric_names):
        # Add traces for 'before' and 'after' for each metric
        fig.add_trace(go.Bar(x=identificadores, y=metrics_before[i], name=f'{metric} sin val'), row=i+1, col=1)
        fig.add_trace(go.Bar(x=identificadores, y=metrics_after[i], name=f'{metric} con val'), row=i+1, col=1)

        # Add annotations for 'before' and 'after' values
        for j, identifier in enumerate(identificadores):
            y_pos_before = max(metrics_before[i][j] / 2, offset_y)
            y_pos_after = max(metrics_after[i][j] / 2, offset_y)
            fig.add_annotation(x=j-0.2, y=y_pos_before, text=str(metrics_before[i][j]), showarrow=False, font=dict(color='white', size = 12), row=i+1, col=1)
            fig.add_annotation(x=j+0.2, y=y_pos_after, text=str(metrics_after[i][j]), showarrow=False, font=dict(color='white', size = 12), row=i+1, col=1)

    # Configure the chart layout
    fig.update_layout(height=height, width=width, title_text= f"Cambio por estación {title}")

    # Show the chart
    fig.show()




def plot_differences_scatter(resultados_before, resultados_after, title, plot_metrics=True, height=500, width=1000):
    identificadores = [resultado['identificador'] for resultado in resultados_before]  # assuming same order

    if plot_metrics:
        # Plot precision, recall, and F1 score
        metrics_before = [[round(before['presicion'], 2) for before in resultados_before],
                          [round(before['recall'], 2) for before in resultados_before],
                          [round(before['f1_score'], 2) for before in resultados_before]]
        metrics_after = [[round(after['presicion'], 2) for after in resultados_after],
                         [round(after['recall'], 2) for after in resultados_after],
                         [round(after['f1_score'], 2) for after in resultados_after]]
        metric_names = ['Precision', 'Recall', 'F1 Score']
    else:
        # Plot true positives, false negatives, and false positives
        metrics_before = [[before['resultados']['Verdaderos Positivos'] for before in resultados_before],
                          [before['resultados']['Falsos Negativos'] for before in resultados_before],
                          [before['resultados']['Falsos Positivos'] for before in resultados_before]]
        metrics_after = [[after['resultados']['Verdaderos Positivos'] for after in resultados_after],
                         [after['resultados']['Falsos Negativos'] for after in resultados_after],
                         [after['resultados']['Falsos Positivos'] for after in resultados_after]]
        metric_names = ['Verdaderos Positivos', 'Falsos Negativos', 'Falsos Positivos']

    # Create subplots: one row for each metric
    fig = make_subplots(rows=3, cols=1)

    for i, metric in enumerate(metric_names):
        # Add traces for 'before' and 'after' for each metric
        fig.add_trace(go.Scatter(x=identificadores, y=metrics_before[i], mode='lines+markers', name=f'{metric} sin val'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=identificadores, y=metrics_after[i], mode='lines+markers', name=f'{metric} con val'), row=i+1, col=1)

        # Add annotations for each data point
        for j, identifier in enumerate(identificadores):
            # Calculate a dynamic offset based on the y-coordinate of the data point
            y_offset_before = 0.07 * metrics_before[i][j]
            y_offset_after = 0.08 * metrics_after[i][j]

            # Add the dynamic offset to the y parameter
            fig.add_annotation(x=identifier, y=metrics_before[i][j]+y_offset_before, text=str(metrics_before[i][j]), showarrow=False, font=dict(color='black', size=10), row=i+1, col=1)
            fig.add_annotation(x=identifier, y=metrics_after[i][j]+y_offset_after, text=str(metrics_after[i][j]), showarrow=False, font=dict(color='black', size=10), row=i+1, col=1)

    # Configure the chart layout
    fig.update_layout(height=height, width=width, title_text= f"Cambio por estación {title}")

    # Show the chart
    fig.show()
