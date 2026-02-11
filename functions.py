import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
def bi_cat_normplot(df, column, hue_column):
    """
    Візуалізує розподіл категоріальної змінної відносно бінарної цільової змінної.

    Parameters
    ----------
    df : pandas.DataFrame
        Вхідний датафрейм з даними.
    column : str
        Назва категоріальної ознаки для аналізу.
    hue_column : str
        Назва бінарної цільової змінної (класу).

    Description
    -----------
    Функція будує два графіки:
    1) Нормалізований розподіл (у відсотках) значень категоріальної ознаки
       для кожного класу цільової змінної.
    2) Абсолютну кількість спостережень у кожній категорії з підписами
       кількості та відсоткового співвідношення класів.

    Використовується для первинного EDA та аналізу залежності
    між категоріальною ознакою і цільовою змінною.
    """
    unique_hue_values = df[hue_column].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(14, 6)

    # --- ГРАФІК №1 (відсотки між категоріями) ---
    pltname1 = f'Нормалізований розподіл значень за категорією: {column}'

    proportions = df.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions * 100).round(2)
    prop_table = proportions.unstack(hue_column)

    ordered_categories = prop_table.sort_values(
        by=unique_hue_values[0], ascending=False
    ).index

    ax1 = prop_table.loc[ordered_categories].plot.bar(
        ax=axes[0], title=pltname1
    )

    for container in ax1.containers:
        ax1.bar_label(container, fmt='{:,.1f}%')

    # --- ГРАФІК №2 (кількість + відсотки) ---
    pltname2 = f'Кількість даних та відсоток класів у категорії: {column}'

    counts = df.groupby(column)[hue_column].value_counts().unstack(hue_column)

    percentages = (
        df.groupby(column)[hue_column]
        .value_counts(normalize=True)
        .unstack(hue_column)
        .fillna(0) * 100
    )

    counts = counts.loc[ordered_categories]
    percentages = percentages.loc[ordered_categories]

    ax2 = counts.plot.bar(ax=axes[1], title=pltname2)

    for container in ax2.containers:
        class_name = int(container.get_label())

        for bar, count, pct in zip(
            container,
            counts[class_name],
            percentages[class_name]
        ):
            ax2.annotate(
                f'{count}\n{pct:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 10),
                textcoords="offset points",
                ha='center', va='bottom'
            )

    plt.tight_layout()

def predict_and_plot(model_pipeline, inputs, targets, name=''):
    """
    Обчислює якість класифікаційної моделі та візуалізує матрицю невідповідностей.

    Parameters
    ----------
    model_pipeline : sklearn Pipeline
        Навчений пайплайн моделі.
    inputs : array-like або pandas.DataFrame
        Ознаки (features) для прогнозування.
    targets : array-like
        Справжні значення цільової змінної.
    name : str, optional
        Назва моделі або датасету для відображення виводу.

    Description
    -----------
    Функція:
    - обчислює ROC-AUC на основі ймовірностей,
    - будує нормалізовану матрицю невідповідностей,
    - виводить основну метрику якості моделі.

    Порог класифікації фіксований і дорівнює 0.5.
    """
    preds = model_pipeline.predict(inputs)
    probs = model_pipeline.predict_proba(inputs)[:, 1]

    roc_auc = roc_auc_score(targets, probs)
    print(f"Area under ROC score on {name} dataset: {roc_auc*100:.2f}%")

    confusion_matrix_ = confusion_matrix(targets, preds, normalize='true')

    plt.figure()
    sns.heatmap(confusion_matrix_, annot=True, cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title(f'{name} Confusion Matrix')
    plt.show()

    return preds

def predict_and_plot_with_threshold(
    model_pipeline,
    inputs,
    targets,
    threshold=0.5,
    name=''
):
    """
    Оцінює модель класифікації з користувацьким порогом прийняття рішення.

    Parameters
    ----------
    model_pipeline : sklearn Pipeline
        Навчений пайплайн моделі.
    inputs : array-like або pandas.DataFrame
        Ознаки (features) для прогнозування.
    targets : array-like
        Справжні значення цільової змінної.
    threshold : float, default=0.5
        Порог для перетворення ймовірностей у класи.
    name : str, optional
        Назва моделі або експерименту.

    Description
    -----------
    Функція:
    - отримує ймовірності класу,
    - формує передбачення за заданим порогом,
    - обчислює ROC-AUC, Precision, Recall та F1-score,
    - будує нормалізовану матрицю невідповідностей.

    Використовується для аналізу впливу порогу класифікації
    на якість моделі та баланс між precision і recall.
    """
    probs = model_pipeline.predict_proba(inputs)[:, 1]
    preds = (probs >= threshold).astype(int)

    roc_auc = roc_auc_score(targets, probs)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)

    print(f'\n{name}')
    print(f'Threshold: {threshold}')
    print(f'ROC-AUC: {roc_auc*100:.2f}%')
    print(f'Precision: {precision*100:.2f}%')
    print(f'Recall: {recall*100:.2f}%')
    print(f'F1-score: {f1*100:.2f}%')

    cm = confusion_matrix(targets, preds, normalize='true')

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title(f'{name} Confusion Matrix (threshold={threshold})')
    plt.show()

    return preds, probs
