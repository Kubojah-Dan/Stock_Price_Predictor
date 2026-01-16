import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, labels=[0,1]):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_cumulative_returns(df):
    plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df['cumulative_strategy'], label='Strategy')
    plt.plot(df['Date'], df['cumulative_market'], label='Market')
    plt.legend()
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.show()
