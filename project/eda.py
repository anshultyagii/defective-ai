import matplotlib.pyplot as plt
import seaborn as sns
def plot_degradation(df):
    sns.lineplot(data=df, x='Aging_Hours', y='RDS_on_Ohm', hue='Run_ID')
    plt.title('Degradation over Time')
    plt.savefig('project/eda_plot.png')
    plt.close()
