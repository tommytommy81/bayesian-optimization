def plot_investment_decisions(data):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.countplot(x='investment_outcome', data=data)
    plt.title('Investment Decisions Distribution')
    plt.xlabel('Investment Outcome (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.show()

def summarize_data(data):
    return data.describe()

def check_missing_values(data):
    return data.isnull().sum()