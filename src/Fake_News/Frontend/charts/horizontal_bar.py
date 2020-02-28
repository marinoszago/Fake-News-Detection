import numpy as np
import matplotlib.pyplot as plt


def showBarBySubjectCount(statements_count_arr, subjects_arr):
    # Fixing random state for reproducibility

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # Example data
    subjects = subjects_arr
    y_pos = np.arange(len(subjects))
    performance = statements_count_arr
    error = np.random.rand(len(subjects))

    ax.barh(y_pos, performance, xerr=error, align='center',
            color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(subjects)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Statements')
    ax.set_title('Statements with count >= 500 per Subject')

    plt.show()


if __name__ == "__main__":
    print("Running horizontal_bar.py as single file... Please wait for the chart to show up!")

