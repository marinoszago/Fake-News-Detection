import matplotlib.pyplot as plt
import time


def createStatementTypePie(claim_count, attack_count, flip_count):
    labels = 'Claim(' + str(claim_count) + ')', \
             'Attack(' + str(attack_count) + ')', \
             'Flip(' + str(flip_count) + ')'

    sizes = [claim_count, attack_count, flip_count]
    explode = (0, 0, 0)

    fig1, ax1 = plt.subplots()

    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')

    total = sum(sizes)
    plt.legend(
        loc='upper left',
        labels=['%s, %1.1f%%' % (
            l, (float(s) / total) * 100) for l, s in zip(labels, sizes)],
        prop={'size': 11},
        bbox_to_anchor=(0.0, 1),
        bbox_transform=fig1.transFigure
    )

    plt.show()


def ShowPie_ByAnnotation(pf_count,ff_count,hf_count,nf_count,f_count,bt_count,ht_count,mt_count,t_count):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Pants-Fire('+str(pf_count)+')', \
             'Full-Flop('+str(ff_count)+')', \
             'Half-Flip('+str(hf_count)+')', \
             'No-Flip('+str(nf_count)+')', \
             'False('+str(f_count)+')', \
             'Barely-True('+str(bt_count)+')', \
             'Half-True('+str(ht_count)+')', \
             'Mostly-True('+str(mt_count)+')', \
             'True('+str(t_count)+')'

    sizes = [pf_count, ff_count, hf_count, nf_count, f_count, bt_count, ht_count, mt_count, t_count]
    explode = (0, 0, 0, 0, 0.1, 0, 0, 0, 0.1)

    fig1, ax1 = plt.subplots()

    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')


    total = sum(sizes)
    plt.legend(
        loc='upper left',
        labels=['%s, %1.1f%%' % (
            l, (float(s) / total) * 100) for l, s in zip(labels, sizes)],
        prop={'size': 11},
        bbox_to_anchor=(0.0, 1),
        bbox_transform=fig1.transFigure
    )

    plt.show()


if __name__ == "__main__":
    print("Running pie_chart.py as single file... Please wait for the chart to show up!")
    time.sleep(1)

    """Show with dummy data"""
    ShowPie_ByAnnotation(100, 450, 50, 77, 300, 690, 236, 40, 176)
