import pandas as pd
from sklearn.feature_selection import r_regression
import os
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes

# function for setting the colors of the box plots pairs
box_elements = ['boxes', 'caps', 'whiskers', 'medians']
box_colors = ['#eba176', '#84cacc', '#c6cc4a', '#da6088', '#aaaaaa']


def setBoxColors(bp):
    index_1 = 0
    index_2 = 0
    for ci in range(len(bp['boxes'])):
        c = box_colors[ci]
        setp(bp['boxes'][index_1], color=c)
        setp(bp['caps'][index_2], color=c)
        setp(bp['caps'][index_2 + 1], color=c)
        setp(bp['whiskers'][index_2], color=c)
        setp(bp['whiskers'][index_2 + 1], color=c)
        setp(bp['medians'][index_1], color=c)
        index_1 += 1
        index_2 += 2


def stuff_str(s, l, c=" ", append_left=False):
    while (len(s) < l):
        if append_left:
            s = c + s
        else:
            s += c
    return s


def print_mat(mat):
    for row in mat:
        print(" | ".join([stuff_str(str(round(val, 3)), 6) for val in row]))


def plot_df(df, seperator_col, value_cols, negatives=[], limit_y=True):
    prompts = [val for val in df[seperator_col].unique()]

    # data to plot
    data_dict = {}

    for p in prompts:
        df_p = df[df[seperator_col] == p]
        values = []
        for col in value_cols:
            if negatives.__contains__(col):
                values.append([1-val for val in df_p[col]])
            else:
                values.append([val for val in df_p[col]])

        data_dict[p] = values

    legend_space = 3
    fig = figure(figsize=(len(prompts)+legend_space, 8))
    fig.subplots_adjust(bottom=0.25, right=1-(legend_space/(len(prompts)+legend_space)), top=0.95, left=1/(len(prompts)+legend_space))
    ax = axes()

    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='red')

    i = 1
    for label in data_dict.keys():
        bp = boxplot(data_dict[label], positions=[i + off for off in range(len(value_cols))], widths=0.6,
                     flierprops={'marker': 'o', 'markersize': 3},
                     showmeans=True, meanprops=meanpointprops)
        i += 6
        setBoxColors(bp)

    # set axes limits and labels
    xlim(0, len(data_dict.keys())*6)
    if limit_y:
        ylim(0, 1)

    center = int(len(box_colors) / 2 + 1)
    x_ticks = []
    for i in range(1, len(data_dict.keys()) + 1):
        x_ticks.append(center * (2 * i - 1))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(data_dict.keys(), rotation=45, fontsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=15)

    # draw temporary green and 'orange' lines and use them to create a legend
    h1, = plot([1, 1], '-', color=box_colors[0])
    h2, = plot([1, 1], '-', color=box_colors[1])
    h3, = plot([1, 1], '-', color=box_colors[2])
    h4, = plot([1, 1], '-', color=box_colors[3])
    h5, = plot([1, 1], '-', color=box_colors[4])
    legend_list = []
    for col in value_cols:
        if negatives.__contains__(col):
            legend_list.append("1-"+col)
        else:
            legend_list.append(col)
    outlier, = plot([1, 1], 'o', markeredgecolor='black', markerfacecolor='white')
    mean, = plot([1, 1], 'D', markeredgecolor='black', markerfacecolor='red')
    legend((h1, h2, h3, h4, h5, outlier, mean), legend_list, loc=(1 + (0.2/len(prompts)), 0), fontsize=15)
    h1.set_visible(False)
    h2.set_visible(False)
    h3.set_visible(False)
    h4.set_visible(False)
    h5.set_visible(False)
    outlier.set_visible(False)
    mean.set_visible(False)

    try:
        os.mkdir("plots")
    except FileExistsError:
        pass

    title = "_".join(legend_list) + "_separate_by_" + seperator_col
    savefig("plots/"+title+".png")
    show()


def pearson_mat(df):
    features = df.values.tolist()
    p_mat = []

    for i in range(len(df.columns)):
        target = [f[i] for f in features]
        p_mat.append(list(r_regression(features, target)))

    return p_mat


if __name__ == "__main__":
    print("Running test.py")
    features = ["FractionUnique", "meanLevenstein", "meanGST", "meanLCS", "meanVCos"]
    negative_features = ["meanGST", "meanVCos", "meanLCS"]

    df = pd.read_csv("variance.tsv", sep="\t")
    df_feature = df[features]

    print(features)
    print_mat(pearson_mat(df_feature))

    plot_df(df, "Language", features, negative_features)
    plot_df(df, "Variable", features, negative_features)

    plot_df(df, "Language", ["meanLen"], limit_y=False)
    plot_df(df, "Variable", ["meanLen"], limit_y=False)



