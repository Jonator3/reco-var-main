from analyze import *


subsample_sizes = [3000, 2000, 1000, 500]
subsample_count = 10


def run_task(lang, var, df_prompt):
    print(lang, var, len(df_prompt))

    cb_vals = [corpus_based_measures.get(m)(df_prompt) for m in corpus_based_measures.keys()]
    tb_vals = [getMeanMeasureValues(df_prompt, m) for m in text_based_measures.keys()]

    return [lang, var, len(df_prompt)] + cb_vals + tb_vals


if __name__ == "__main__":
    tasks_args = []

    df_var = pd.DataFrame(columns=["Language", "Variable", "numAnswers"] + [m for m in corpus_based_measures.keys()] + ["mean" + m for m in text_based_measures.keys()])

    lang = "en"
    for subsample_size in subsample_sizes:
        for var in df['Variable'].unique():
            for _ in range(subsample_count):
                df_prompt = df[(df['Language'] == lang) & (df['Variable'] == var)]
                df_prompt = df_prompt.sample(subsample_size)
                tasks_args.append((lang, var, df_prompt))

    results = parallelprozessing.run_parallel(run_task, tasks_args)

    for res in results:
        df_var.loc[len(df_var.index)] = res
    df_var.to_csv('variance-en-sub.tsv', sep="\t")