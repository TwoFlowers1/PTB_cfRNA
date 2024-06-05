import pandas as pd
import statsmodels.api as sm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

def compute_p_value(y, X, included, new_column):
    try:
        model = sm.Logit(y, X[included + [new_column]]).fit(disp=0)
        return new_column, model.pvalues[new_column]
    except:
        return new_column, None

def forward_regression(X, y, initial_list=[], threshold_in=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {executor.submit(compute_p_value, y, X, included, new_column): new_column for new_column in excluded}
            for future in as_completed(futures):
                new_column, p_value = future.result()
                if p_value is not None:
                    new_pval[new_column] = p_value

        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f'Add {best_feature} with p-value {best_pval}')

        if not changed:
            break

    return included

data = pd.read_csv("train.exp.group.txt", sep='\t', header=0, low_memory=False)
data.pop("gene_name")
Y_train = data.pop("group")
X_train = (data - data.min()) / (data.max() - data.min())

selected = forward_regression(X_train, Y_train)
print("final result:", selected)
