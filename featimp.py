import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.utils import check_random_state
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from pandas.api.types import is_string_dtype, is_object_dtype
from pandas.api.types import is_categorical_dtype


### 1. clean bulldozer
# functions to help clean_bulldozer
def df_normalize_strings(df):
    for col in df.columns:
        if is_string_dtype(df[col]) or is_object_dtype(df[col]):
            df[col] = df[col].str.lower()
            df[col] = df[col].fillna(np.nan)  # make None -> np.nan
            df[col] = df[col].replace('none or unspecified', np.nan)
            df[col] = df[col].replace('none', np.nan)
            df[col] = df[col].replace('#name?', np.nan)
            df[col] = df[col].replace('', np.nan)


def extract_sizes(df, colname):
    df[colname] = df[colname].str.extract(r'([0-9.]*)', expand=True)
    df[colname] = df[colname].replace('', np.nan)
    df[colname] = pd.to_numeric(df[colname])


def df_string_to_cat(df):
    for col in df.columns:
        if is_string_dtype(df[col]):
            df[col] = df[col].astype('category').cat.as_ordered()


def df_cat_to_catcode(df):
    for col in df.columns:
        if is_categorical_dtype(df[col]):
            df[col] = df[col].cat.codes + 1


def fix_missing_num(df, colname):
    df[colname + '_na'] = pd.isnull(df[colname])
    df[colname].fillna(df[colname].median(), inplace=True)


# clean bulldozer dataset
def clean_bulldozer(path):
    df_raw = pd.read_feather(path)
    df = df_raw.copy()
    del df['MachineID']
    del df['SalesID']
    df['auctioneerID'] = df['auctioneerID'].astype(str)
    df_normalize_strings(df)
    extract_sizes(df, 'Tire_Size')
    extract_sizes(df, 'Undercarriage_Pad_Width')
    df_string_to_cat(df)
    df_cat_to_catcode(df)
    fix_missing_num(df, 'Tire_Size')
    fix_missing_num(df, 'Undercarriage_Pad_Width')
    df.loc[df.YearMade < 1950, 'YearMade'] = np.nan
    fix_missing_num(df, 'YearMade')
    df.loc[df.eval("saledate.dt.year < YearMade"), 'YearMade'] = df[
        'saledate'].dt.year
    df.loc[df.eval(
        "MachineHoursCurrentMeter==0"), 'MachineHoursCurrentMeter'] = np.nan
    fix_missing_num(df, 'MachineHoursCurrentMeter')
    X, y = df.drop(['SalePrice', 'saledate'], axis=1), df['SalePrice']
    return X, y


def drop_bool_object(df):
    cols = [col for col in df.columns if df[col].dtypes in ['bool', 'object']]
    df = df.drop(df[cols], axis=1)
    return df

### 2.plot
# scale the range before plot, if necessary (for PCA, SPEARMAN, mRMR)
def scale(x, out_range=(0, 0.8)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


# plot I
def plot_imp(I, X, top_n=10):
    df_imp = pd.DataFrame({"feature": X.columns, "importance": I})
    df_imp = df_imp.sort_values('importance', ascending=False)[:top_n]
    # plt.figure(figsize=(15, 10))
    ax = sns.barplot(data=df_imp, x="importance", y="feature",
                     palette="Blues_d")
    ax.set_ylabel('')
    ax.set_xlabel('')


# plot I with std
def plot_std(df_mRMR):
    ax = sns.barplot(data=df_mRMR, x="I", y="features",ci="sd",)
    ax.set_ylabel('')
    ax.set_xlabel('')


# plot importances from different methods for lgb and linear models
def plot_lgb(k,X,y,I_pca,I_spearman,I_mRMR,I_hierarchy,I_sklearn,I_permutation_oob,I_dropcol_oob):
    df = pd.DataFrame({'k':np.arange(1,k+1)
                       , 'mae_pca':mae_I(X,y,sort_I(I_pca),k)
                       ,'mae_spearman':mae_I(X,y,sort_I(I_spearman),k)
                       ,'mae_mRMR':mae_I(X,y,sort_I(I_mRMR),k)
                       ,'mae_hierarchy':mae_I(X,y,sort_I(I_hierarchy),k)
                       ,'mae_sklearn':mae_I(X,y,sort_I(I_sklearn),k)
                       ,'mae_permutation_oob':mae_I(X,y,sort_I(I_permutation_oob),k)
                       ,'mae_dropcol_oob':mae_I(X,y,sort_I(I_dropcol_oob),k)
                      })
    sns.lineplot(df['k'],df['mae_pca'],marker='o', label='pca')
    sns.lineplot(df['k'],df['mae_spearman'],marker='p', label='spearman')
    sns.lineplot(df['k'],df['mae_mRMR'],marker='1', label='mRMR')
    sns.lineplot(df['k'],df['mae_hierarchy'],marker='s', label='hierarchy')
    sns.lineplot(df['k'],df['mae_sklearn'],marker='d', label='sklearn')
    sns.lineplot(df['k'],df['mae_permutation_oob'],marker='*', label='permutation')
    sns.lineplot(df['k'],df['mae_dropcol_oob'],marker=11, label='dropcol')
    plt.ylabel('20% 5fold CV MAE ($)')
    plt.title('lgb model')


def plot_linear(k,X,y,I_pca,I_spearman,I_mRMR,I_hierarchy,I_sklearn,I_permutation_oob,I_dropcol_oob):
    df_linear = pd.DataFrame({'k':np.arange(1,k+1)
                       , 'mae_pca':mae_I_linear(X,y,sort_I(I_pca),k)
                       ,'mae_spearman':mae_I_linear(X,y,sort_I(I_spearman),k)
                       ,'mae_mRMR':mae_I(X,y,sort_I(I_mRMR),k)
                       ,'mae_hierarchy':mae_I_linear(X,y,sort_I(I_hierarchy),k)
                       ,'mae_sklearn':mae_I_linear(X,y,sort_I(I_sklearn),k)
                       ,'mae_permutation_oob':mae_I_linear(X,y,sort_I(I_permutation_oob),k)
                       ,'mae_dropcol_oob':mae_I_linear(X,y,sort_I(I_dropcol_oob),k)
                      })
    sns.lineplot(df_linear['k'],df_linear['mae_pca'],marker='o', label='pca')
    sns.lineplot(df_linear['k'],df_linear['mae_spearman'],marker='p', label='spearman')
    sns.lineplot(df_linear['k'],df_linear['mae_mRMR'],marker='1', label='mRMR')
    sns.lineplot(df_linear['k'],df_linear['mae_hierarchy'],marker='s', label='hierarchy')
    sns.lineplot(df_linear['k'],df_linear['mae_sklearn'],marker='d', label='sklearn')
    sns.lineplot(df_linear['k'],df_linear['mae_permutation_oob'],marker='*', label='permutation')
    sns.lineplot(df_linear['k'],df_linear['mae_dropcol_oob'],marker=11, label='dropcol')
    plt.ylabel('20% 5fold CV MAE ($)')
    plt.title('linear model')


### 3.pca
def pca_importance(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X_transform = scaler.transform(X)
    pca = PCA()
    x_new = pca.fit_transform(X_transform)
    pca_ratio = pca.explained_variance_ratio_
    # each pc direction explained how much of the data
    pca_features = abs( pca.components_ )
    # line1 is pc1, it's consisted by different portion of features
    normed_pca_features = normalize(pca_features, axis=1, norm='l1')
    I = np.dot(pca_ratio,normed_pca_features)
    return I

### 4.mRMR method
# choose the first element
def init_I(X_num, y):
    init_I = []
    for i in range(X_num.shape[1]):
        X_k = X_num.iloc[:, i]
        # the first term
        rho_k, pval_k = spearmanr(X_k, y)
        rho_k = abs(rho_k)
        # the second term
        sum_I_S = 0
        ct_S = 0
        for j in range(X_num.shape[1]):
        # look it's correlation with all the other cols
            if j != i:
                rho, pval = spearmanr(X_k, X_num.iloc[:, j])
                sum_I_S += abs(rho)
                ct_S += 1
        if ct_S == 0:
            init_I.append(rho_k)
        else:
            init_I.append(rho_k - sum_I_S / ct_S)
    return init_I


# choose all the 10 elements
def mRMR_S(X_num, y, num_features=10):
    # select 10 features in total
    I = init_I(X_num, y)
    S = []
    S.append(np.argmax(I))
    # take the col that has biggest init I, as the first element
    for m in range(1, num_features):  # add another element in each loop
        mRMRs = [-np.inf] * X_num.shape[1]
        # if not changed, it would be -inf, impossible to be picked in argmax
        for i in range(X_num.shape[1]):
            if i not in S:  # only consider to add those not in S
                X_k = X_num.iloc[:, i]
                # the first term
                rho_k, pval_k = spearmanr(X_k, y)
                rho_k = abs(rho_k)
                # the second term, only pick those in S
                sum_I_S = 0
                ct_S = 0
                for j in S:
                    # only look at the features already in S (we have m, we want to make m+1)
                    rho, pval = spearmanr(X_k, X_num.iloc[:, j])
                    sum_I_S += abs(rho)
                    ct_S += 1
                if ct_S == 0:
                    mRMRs[i] = (rho_k)
                else:
                    mRMRs[i] = (rho_k - sum_I_S / ct_S)
        S.append(np.argmax(mRMRs))
    return S


# get its importance
def final_S_I(X_num, y, S):
    # recalculate final I with these features in S
    final_S_I = []
    for k in S:
        X_k = X_num.iloc[:, k]
        # the first term
        rho_k, pval_k = spearmanr(X_k, y)
        rho_k = abs(rho_k)
        # the second term
        sum_I_S = 0
        ct_S = 0
        for j in S:  # only look at the features already in S  (we have m-1 features)
            if j != k:
                rho, pval = spearmanr(X_k, X_num.iloc[:, j])
                sum_I_S += abs(rho)
                ct_S += 1
        if ct_S == 0:
            final_S_I.append(rho_k)
        else:
            final_S_I.append(rho_k - sum_I_S / ct_S)
    return final_S_I


# use hierarchy to get the subsets
def feature_hierarchy(X_num,depth):
    corr = spearmanr(X_num).correlation
    corr_linkage = hierarchy.ward(corr)
    cluster_ids = hierarchy.fcluster(corr_linkage, depth, criterion='distance')
    # bigger depth, less cluster
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    feature_groups = [v for v in cluster_id_to_feature_ids.values()]
    return feature_groups


def k_rho(X_num, y,feature_groups):
    k_all = []
    rho_all = []
    for i in range(len(feature_groups)):
        S = feature_groups[i]
        rho = []
        for k in S:
            X_k = X_num.iloc[:, k]
            rho_k = abs(spearmanr(X_k, y)[0])
            rho_others = [abs(spearmanr(X_k, X_num.iloc[:, j])[0]) for j in S if j!=k]
            rho_k -= np.mean(rho_others)
            rho.append(rho_k)
        selected = np.argmax(rho)
        selected_k = S[selected]
        selected_rho = rho[selected]
        k_all.append(selected_k)
        rho_all.append(selected_rho)
    return k_all, rho_all


# link I with features
def new_I(I,features, X):
    new_I = [0]*X.shape[1]
    for i, feature_i in enumerate(features):
        new_I[feature_i] = I[i]
    return new_I

### 6. dropcol method
# dropcol with metric like r_square
def dropcol_importances(model, metric, X_train, y_train):
    model_ = clone(model)
    model_.fit(X_train, y_train)
    baseline = metric(y_train, model.predict(X_train))
    imp = []
    for col in X_train.columns:
        X_new = X_train.drop(col, axis=1)
        model_ = clone(model)
        model_.fit(X_new, y_train)
        m = metric(y_train, model_.predict(X_new))
        imp.append(baseline - m)
    return imp

# rfpimp, dropcol with oob
def dropcol_importances_oob(rf, X_train, y_train):
    rf_ = clone(rf)
    rf_.fit(X_train, y_train, )
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X_new = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.fit(X_new, y_train)
        o = rf_.oob_score_
        imp.append(baseline - o)
    return imp


### 7.permutation method
# sklearn function
def _generate_sample_indices(random_state, n_samples):
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)
    return sample_indices


def _generate_unsampled_indices(random_state, n_samples):
    """ updated for n_samples_bootstrap, so I rewrite it
        """
    sample_indices = _generate_sample_indices(random_state, n_samples)
    sample_counts = np.bincount(sample_indices, minlength=n_samples)
    unsampled_mask = sample_counts == 0
    indices_range = np.arange(n_samples)
    unsampled_indices = indices_range[unsampled_mask]
    return unsampled_indices


# rfpimp function with oob score
def oob_classifier_accuracy(rf, X_train, y_train):
    X = X_train.values
    y = y_train.values
    n_samples = len(X)
    n_classes = len(np.unique(y))
    predictions = np.zeros((n_samples, n_classes))
    for tree in rf.estimators_:
        unsampled_indices = _generate_unsampled_indices(tree.random_state,
                                                        n_samples)
        tree_preds = tree.predict_proba(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds
    predicted_class_indexes = np.argmax(predictions, axis=1)
    predicted_classes = [rf.classes_[i] for i in predicted_class_indexes]
    oob_score = np.mean(y == predicted_classes)
    return oob_score


def oob_regression_r2_score(rf, X_train, y_train):
    X = X_train.values
    y = y_train.values
    n_samples = len(X)
    predictions = np.zeros(n_samples)
    n_predictions = np.zeros(n_samples)
    for tree in rf.estimators_:
        unsampled_indices = _generate_unsampled_indices(tree.random_state,
                                                        n_samples)
        tree_preds = tree.predict(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds
        n_predictions[unsampled_indices] += 1
    if (n_predictions == 0).any():
        n_predictions[n_predictions == 0] = 1
    predictions /= n_predictions
    oob_score = r2_score(y, predictions)
    return oob_score


def permutation_importances_oob(rf, oob, X_train, y_train):
    rf.fit(X_train, y_train)
    baseline = oob(rf, X_train, y_train)
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        o = oob(rf, X_train, y_train)
        X_train[col] = save
        imp.append(baseline - o)
    return np.array(imp)


# with simple metric
def permutation_importances(model, metric, X_train, y_train):
    model.fit(X_train, y_train)
    baseline = metric(y_train, model.predict(X_train))
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = metric(y_train, model.predict(X_train))
        X_train[col] = save
        imp.append(baseline - m)
    return imp

### 8. Summarize : compare strategies
def sort_I(I):
    I_sorted = sorted(enumerate(I), key=lambda x: x[1], reverse=True)
    return I_sorted

def sort_mean_I(I):
    I_sorted = sorted(enumerate(I), key=lambda x: np.mean(x[1]), reverse=True)
    return I_sorted

def mae_I(X, y, sorted_I, k_max):
    n_iters = 5
    features = list(X.columns)
    mae = []
    for k in range(k_max):
        selected_index = [sorted_I[i][0] for i in range(k+1)]
        selected_feature = [features[i] for i in selected_index]
        errs = []
        for i in range(n_iters):
            model, err = lgb_model(X[selected_feature], y)
            errs.append(err)
        mae.append(np.mean(errs))
    return mae


def mae_I_linear(X,y,sorted_I,k_max):
    features = list(X.columns)
    mae = []
    reg = LinearRegression()
    for k in range(k_max):
        selected_index = [sorted_I[i][0] for i in range(k+1)]
        selected_feature = [features[i] for i in selected_index]
        errs = -cross_val_score(reg,scoring='neg_mean_absolute_error', X=X[selected_feature], y=y, cv=5)
        mae.append(np.mean(errs))
    return mae


def lgb_model(X,y):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmsle',
        'max_depth': 6,
        'learning_rate': 0.1,
        'verbose': 1}
    x_train, x_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=1)
    d_train = lgb.Dataset(x_train, label=y_train)
    d_valid = lgb.Dataset(x_valid, label=y_valid)
    watchlist = [d_valid]
    model = lgb.train(params, d_train, 100, watchlist)
    err = mean_absolute_error(y_valid, model.predict(x_valid))
    return model, err

### 9. Variance
def I_boostrap_permutation(model,n,X,y):
    I = []
    for i in range(n):
        X_subsample = X.sample(frac=0.7)
        y_subsample = y[X_subsample.index]
        I_permutation_oob = permutation_importances_oob(model, oob_regression_r2_score, X_subsample, y_subsample )
        I.append(I_permutation_oob)
    return I

def re_order(I_20):
    I = []
    for feature_number in range(len(I_20[0])):
        I.append([I_20[i][feature_number] for i in range(20)])
    return I

flatten = lambda l: [item for sublist in l for item in sublist]


### 10. p-value
def display_distributions(actual_imp_df_, null_imp_df_, feature_):
    """
    refer to
    https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
    """
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_split'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_split'].mean(),
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (split) Distribution for %s ' % feature_.upper())
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_gain'].mean(),
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Gain Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (gain) Distribution for %s ' % feature_.upper())

def lgb_feature_importances(X,y,shuffle, seed=None):
    if shuffle:
        y = y.copy().sample(frac=1.0)
    model,validation_err = lgb_model(X,y)
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(X.columns)
    imp_df["importance_gain"] = model.feature_importance(importance_type='gain')
    imp_df["importance_split"] = model.feature_importance(importance_type='split')
    return imp_df