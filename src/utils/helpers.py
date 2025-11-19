from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#import graphviz
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import shap
import streamlit as st
import umap.umap_ as umap
import xgboost as xgb
from plotly.subplots import make_subplots
from scipy.stats import kruskal, mannwhitneyu, ranksums, ttest_ind
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from streamlit_plotly_events import plotly_events

# Import Python implementations instead of R
from python_support import (
    beta_dim_red,
    cut_var,
    perform_ancom_alternative,
    shannon,
    simpson,
)

pio.templates.default = "plotly"

def filter_by_correlation(matrix, threshold=0.95):
    cols = matrix.columns
    corr_matrix = matrix.corr().abs()
    #upper matrix to not remove both features (symetric correlation)
    upper = pd.DataFrame(np.triu(corr_matrix, k=1))
    #remove feature at index i if correlated with any other feature at 0.95 at least
    features_to_remove = [cols[i] for i,col in enumerate(upper.columns) if any(upper[col] > threshold)]
    print(f'{len(features_to_remove)} features among {len(cols)} are highly correlated with other and can be removed.')
    keep_cols = list(set(cols).difference(features_to_remove))
    uncorrelated_matrix = matrix[keep_cols]
    return uncorrelated_matrix

def make_ratio(data):
    sums = data.sum(axis=1)
    data = data.div(sums, axis=0)
    return data

def centered_log_ratio(data):
    data = data + 1
    geometric_means = np.exp(np.log(data).mean(axis=1)) # this is mathematically equivalent to doing nth root of product of features
    # divide each element of the data by the geometric mean for its corresponding sample
    data_clr = data.div(geometric_means, axis=0)
    # take the natural logarithm of the CLR-transformed data
    return np.log10(data_clr)


# Load data
@st.cache_data(experimental_allow_widgets=True)
def load_data():
    last_level = st.radio(
        "What\'s the last level of the taxonomy that data has?",
        ('kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'))
    
    level_to_int = {'kingdom': 1, 'phylum': 2, 'class': 3, 'order': 4, 'family': 5, 'genus': 6, 'species': 7}
    last_level = level_to_int[last_level]

    otu_separator = st.radio(
    "What\'s the separator used to separate the toxonomy levels in the taxonomy column in the OTU file ?",
    (';', '/', '-', ',', 'other'))

    st.caption('e.g. If the taxa in your OTU file has the following format: Bacteria;Proteobacteria;Gammaproteobacteria;Burkholderiales;Comamonadaceae;Curvibacter;gracilis')
    st.caption('-> The separator is ";"')

    if otu_separator == 'other':
        otu_separator = st.text_input('Please specify the separator used for the toxonomy in the OTU file:')
    st.success('You selected this separator : ' + otu_separator)

    st.write('The website supports OTU files with can be read counts data or relative abundance data. Choose which type of data are you using:')
    otu_type = st.radio("What's the type of the OTU data ?", ('Read counts', 'Relative abundance'))

    st.header('Upload your data')
    int_to_str_var = dict({0: 0, 1: 1})
    col1, col2 = st.columns(2)
    with col1:
        metadata_bool = False
        st.write('Upload your metadata file')
        metadata_raw = st.file_uploader("Choose a file", type='csv', key='mt')
        if metadata_raw is not None:
            metadata_raw = pd.read_csv(metadata_raw)
            if 'bin_var' not in metadata_raw.columns or 'id' not in metadata_raw.columns: 
                st.error('Your metadata file must have a column named "bin_var" for the binary variable your\'re studying and a column named "id" for the sample ids')
                st.write('Your metadata file has the following columns:')
                st.write(metadata_raw.columns) 
            else:
                metadata_raw = metadata_raw[['bin_var', 'id']].astype({'id': str}).set_index('id')
                if metadata_raw['bin_var'].nunique() > 2:
                    st.error('Your binary variable must have only 2 unique values while it has the following values:')
                    st.error(list(metadata_raw['bin_var'].unique()))
                else: 
                    if sorted(metadata_raw['bin_var'].unique()) == [0, 1]: 
                        st.success('File uploaded successfully')
                        metadata_bool = True
                    else:
                        st.write('Your binary variable has values different than 0/1 (needed for AI part). Please assign them manually to 0 and 1 in the following table, then press enter on your keyboard:')
                        frame = pd.DataFrame(np.array([metadata_raw['bin_var'].unique(), [None, None]]).T, columns = ['initial values', 'bin_var'])
                        frame.index.names = ['row number (ignore)']
                        vals = st.experimental_data_editor(frame)

                        if all(vals['bin_var']): # all not null
                            try:
                                vals = vals.astype({'bin_var': int})
                                if sorted(vals['bin_var'].unique()) == [0, 1]:
                                    int_to_str_var = dict({int(vals['bin_var'][0]): vals['initial values'][0], int(vals['bin_var'][1]): vals['initial values'][1]})
                                    metadata_raw['bin_var'] = metadata_raw['bin_var'].map(vals.set_index('initial values')['bin_var'])
                                    st.success('Your binary variable has been successfully assigned to 0 and 1')
                                    st.success('File uploaded successfully')
                                    metadata_bool = True
                            except:
                                st.write('Put only 0 and 1 in the table.')


    with col2:
        st.write('Upload your microbiome file')
        mcb_raw = st.file_uploader("Choose a file", type='csv', key='mcr')
        if mcb_raw is not None:
            mcb_raw = pd.read_csv(mcb_raw)
            if 'taxonomy' not in mcb_raw.columns:
                st.error('Your metadata file must have a column named "taxonomy"')

            if mcb_raw is not None:
                st.success('File uploaded successfully')
     
    proceed = metadata_raw is not None and mcb_raw is not None and metadata_bool
    return metadata_raw, mcb_raw, proceed, otu_separator, otu_type, int_to_str_var, level_to_int, last_level

# Separate data to taxa levels
@st.cache_data
def mcb_to_taxa(mcb_raw, metadata_raw, otu_separator, level_to_int, last_level):
    mcb = mcb_raw.copy()
    metadata = metadata_raw.copy()
    all_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    try:
        mcb[all_levels[:last_level]] = mcb['taxonomy'].str.split(otu_separator, expand=True)
    except:            
        st.error('If you\'re trying to upload new data, make sure to choose the right separator for your OTU file. If you did, the error probably comes from not choosing the right last taxa level.')
        st.stop()

    mcb = mcb.drop(columns=['taxonomy'], axis=1)
    
    mcbs = []

    # group by kingdom
    if last_level >= level_to_int['kingdom']:
        mcb_k = mcb.iloc[:, 0:len(mcb.columns)-last_level]
        mcb_k['kingdom'] = mcb['kingdom']
        mcb_k = mcb_k.groupby(mcb_k.columns[-1]).sum().transpose()
        mcb_k = mcb_k.join(metadata, how='inner')
        mcb_k.index.names = ['id']
        mcbs.append(mcb_k)

    # group by phylum
    if last_level >= level_to_int['phylum']:
        mcb_p = mcb.iloc[:, 0:len(mcb.columns)-last_level]
        mcb_p['phylum'] = mcb['phylum']
        mcb_p = mcb_p.groupby(mcb_p.columns[-1]).sum().transpose()
        mcb_p = mcb_p.join(metadata, how='inner')
        mcb_p.index.names = ['id']
        mcbs.append(mcb_p)

    # group by class
    if last_level >= level_to_int['class']:
        mcb_c = mcb.iloc[:, 0:len(mcb.columns)-last_level]
        mcb_c['class'] = mcb['class']
        mcb_c = mcb_c.groupby(mcb_c.columns[-1]).sum().transpose()
        mcb_c = mcb_c.join(metadata, how='inner')
        mcb_c.index.names = ['id']
        mcbs.append(mcb_c)

    # group by order
    if last_level >= level_to_int['order']:
        mcb_o = mcb.iloc[:, 0:len(mcb.columns)-last_level]
        mcb_o['order'] = mcb['order']
        mcb_o = mcb_o.groupby(mcb_o.columns[-1]).sum().transpose()
        mcb_o = mcb_o.join(metadata, how='inner')
        mcb_o.index.names = ['id']
        mcbs.append(mcb_o)

    # group by family
    if last_level >= level_to_int['family']:
        mcb_f = mcb.iloc[:, 0:len(mcb.columns)-last_level]
        mcb_f['family'] = mcb['family']
        mcb_f = mcb_f.groupby(mcb_f.columns[-1]).sum().transpose()
        mcb_f = mcb_f.join(metadata, how='inner')
        mcb_f.index.names = ['id']
        mcbs.append(mcb_f)

    # group by genus
    if last_level >= level_to_int['genus']:
        mcb_g = mcb.iloc[:, 0:len(mcb.columns)-last_level]
        mcb_g['genus'] = mcb['genus']
        mcb_g = mcb_g.groupby(mcb_g.columns[-1]).sum().transpose()
        mcb_g = mcb_g.join(metadata, how='inner')
        mcb_g.index.names = ['id']
        mcbs.append(mcb_g)

    # group by species
    if last_level >= level_to_int['species']:
        mcb_s = mcb.iloc[:, 0:len(mcb.columns)-last_level]
        mcb_s['species'] = mcb['species']
        mcb_s = mcb_s.groupby(mcb_s.columns[-1]).sum().transpose()
        mcb_s = mcb_s.join(metadata, how='inner')
        mcb_s.index.names = ['id']
        mcbs.append(mcb_s)

    
    return mcbs

# process data
@st.cache_data
def process_level_abundance(mcb, level, keepFeature_percentage, min_var): 
    '''
    keepFeature_percentage: minimum percentage of non-zero elements to allow in each column
    variance used to select features
    level: kingdom, order, family, genus, species, class, phylum
    '''
    mcb = mcb.drop(columns=['bin_var'], axis=1)

    min_nonzero = int((keepFeature_percentage/100) * mcb.shape[0])
    kept_features = [feature for feature in mcb.columns if (mcb[feature] > 0).sum() > min_nonzero]

    selector = VarianceThreshold(min_var)
    fitted_mcb = selector.fit(mcb)
    kept_features_fitted = np.array(mcb.columns)[fitted_mcb.get_support()]
    kept_features_fitted = list(set(kept_features_fitted).intersection(set(kept_features)))

    if level in ['kingdom', 'order', 'family', 'genus', 'species', 'class', 'phylum']:
        mcb = centered_log_ratio(make_ratio(mcb))
        mcb = mcb[kept_features_fitted]
        return mcb, kept_features_fitted
    elif level == 'phyla ratios':
        mcb = mcb[kept_features_fitted]
        phyla_pairs = list(combinations(mcb.columns, 2))
        ratios = pd.DataFrame()
        for p1, p2 in phyla_pairs:
            ratios[f'{p1}/{p2}'] = (mcb[p1]+1).div((mcb[p2]+1))
        return ratios
    
@st.cache_data
def process_level_rel_abundance(mcb, level, keepFeature_percentage, cutoff): 
    '''
    keepFeature_percentage: minimum percentage of non-zero elements to allow in each column
    variance used to select features
    level: kingdom, order, family, genus, species, class, phylum
    '''
    mcb = mcb.drop(columns=['bin_var'], axis=1)
    min_nonzero = int((keepFeature_percentage/100) * mcb.shape[0])
    kept_features = [feature for feature in mcb.columns if (mcb[feature] > 0).sum() >= min_nonzero]
    if cutoff > 0:
        # Use Python implementation instead of R
        support = cut_var(mcb, cutoff)
        kept_features_var = np.array(mcb.columns)[~support]  # cut_var returns True for features to remove
        kept_features = list(set(kept_features).intersection(set(kept_features_var)))
    if level in ['kingdom', 'order', 'family', 'genus', 'species', 'class', 'phylum']:
        mcb = centered_log_ratio(mcb)
        mcb = mcb[kept_features]
        return mcb, kept_features
    elif level == 'phyla ratios':
        mcb = mcb[kept_features]
        phyla_pairs = list(combinations(mcb.columns, 2)) 
        ratios = pd.DataFrame()
        for p1, p2 in phyla_pairs:
            ratios[f'{p1}/{p2}'] = (mcb[p1]+1).div((mcb[p2]+1))
        return ratios
    
def show_pvals(pvals, title):
    colorscale = [[0, 'green'], [0.05, 'green'], [0.05, 'grey'], [1, 'grey']]
    fig = go.Figure(data=go.Heatmap(
    z=pvals,
    colorscale=colorscale,
    zmin=0,  # Defines the value that corresponds to the first color of the colorscale
    zmax=1,  # Defines the value that corresponds to the last color of the colorscale
    showscale=True  # Show the color scale
    ))

    # Add annotations
    for i in range(len(pvals.columns)):
        for j in range(len(pvals.index)):

            fig.add_annotation(
                x=i,
                y=j,
                text=str(pvals.iat[j, i]),
                showarrow=False,
                font=dict(
                    size=16,
                    color="Black"
                )
            )

    # Add a title to the color bar
    fig['data'][0]['colorbar']['title'] = 'P-values'

    fig.update_layout(title_text = title , 
        xaxis = dict(
        tickmode = 'array',
        tickvals = list(range(len(pvals.columns))),
        ticktext = pvals.columns.tolist()
        ),
    yaxis = dict(
        tickmode = 'array',
        tickvals = list(range(len(pvals.index))),
        ticktext = pvals.index.tolist()
    )
    )    
    st.plotly_chart(fig, use_container_width=True, config=st.session_state.config, width=100, height=500)

@st.cache_data
def get_results(mcb, y, features, thresholds, folds, kernel, iters, n_neighbors, estimators, max_depth): #returns metrics

    X = pd.DataFrame([], columns = features, index=mcb.index)
    for i in range(len(X.columns)):
        X.iloc[:, i] = np.where(mcb[features[i]]<=thresholds[i],0,1)
    
    #############################              SVM                ####################################

    svm = SVC(probability=True, random_state=42, kernel=kernel)
    skf = StratifiedKFold(n_splits=folds)
    svm_results = []
    y_svm = pd.DataFrame(columns=y.columns)
    x_svm = pd.DataFrame(columns=X.columns)
    shaps = []

    for train_index, test_index in skf.split(X, y):
        x_train_fold, x_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        svm.fit(x_train_fold, y_train_fold)
        svm_results = svm_results + svm.predict_proba(x_test_fold).tolist()

        y_svm = pd.concat([y_svm , y_test_fold])
        x_svm =  pd.concat([x_svm, x_test_fold])
        explainer = shap.KernelExplainer(svm.predict, x_train_fold)
        shap_vals = explainer.shap_values(x_test_fold)
        shaps = shaps + shap_vals.tolist()

    x_svm = x_svm.astype(int)
    y_svm = y_svm.astype(int)
    svm_results, shaps = np.array(svm_results), np.array(shaps)
    fpr_s, tpr_s, thresholds_s = roc_curve(y, y_score=svm_results[:,1])


#############################              KNN                ####################################
    try:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        skf = StratifiedKFold(n_splits=folds)
        knn_results = []
        y_knn = pd.DataFrame(columns=y.columns)
        x_knn = pd.DataFrame(columns=X.columns)

        for train_index, test_index in skf.split(X, y):
            x_train_fold, x_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            knn.fit(x_train_fold, y_train_fold)
            knn_results = knn_results + knn.predict_proba(x_test_fold).tolist()
            y_knn =pd.concat([y_knn , y_test_fold])
            x_knn =  pd.concat([x_knn, x_test_fold])

        x_knn = x_knn.astype(int)
        y_knn = y_knn.astype(int)
        knn_results = np.array(knn_results)
        fpr_k, tpr_k, thresholds_k = roc_curve(y_knn, y_score=knn_results[:,1])

        
        #############################              LOG                ####################################
        log = LogisticRegression(max_iter=iters,random_state=42)
        skf = StratifiedKFold(n_splits=folds)
        log_results = []
        y_log = pd.DataFrame(columns=y.columns)
        x_log = pd.DataFrame(columns=X.columns)

        for train_index, test_index in skf.split(X, y):
            x_train_fold, x_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            log.fit(x_train_fold, y_train_fold)
            log_results = log_results + log.predict_proba(x_test_fold).tolist()
            y_log = pd.concat([y_log , y_test_fold])
            x_log = pd.concat([x_log , x_test_fold])

        x_log = x_log.astype(int)
        y_log = y_log.astype(int)
        log_results = np.array(log_results)
        fpr_l, tpr_l, thresholds_l = roc_curve(y, y_score=log_results[:,1])
        
        #############################              XGB                ####################################
        xgb_model = xgb.XGBRegressor()
        skf = StratifiedKFold(n_splits=folds)
        clf = GridSearchCV(xgb_model, {'max_depth': [max_depth],
                                    'n_estimators': [estimators]}, verbose=1, n_jobs=2)
        
        xg_results = []
        y_xg = pd.DataFrame(columns=y.columns)
        x_xg = pd.DataFrame(columns=X.columns)

        for train_index, test_index in skf.split(X, y):
            x_train_fold, x_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            clf.fit(x_train_fold, y_train_fold)
            xg_results = xg_results + clf.predict(x_test_fold).tolist()
                    
            y_xg = pd.concat([y_xg , y_test_fold])
            x_xg =  pd.concat([x_xg, x_test_fold])

        x_xg = x_xg.astype(int)
        y_xg = y_xg.astype(int)
        xg_results, shaps = np.array(xg_results), np.array(shaps)
        fpr_xg, tpr_xg, thresholds_xg = roc_curve(y, y_score=xg_results)

        optimal_xg = []
        for op in thresholds_xg:
            yp = np.where(xg_results>=op,1,0)
            optimal_xg.append(confusion_matrix(y_xg, yp)[0,1] + confusion_matrix(y_xg, yp)[1,0])

        optimal_xg = thresholds_xg[np.argmin(optimal_xg)]
        y_preds_xg = np.where(xg_results>=optimal_xg,1,0)
        confusion_xg = confusion_matrix(y_xg, y_preds_xg)
    except Exception as e:
            st.error('Check if the features output by the decision tree are unique, run the tree again to get new features. If the there\'s twice the same feature, the binary transform can\'t be performed.')
            raise e
    ####################################################################################################################
    
    optimal_svm = []
    for op in thresholds_s:
        yp = np.where(svm_results[:,1]>=op,1,0)
        optimal_svm.append(confusion_matrix(y_svm, yp)[0,1] + confusion_matrix(y_svm, yp)[1,0])

    optimal_knn = []
    for op in thresholds_k:
        yp = np.where(knn_results[:,1]>=op,1,0)
        optimal_knn.append(confusion_matrix(y_knn, yp)[0,1] + confusion_matrix(y_knn, yp)[1,0])

    optimal_log = []
    for op in thresholds_l:
        yp = np.where(log_results[:,1]>=op,1,0)
        optimal_log.append(confusion_matrix(y_log, yp)[0,1] + confusion_matrix(y_log, yp)[1,0])

    optimal_svm = thresholds_s[np.argmin(optimal_svm)]
    optimal_knn = thresholds_k[np.argmin(optimal_knn)]
    optimal_log = thresholds_l[np.argmin(optimal_log)]
    
    ####################################################################################################################
    y_preds_s = np.where(svm_results[:,1]>=optimal_svm,1,0)
    y_preds_k = np.where(knn_results[:,1]>=optimal_knn,1,0)
    y_preds_l = np.where(log_results[:,1]>=optimal_log,1,0)

    confusion_s = confusion_matrix(y_svm, y_preds_s)
    confusion_k = confusion_matrix(y_knn, y_preds_k)
    confusion_l = confusion_matrix(y_log, y_preds_l)

    ####################################################################################################################
    metrics = [accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, balanced_accuracy_score, cohen_kappa_score]
    preds = [y_preds_l, y_preds_s, y_preds_k, y_preds_xg]
    ys = [y_log, y_svm, y_knn, y_xg]
    vals=[]
    
    for i,pred in enumerate(preds):
        vals=vals+[[metric(ys[i], pred) for metric in metrics]]

    evals = pd.DataFrame(vals, columns=['accuracy', 'roc_auc', 'recall', 'precision', 'f1', 'balanced_accuracy','cohen_kappa'])
    evals = pd.concat([evals, pd.DataFrame([recall_score(ys[i],pred,pos_label=0) for i, pred in enumerate(preds)], columns=['specificity'])] ,axis=1)

    evals['clfs'] = ['log', 'svm', 'knn', 'xgboost']
    evals.set_index(['clfs'], inplace=True)
    evals = evals.round(3)

    #########################################   BOXPLOTS    ###############################################################

    box_plot = mcb[features]
    box_plot['bin_var'] = y['bin_var']

    return evals, box_plot, shaps, confusion_s, confusion_l, confusion_k, confusion_xg, features, x_svm

def show_confusions(confusion_s, confusion_l, confusion_k, confusion_xg):
    st.subheader('Confusion Matrices')
    st.caption('Confusion matrices of the results of each algorithm.')
    col1, col2 = st.columns(2, gap = 'large')
    col3, col4 = st.columns(2, gap = 'large')

    with col1:
        st.subheader('SVM')
        confusion_s = pd.DataFrame(confusion_s, columns=[f'predicted {st.session_state.int_to_str_var[0]}', f'predicted {st.session_state.int_to_str_var[1]}'], index=[f'actual {st.session_state.int_to_str_var[0]}', f'actual {st.session_state.int_to_str_var[1]}'])
        fig = px.imshow(confusion_s, text_auto=True, color_continuous_scale='Brwnyl', labels=dict(x="predictions", y="actual values"))
        fig.layout.coloraxis.showscale = False
        fig.update_layout(xaxis = dict(tickmode = 'array', tickvals = [0, 1], ticktext = [st.session_state.int_to_str_var[0], st.session_state.int_to_str_var[1]]), yaxis = dict(tickmode = 'array', tickvals = [0, 1], ticktext = [st.session_state.int_to_str_var[0], st.session_state.int_to_str_var[1]]))
        fig.update_traces(textfont_size=20)
        st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)
        conf1 = confusion_s.to_csv(index=False)
        st.download_button("Export confusion matrix", conf1, 'confusion_svm.csv', key='download7')
    
    with col2:
        st.subheader('KNN')
        confusion_k = pd.DataFrame(confusion_k, columns=[f'predicted {st.session_state.int_to_str_var[0]}', f'predicted {st.session_state.int_to_str_var[1]}'], index=[f'actual {st.session_state.int_to_str_var[0]}', f'actual {st.session_state.int_to_str_var[1]}'])
        fig = px.imshow(confusion_k, text_auto=True, color_continuous_scale='Brwnyl', labels=dict(x="predictions", y="actual values"))
        fig.layout.coloraxis.showscale = False
        fig.update_layout(xaxis = dict(tickmode = 'array', tickvals = [0, 1], ticktext = [st.session_state.int_to_str_var[0], st.session_state.int_to_str_var[1]]), yaxis = dict(tickmode = 'array', tickvals = [0, 1], ticktext = [st.session_state.int_to_str_var[0], st.session_state.int_to_str_var[1]]))
        fig.update_traces(textfont_size=20)
        st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)
        conf2 = confusion_k.to_csv(index=False)
        st.download_button("Export confusion matrix", conf2, 'confusion_knn.csv', key='download8')
    
    with col3:
        st.subheader('Logistic Regression')
        confusion_l = pd.DataFrame(confusion_l, columns=[f'predicted {st.session_state.int_to_str_var[0]}', f'predicted {st.session_state.int_to_str_var[1]}'], index=[f'actual {st.session_state.int_to_str_var[0]}', f'actual {st.session_state.int_to_str_var[1]}'])
        fig = px.imshow(confusion_l, text_auto=True, color_continuous_scale='Brwnyl', labels=dict(x="predictions", y="actual values"))
        fig.layout.coloraxis.showscale = False
        fig.update_layout(xaxis = dict(tickmode = 'array', tickvals = [0, 1], ticktext = [st.session_state.int_to_str_var[0], st.session_state.int_to_str_var[1]]), yaxis = dict(tickmode = 'array', tickvals = [0, 1], ticktext = [st.session_state.int_to_str_var[0], st.session_state.int_to_str_var[1]]))
        fig.update_traces(textfont_size=20)
        st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)
        conf3 = confusion_l.to_csv(index=False)
        st.download_button("Export confusion matrix", conf3, 'confusion_log.csv', key='download9')

    with col4:
        st.subheader('XGBoost')
        confusion_xg = pd.DataFrame(confusion_xg, columns=[f'predicted {st.session_state.int_to_str_var[0]}', f'predicted {st.session_state.int_to_str_var[1]}'], index=[f'actual {st.session_state.int_to_str_var[0]}', f'actual {st.session_state.int_to_str_var[1]}'])
        fig = px.imshow(confusion_xg, text_auto=True, color_continuous_scale='Brwnyl', labels=dict(x="predictions", y="actual values"))
        fig.layout.coloraxis.showscale = False
        fig.update_layout(xaxis = dict(tickmode = 'array', tickvals = [0, 1], ticktext = [st.session_state.int_to_str_var[0], st.session_state.int_to_str_var[1]]), yaxis = dict(tickmode = 'array', tickvals = [0, 1], ticktext = [st.session_state.int_to_str_var[0], st.session_state.int_to_str_var[1]]))
        fig.update_traces(textfont_size=20)
        st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)
        conf4 = confusion_xg.to_csv(index=False)
        st.download_button("Export confusion matrix", conf4, 'confusion_xgb.csv', key='download10')

@st.cache_data
def show_shap(shap_values, x_svm):
    st.subheader('Feature Importance (SHAP) usinng SVM predictions.')
    st.write(f'Remember that 0 represents: {st.session_state.int_to_str_var[0]} and 1 represents: {st.session_state.int_to_str_var[1]}. This means that a positive value for a feature means that it is more likely to be {st.session_state.int_to_str_var[1]} and a negative value means that it is more likely to be {st.session_state.int_to_str_var[0]}.')
    fig = plt.figure(figsize=(10, 4))
    shap.summary_plot(shap_values, x_svm)
    st.pyplot(fig)

def show_stats(mcb, mcb_unprocessed, y, features):
    st.subheader('Statistical tests')   
    st.caption(f'The following tests are performed to test if each variable is statistically significant between {st.session_state.int_to_str_var[0]} and {st.session_state.int_to_str_var[1]}. The tests are performed on both processed data (until CLR) and original data.')
    st.caption('Note that some new features were introduced (phyla ratios), there\'s no equivalent in the original data. Thus you can\'t see the results of the tests applied on these features in the second table which shows results on the original data (a "nan" will be showed instead).')

    grp_1 = mcb.loc[y['bin_var']==0]
    grp_2 = mcb.loc[y['bin_var']==1]
    
    tests = ttest_ind, mannwhitneyu, ranksums, kruskal
    tests_names = ['t-test', 'mannwhitneyu', 'ranksums', 'kruskal']
    pvals = pd.DataFrame(columns=tests_names, index=features)
    for i, test in enumerate(tests):
        for feature in features:
            pvals.loc[feature, tests_names[i]] = "{:.2e}".format(test(grp_1[feature], grp_2[feature])[1])
            
    #fig = px.imshow(pvals, text_auto=True, color_continuous_scale='Brwnyl', title='p-values of each test on processed data (only identified features)')
    #st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)

    show_pvals(pvals, 'p-values of each test on processed data (only identified features)')
    csv_data1 = pvals.to_csv(index=False)
    st.download_button("Export p-values", csv_data1, 'p-values_processed.csv', key='download3')
    
    grp_1 = mcb_unprocessed.loc[y['bin_var']==0]
    grp_2 = mcb_unprocessed.loc[y['bin_var']==1]
    
    tests = ttest_ind, mannwhitneyu, ranksums, kruskal
    tests_names = ['t-test', 'mannwhitneyu', 'ranksums', 'kruskal']
    pvals = pd.DataFrame(columns=tests_names, index=features)
    for i, test in enumerate(tests):
        for feature in features:
            if feature in mcb_unprocessed.columns:
                pvals.loc[feature, tests_names[i]] = "{:.2e}".format(test(grp_1[feature], grp_2[feature])[1])
            
    #fig = px.imshow(pvals, text_auto=True, color_continuous_scale='Brwnyl', title='p-values of each test on original data (only depicted features)')
    #st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)

    show_pvals(pvals, 'p-values of each test on original data (only identified features)')
    csv_data2 = pvals.to_csv(index=False)
    st.download_button("Export p-values", csv_data2, 'p-values_processed.csv', key='download4')
    st.write('Note that statistical tests may have requirements, such as normality, so they may not be fit for your data.')

@st.cache_data
def show_boxes(data_plot, features, raw_data, y_raw):
    
    st.subheader('More plots')
    st.write('Boxplots of each variable using both processed and original data seperated by the signature. This is to get a visual idea of the distribution of each identified variable.')
    data_plot['bin_var'] = data_plot['bin_var'].replace({0: st.session_state.int_to_str_var[0], 1: st.session_state.int_to_str_var[1]})
    raw_data['bin_var'] = y_raw.replace({0: st.session_state.int_to_str_var[0], 1: st.session_state.int_to_str_var[1]})

    titles = []
    for feature in features:
        if feature in raw_data.columns:
            titles.append(f'{feature} (input: processed data)')
            titles.append(f'{feature} (input: original data)')
        else:
            titles.append(f'{feature} (input: ratio of original data)')
            titles.append(f'no equivalent in original data')


    colors = px.colors.qualitative.Dark24
    if len(features) > len(colors):
        st.error('Too much features (24 maximum for boxplots)')
        st.stop()
    fig_f = make_subplots(rows=len(features), cols=2, subplot_titles=titles)
    fig_f.update_layout(height=2000, width=1000)

    for i, feature in enumerate(features):
        fig = go.Box(x = data_plot['bin_var'], y = data_plot[feature], marker=dict(size=4, opacity=0.5), boxpoints='all', jitter=0.3, pointpos=0, name=feature, marker_color=colors[i])
        fig_f.add_trace(fig, row=i+1, col=1)
        if feature in raw_data.columns:
            fig = go.Box(x = raw_data['bin_var'], y = raw_data[feature], marker=dict(size=4, opacity=0.5), marker_color=colors[i], boxpoints='all', jitter=0.3, pointpos=0, name=feature)
            fig_f.add_trace(fig, row=i+1, col=2)
        else:
            fig_f.add_trace(go.Box(), row=i+1, col=2)



    st.plotly_chart(fig_f, config=st.session_state.config)


