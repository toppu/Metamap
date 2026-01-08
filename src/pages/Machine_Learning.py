import pandas as pd
import plotly.io as pio
import streamlit as st
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from st_pages import add_page_title

from src.utils.helpers import (
    filter_by_correlation,
    get_results,
    process_level_abundance,
    process_level_rel_abundance,
    show_boxes,
    show_confusions,
    show_shap,
    show_stats,
)

add_page_title()
pio.templates.default = "plotly"

try:
    y = st.session_state.y
    st.divider()
    st.header('Data processing')
    mcbs = []

    if 'proceed' in st.session_state.keys() and st.session_state.proceed and st.session_state.otu_type == 'Read counts':
        st.write('**For the AI part, data is processed following the pipeline described in the publication. The first steps are to remove features with high number of 0 values and low variance.**')
        st.write('Choose below the minimum proportion of values that need to be non-zero for a feature to be kept.')
        st.caption('For example, if you have 10 samples and the threshold is 10%, a feature will be kept if it has at least 1 non-zero value.')
        if 'keepFeature_percentage' not in st.session_state.keys():
            keepFeature_percentage = st.number_input('Insert a threshold (between 1% and 100%)', min_value=0.0, max_value=100.0, value=15.0)
            st.session_state.keepFeature_percentage = keepFeature_percentage
        else:
            st.session_state.keepFeature_percentage = st.number_input('Insert a threshold (between 1% and 100%, default: 15)', min_value=0.0, max_value=100.0, value=st.session_state.keepFeature_percentage)
        st.success("" + str(st.session_state.keepFeature_percentage) + "%")

        st.write('Choose below the minimum variance cutoff for a feature to be kept.')
        st.caption('For example, if the threshold is 10, a feature will be kept if it has a variance of at least 10 across all samples.')
        if 'min_var' not in st.session_state.keys():
            min_var = st.number_input('Insert a variance cutoff', min_value=0.0, max_value=10000.0, value=10.0)
            st.session_state.min_var = min_var
        else:
            st.session_state.min_var = st.number_input('Insert a variance cutoff', min_value=0.0, max_value=10000.0, value=st.session_state.min_var)
        st.success(st.session_state.min_var)

        features_f = features_g = features_p = features_c = features_o = features_k = features_s = []
        if st.session_state.last_level >= st.session_state.level_to_int['family']:
            mcb_f, features_f = process_level_abundance(st.session_state.mcb_f, 'family', st.session_state.keepFeature_percentage, st.session_state.min_var)
            mcbs.append(mcb_f)
        if st.session_state.last_level >= st.session_state.level_to_int['genus']:
            mcb_g, features_g = process_level_abundance(st.session_state.mcb_g, 'genus', st.session_state.keepFeature_percentage, st.session_state.min_var)
            mcbs.append(mcb_g)
        if st.session_state.last_level >= st.session_state.level_to_int['phylum']:
            mcb_p, features_p = process_level_abundance(st.session_state.mcb_p, 'phylum', st.session_state.keepFeature_percentage, st.session_state.min_var)
            mcbs.append(mcb_p)
        if st.session_state.last_level >= st.session_state.level_to_int['class']:
            mcb_c, features_c = process_level_abundance(st.session_state.mcb_c, 'class', st.session_state.keepFeature_percentage, st.session_state.min_var)
            mcbs.append(mcb_c)
        if st.session_state.last_level >= st.session_state.level_to_int['order']:
            mcb_o, features_o = process_level_abundance(st.session_state.mcb_o, 'order', st.session_state.keepFeature_percentage, st.session_state.min_var)
            mcbs.append(mcb_o)
        if st.session_state.last_level >= st.session_state.level_to_int['kingdom']:
            mcb_k, features_k = process_level_abundance(st.session_state.mcb_k, 'kingdom', st.session_state.keepFeature_percentage, st.session_state.min_var)
            mcbs.append(mcb_k)
        if st.session_state.last_level >= st.session_state.level_to_int['species']:
            mcb_s, features_s = process_level_abundance(st.session_state.mcb_s, 'species', st.session_state.keepFeature_percentage, st.session_state.min_var)
            mcbs.append(mcb_s)
        if st.session_state.last_level >= st.session_state.level_to_int['phylum']:
            mcb_p_r = process_level_abundance(st.session_state.mcb_p, 'phyla ratios', st.session_state.keepFeature_percentage, st.session_state.min_var)
            mcbs.append(mcb_p_r)
        

    elif 'proceed' in st.session_state.keys() and st.session_state.proceed and st.session_state.otu_type == 'Relative abundance':

        st.write('**For the AI part, data is processed following the pipeline described in the publication. The first step is to remove features with high number of 0 values and low variance features.**')

        st.write('Choose below the minimum proportion of values that need to be non-zero for a feature to be kept.')
        st.caption('For example, if you have 10 samples and the threshold is 10%, a feature will be kept if it has at least 1 non-zero value.')
        if 'keepFeature_percentage' not in st.session_state.keys():
            keepFeature_percentage = st.number_input('Insert a threshold (between 1% and 100%)', min_value=0.0, max_value=100.0, value=15.0)
            st.session_state.keepFeature_percentage = keepFeature_percentage
        else:
            st.session_state.keepFeature_percentage = st.number_input('Insert a threshold (between 1% and 100%)', min_value=0.0, max_value=100.0, value=st.session_state.keepFeature_percentage)
        
        st.write('Do you want to remove features with variance near zero ?')
        check_lowvar = st.checkbox('Remove features with variance near zero')
        if check_lowvar:
            st.write('Choose below the frequency cutoff that will be used for the nearZeroVar function (check the R function from caret package).')
            st.caption('It captures the cutoff for the ratio of the most common value to the second most common value')
            if 'cutoff' not in st.session_state.keys() or st.session_state.cutoff == -1:
                cutoff = st.number_input('Insert the threshold', min_value=0.0, max_value=1000.0, value=10.0)
                st.session_state.cutoff = cutoff
            else:
                st.session_state.cutoff = st.number_input('Insert the threshold', min_value=0.0, max_value=1000.0, value=st.session_state.cutoff)
        else:
            st.session_state.cutoff = -1

        if st.session_state.last_level >= st.session_state.level_to_int['family']:
            mcb_f, features_f = process_level_rel_abundance(st.session_state.mcb_f, 'family', st.session_state.keepFeature_percentage, st.session_state.cutoff)
            mcbs.append(mcb_f)
        if st.session_state.last_level >= st.session_state.level_to_int['genus']:
            mcb_g, features_g = process_level_rel_abundance(st.session_state.mcb_g, 'genus', st.session_state.keepFeature_percentage, st.session_state.cutoff)
            mcbs.append(mcb_g)
        if st.session_state.last_level >= st.session_state.level_to_int['phylum']:
            mcb_p, features_p = process_level_rel_abundance(st.session_state.mcb_p, 'phylum', st.session_state.keepFeature_percentage, st.session_state.cutoff)
            mcbs.append(mcb_p)
        if st.session_state.last_level >= st.session_state.level_to_int['class']:
            mcb_c, features_c = process_level_rel_abundance(st.session_state.mcb_c, 'class', st.session_state.keepFeature_percentage, st.session_state.cutoff)
            mcbs.append(mcb_c)
        if st.session_state.last_level >= st.session_state.level_to_int['order']:
            mcb_o, features_o = process_level_rel_abundance(st.session_state.mcb_o, 'order', st.session_state.keepFeature_percentage, st.session_state.cutoff)
            mcbs.append(mcb_o)
        if st.session_state.last_level >= st.session_state.level_to_int['kingdom']:
            mcb_k, features_k = process_level_rel_abundance(st.session_state.mcb_k, 'kingdom', st.session_state.keepFeature_percentage, st.session_state.cutoff)
            mcbs.append(mcb_k)
        if st.session_state.last_level >= st.session_state.level_to_int['phylum']:
            mcb_p_r = process_level_rel_abundance(st.session_state.mcb_p, 'phyla ratios', st.session_state.keepFeature_percentage, st.session_state.cutoff)
            mcbs.append(mcb_p_r)
        if st.session_state.last_level >= st.session_state.level_to_int['species']:
            mcb_s, features_s = process_level_rel_abundance(st.session_state.mcb_s, 'species', st.session_state.keepFeature_percentage, st.session_state.cutoff)
            mcbs.append(mcb_s)
            

        st.write('**Remaining feature:**')
        st.write(pd.concat(mcbs, axis=1))

        csv_mcb2 = pd.concat(mcbs, axis=1).to_csv(index=False)
        st.download_button("Export data", csv_mcb2, 'remaining_features.csv', key='download20')

        # raw data
    
    y_raw = None
    mcb_f_raw = mcb_g_raw = mcb_p_raw = mcb_c_raw = mcb_o_raw = mcb_k_raw = mcb_s_raw = None
    if st.session_state.last_level >= st.session_state.level_to_int['family']:
        mcb_f_raw = st.session_state.mcb_f[features_f]
    if st.session_state.last_level >= st.session_state.level_to_int['genus']:
        mcb_g_raw = st.session_state.mcb_g[features_g]
    if st.session_state.last_level >= st.session_state.level_to_int['phylum']:
        mcb_p_raw = st.session_state.mcb_p[features_p]
    if st.session_state.last_level >= st.session_state.level_to_int['class']:
        mcb_c_raw = st.session_state.mcb_c[features_c]
    if st.session_state.last_level >= st.session_state.level_to_int['order']:
        mcb_o_raw = st.session_state.mcb_o[features_o]
    if st.session_state.last_level >= st.session_state.level_to_int['kingdom']:
        mcb_k_raw = st.session_state.mcb_k
        y_raw = mcb_k_raw['bin_var']
        mcb_k_raw = mcb_k_raw[features_k]
    if st.session_state.last_level >= st.session_state.level_to_int['species']:
        mcb_s_raw = st.session_state.mcb_s[features_s]
    
            
except Exception as e:
    st.error('If files not uploaded, lease upload files first in the Upload data tab.')
    raise e

st.header('Feature selection')
st.write('Select the taxonomic levels to perform the AI predictions on (default: phyla and phyla ratios, you can add and remove as you wish)')
all_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

temp = all_levels[:st.session_state.last_level]
if st.session_state.last_level >= st.session_state.level_to_int['phylum']:
    temp.append('phyla ratios') 
levels = st.multiselect('Taxa level', temp, default=['phylum', 'phyla ratios'])

st.subheader('On each taxonomic level')
st.write('Your dataset may have some unwanted data, such as "unclassified", "unknown", "ambiguous" microorganism in some taxa level. You can remove them from the analyzed data by adding these microorganisms to the lists below (each list corresponds to a taxonomic level).')

# choice of features to remove (ambiguous)
level_to_df = {}
level_to_df_unprocessed = {}

if 'kingdom' in levels:
    st.write('**Kingdom**')
    remove_k = st.multiselect('Remove from kingdom', mcb_k.columns, []) 
    mcb_k = mcb_k.drop(columns=remove_k, axis=1) 
    mcb_k_raw = mcb_k_raw.drop(columns=remove_k, axis=1)
    level_to_df['kingdom'] = mcb_k
    level_to_df_unprocessed['kingdom'] = mcb_k_raw

if 'phylum' in levels:
    st.write('**Phylum**')
    remove_p = st.multiselect('Remove from phylum', mcb_p.columns, [])
    mcb_p = mcb_p.drop(columns=remove_p, axis=1)
    mcb_p_raw = mcb_p_raw.drop(columns=remove_p, axis=1)
    level_to_df['phylum'] = mcb_p
    level_to_df_unprocessed['phylum'] = mcb_p_raw

if 'phyla ratios' in levels:
    st.write('**phyla ratios**')
    remove_p_r = st.multiselect('Remove from phyla ratios', mcb_p_r.columns, [])
    mcb_p_r = mcb_p_r.drop(columns=remove_p_r, axis=1)
    level_to_df['phyla ratios'] = mcb_p_r

if 'class' in levels:
    st.write('**Class**')
    remove_c = st.multiselect('Remove from class', mcb_c.columns, [])
    mcb_c = mcb_c.drop(columns=remove_c, axis=1)
    mcb_c_raw = mcb_c_raw.drop(columns=remove_c, axis=1)
    level_to_df['class'] = mcb_c
    level_to_df_unprocessed['class'] = mcb_c_raw

if 'order' in levels:
    st.write('**Order**')
    remove_o = st.multiselect('Remove from order', mcb_o.columns, [])
    mcb_o = mcb_o.drop(columns=remove_o, axis=1)
    mcb_o_raw = mcb_o_raw.drop(columns=remove_o, axis=1)
    level_to_df['order'] = mcb_o
    level_to_df_unprocessed['order'] = mcb_o_raw

if 'family' in levels:
    st.write('**Family**')
    remove_f = st.multiselect('Remove from family', mcb_f.columns, [])
    mcb_f = mcb_f.drop(columns=remove_f, axis=1)
    mcb_f_raw = mcb_f_raw.drop(columns=remove_f, axis=1)
    level_to_df['family'] = mcb_f
    level_to_df_unprocessed['family'] = mcb_f_raw

if 'genus' in levels:
    st.write('**Genus**')
    remove_g = st.multiselect('Remove from genus', mcb_g.columns, [])
    mcb_g = mcb_g.drop(columns=remove_g, axis=1)
    mcb_g_raw = mcb_g_raw.drop(columns=remove_g, axis=1)
    level_to_df['genus'] = mcb_g
    level_to_df_unprocessed['genus'] = mcb_g_raw

if 'species' in levels:
    st.write('**Species**')
    remove_s = st.multiselect('Remove from species', mcb_s.columns, [])
    mcb_s = mcb_s.drop(columns=remove_s, axis=1)
    mcb_s_raw = mcb_s_raw.drop(columns=remove_s, axis=1)
    level_to_df['species'] = mcb_s
    level_to_df_unprocessed['species'] = mcb_s_raw

data = pd.DataFrame(index=y.index)
data_unprocessed = pd.DataFrame(index=y.index)

for level in levels:
    data = pd.concat([data, level_to_df[level]], axis=1)
    if level != 'phyla ratios':
        data_unprocessed = pd.concat([data_unprocessed, level_to_df_unprocessed[level]], axis=1)

if data.columns.duplicated().sum() > 0: 
    st.error('Error concatenating columns, check that column names are unique between different taxa levels. \n If there\'s for example "Ambiguous Taxa" or "NA" in many levels, it raises this error. You can remove one or all features with this duplicated name in the previous section.')
    st.error(f'Duplicated columns: {data.columns[data.columns.duplicated()]}')
    st.stop()
if data_unprocessed.columns.duplicated().sum() > 0:
    st.write(data_unprocessed.columns)
    st.error('Error concatenating columns, check that column names are unique between different taxa levels. \n If there\'s for example "Ambiguous Taxa" or "NA" in many levels, it raises this error. You can remove one or all features with this duplicated name in the previous section.')
    st.error(f' Duplicated columns: {data_unprocessed.columns[data_unprocessed.columns.duplicated()]}')
    st.stop()

st.divider()
st.subheader('Correlation')
st.write('You can further process the data by removing features that are correlated to each other. This is done by removing one of the two features that have a correlation coefficient higher than the threshold you set.')
is_filter = st.checkbox('Filter correlated features')
filter_threshold = st.number_input('Insert a threshold (between 0 and 1)', min_value=0.0, max_value=1.0, value=0.95)

data_before_correlation = data.copy()

if is_filter:
    data = filter_by_correlation(data, filter_threshold)
st.write('This is the final dataset that will be used for the decision tree and AI predictions.')
st.write(data)
csv_data3 =  data.to_csv(index=False)
st.download_button("Export data", csv_data3, 'remaining_features.csv', key='download5')

st.divider()
st.subheader('Decision tree')
st.write('You can visualize the decision tree that will be used for the AI predictions.')
st.write('The decision tree is built using the following parameters:')

depth = st.number_input('Insert the depth you want to use', min_value=1, max_value=20, value=3)
min_samples_leaf = st.number_input('Insert the minimum number of samples required to be at a leaf node', min_value=1, max_value=20, value=1)

@st.cache_data
def run_tree(tree_max_depth, tree_min_samples_leaf, tree_data):
    classifier = DecisionTreeClassifier(max_depth=tree_max_depth, min_samples_leaf=tree_min_samples_leaf)
    fitted_clf = classifier.fit(tree_data, y)
    return fitted_clf 

if st.button('Run tree again'):
    run_tree.clear()

clf = run_tree(depth, min_samples_leaf, data)
graph = tree.export_graphviz(clf, class_names=[str(st.session_state.int_to_str_var[0]), str(st.session_state.int_to_str_var[1])], feature_names=data.columns, filled=True, rounded=True, special_characters=True)
st.caption('In some cases, it happens that the fontsize is small. You can have a big image by clicking on the arrows on the top right of the image.')
st.graphviz_chart(graph)



features = []
thresholds = []
for ind, threshold in zip(clf.tree_.feature, clf.tree_.threshold):
    if ind != -2:
        features.append(data.columns[ind])
        thresholds.append(threshold)

st.write('**Features that will be used for AI predictions:**')
st.success(features)

st.write('**Thresholds used for binary transformation (last step before AI algorithms):**')
st.success(thresholds)

st.divider()
st.subheader('AI predictions')
st.write('You can now run the AI predictions on your dataset. The AI algorithms used are: SVM, Logistic Regression, KNN, XGboost.')
st.write('You can choose the parameters for each algorithm, and the number of cross-validation folds to use for the evaluation of the algorithms.')
# folds
folds = st.number_input('Insert the number of folds you want to use for cross-validation', min_value=2, max_value=10, value=4)
col1, col2, col3, col4 = st.columns(4, gap='large')
with col1:
    st.write('**SVM**')
    # kernel
    kernel = st.selectbox('Kernel', ['rbf', 'linear', 'poly', 'sigmoid'], index=0)
with col2:
    st.write('**Logistic Regression**')
    # max interations
    max_iter = st.number_input('Max iterations', min_value=100, max_value=10000, value=1000)
with col3:
    st.write('**KNN**')
    # n_neighbors
    n_neighbors = st.number_input('Number of neighbors', min_value=1, max_value=20, value=5)
with col4:
    st.write('**XGboost**')
    # n_estimators
    n_estimators = st.number_input('Number of estimators', min_value=1, max_value=200, value=50)
    # max depth
    max_depth = st.number_input('Max depth', min_value=1, max_value=20, value=3)


st.write('Runninng the AI algorithms may take some time depending on how many folds you choose and how many features you have. If you want to change the parameters, deactivate the button below, otherwise it will start running at each change.')
is_run = st.radio('Choose whether to run the AI predictions', ['Don\'t run AI predictions for now', 'Run AI predictions']) == 'Run AI predictions'
if is_run:
    evals, box_plot, shap_values, confusion_s, confusion_l, confusion_k, confusion_xg, features, x_svm = get_results(data, pd.DataFrame(y), features, thresholds, folds, kernel, max_iter, n_neighbors, n_estimators, max_depth)
    show_confusions(confusion_s, confusion_l, confusion_k, confusion_xg)
    st.subheader('Evaluation Metrics')
    st.write(evals)
    csv_evals = evals.to_csv(index=False)
    st.download_button("Export metrics", csv_evals, 'metrics.csv', key='download11')
    show_shap(shap_values, x_svm)
    show_stats(data, data_unprocessed, pd.DataFrame(y), features)
    show_boxes(box_plot, features, data_unprocessed, y_raw)

    st.subheader('Summary')

    st.markdown('**Number of samples:** ' + str(data.shape[0]))
    st.markdown('**Cutoff used to remove features with a high number of zeroes:** ' + str(st.session_state.keepFeature_percentage))
    if st.session_state.otu_type == 'Read counts':
        mk_var = f'Minimum variance cutoff: {str(st.session_state.min_var)}'
        st.markdown('**Minimum variance cutoff:** ' + str(st.session_state.min_var)) 
    else:
        if st.session_state.cutoff > 0:

            mk_var = f'Cutoff used for NearZeroVar: {str(st.session_state.cutoff)}'
            st.markdown('**Cutoff used for NearZeroVar:** ' + str(st.session_state.cutoff)) 
        else:
            mk_var = ''

    st.markdown('**Taxa levels considered for the analysis:** ' + str(levels))
    st.markdown('**Number of features before correlation filter:** ' + str(data_before_correlation.shape[1]))  

    if is_filter:
        st.markdown('**Cutoff used for correlation:** ' + str(filter_threshold))
        st.markdown('**Number of features remaining after removing correlated features:** ' + str(len(data.columns)))
        mk4 = f'Cutoff used for correlation: {str(filter_threshold)}'
        mk5 = f'Number of features remaining after removing correlated features (if has been selected): {str(len(data.columns))}'
    else:
        mk4 = ''
        mk5 = ''

    st.markdown('**Depth used for decision tree:** ' + str(depth))
    st.markdown('**Minimum number of samples required to be at a leaf node:** ' + str(min_samples_leaf))
    st.markdown('**Features identified and used for AI predictions:** ' + str(features))
    st.markdown('**Thresholds used for binary transformation (last step before AI algorithms):** ' + str(thresholds))
    st.markdown('**Number of folds used for cross-validation:** ' + str(folds))
    st.markdown('**Kernel used for SVM:** ' + str(kernel))
    st.markdown('**Max iterations used for Logistic Regression:** ' + str(max_iter))
    st.markdown('**Number of neighbors used for KNN:** ' + str(n_neighbors))
    st.markdown('**Number of estimators used for XGboost:** ' + str(n_estimators))
    st.markdown('**Max depth used for XGboost:** ' + str(max_depth))

    export_text = 'Number of samples: ' + str(data.shape[0]) + '\n' + \
        'Cutoff used to remove features with a high number of zeroes: ' + str(st.session_state.keepFeature_percentage) + '\n' + \
        mk_var + '\n' + \
        'Taxa levels considered for the analysis: ' + str(levels) + '\n' + \
        'Number of features before correlation filter: ' + str(data_before_correlation.shape[1]) + '\n' + \
        mk4 + '\n' + \
        mk5 + '\n' + \
        'Depth used for decision tree: ' + str(depth) + '\n' + \
        'Minimum number of samples required to be at a leaf node: ' + str(min_samples_leaf) + '\n' + \
        'Features identified and used for AI predictions: ' + str(features) + '\n' + \
        'Thresholds used for binary transformation (last step before AI algorithms): ' + str(thresholds) + '\n' + \
        'Number of folds used for cross-validation: ' + str(folds) + '\n' + \
        'Kernel used for SVM: ' + str(kernel) + '\n' + \
        'Max iterations used for Logistic Regression: ' + str(max_iter) + '\n' + \
        'Number of neighbors used for KNN: ' + str(n_neighbors) + '\n' + \
        'Number of estimators used for XGboost: ' + str(n_estimators) + '\n' + \
        'Max depth used for XGboost: ' + str(max_depth) + '\n'

        
    st.download_button("Export summary", export_text, 'summary.txt', key='download12')

