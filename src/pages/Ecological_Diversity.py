from st_pages import add_page_title

add_page_title()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.feature_selection import VarianceThreshold
from streamlit_plotly_events import plotly_events
from streamlit_tags import st_tags

from helpers import *

# Import Python implementations instead of R
from python_support import beta_dim_red, perform_ancom_alternative, shannon, simpson

pio.templates.default = "plotly"

if "proceed" in st.session_state.keys() and st.session_state.proceed:
        
        st.header('Alpha Diversity')

        st.write("**Choose which taxa levels to plot (default: Phyla)**")
        all_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

        levels = st.multiselect('Taxa levels to plot', all_levels[:st.session_state.last_level], default=['phylum'])

        if st.session_state.last_level >= st.session_state.level_to_int['kingdom']:
            mcb_k = st.session_state.mcb_k
            mcb_k = mcb_k.drop(columns=['bin_var'])
        if st.session_state.last_level >= st.session_state.level_to_int['phylum']:
            mcb_p = st.session_state.mcb_p
            mcb_p = mcb_p.drop(columns=['bin_var'])
        if st.session_state.last_level >= st.session_state.level_to_int['class']:
            mcb_c = st.session_state.mcb_c
            mcb_c = mcb_c.drop(columns=['bin_var'])
        if st.session_state.last_level >= st.session_state.level_to_int['order']:
            mcb_o = st.session_state.mcb_o
            mcb_o = mcb_o.drop(columns=['bin_var'])
        if st.session_state.last_level >= st.session_state.level_to_int['family']:
            mcb_f = st.session_state.mcb_f
            mcb_f = mcb_f.drop(columns=['bin_var'])
        if st.session_state.last_level >= st.session_state.level_to_int['genus']:
            mcb_g = st.session_state.mcb_g
            mcb_g = mcb_g.drop(columns=['bin_var'])
        if st.session_state.last_level >= st.session_state.level_to_int['species']:
            mcb_s = st.session_state.mcb_s
            mcb_s = mcb_s.drop(columns=['bin_var'])
        
        y = st.session_state.y
        
        alpha_measure = st.radio("Pick one alpha diversity measure", ('Shannon', 'Simpson'))
        alpha_data = pd.DataFrame(y.copy())
        
        # Use Python implementations
        alpha_func = simpson if alpha_measure == 'Simpson' else shannon

        if 'species' in levels:
            alpha_data['species'] = alpha_func(mcb_s)
        if 'genus' in levels:
            alpha_data['genus'] = alpha_func(mcb_g)
        if 'family' in levels:
            alpha_data['family'] = alpha_func(mcb_f)
        if 'order' in levels:
            alpha_data['order'] = alpha_func(mcb_o)
        if 'class' in levels: 
            alpha_data['class'] = alpha_func(mcb_c)
        if 'phylum' in levels:
            alpha_data['phylum'] = alpha_func(mcb_p)
        if 'kingdom' in levels:
            alpha_data['kingdom'] = alpha_func(mcb_k)
        
        fig = px.box(alpha_data, y=levels, color='bin_var', title=f'Alpha Diversity using {alpha_measure}')
        fig.update_traces(marker=dict(size=4, opacity=0.5), boxpoints='all', jitter=0.3, pointpos=0)
        fig.for_each_trace(lambda t: t.update(name = st.session_state.int_to_str_var[int(t.name)]))
        st.plotly_chart(fig, config=st.session_state.config)

        st.subheader('Statistical tests of alpha diversity')
        stat_tests = ttest_ind, mannwhitneyu, ranksums, kruskal
        stat_tests_names = ['t-test', 'Mann-Whitney U', 'Wilcoxon rank-sum', 'Kruskal-Wallis']
        pvals = pd.DataFrame(columns=stat_tests_names, index=levels)
        for i, test in enumerate(stat_tests):
            for level in levels:
                pvals.loc[level, stat_tests_names[i]] = "{:.2e}".format(test(list(alpha_data[alpha_data['bin_var']==0][level]), list(alpha_data[alpha_data['bin_var']==1][level]))[1])
            
            
        # fig = px.imshow(pvals, text_auto=True, color_continuous_scale='Brwnyl', title='p-values of each test on alpha diversity data (chosen taxa levels shown)')
        # st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)
 
        show_pvals(pvals, 'p-values of each test on alpha diversity measures')
        csv_data = pvals.to_csv(index=False)
        st.download_button("Export p-values", csv_data, 'p-values.csv', key='download2')


        
        
        st.divider() 
        st.header('Beta Diversity') 

        st.write("**Choose which taxa levels to consider for the analysis (dimensionality reduction) (default: all levels)**")
        levels = st.multiselect('Taxa levels to consider', all_levels[:st.session_state.last_level], default=all_levels[:st.session_state.last_level])

        st.write("**Second, choose which beta diversity measure and with dimensionality reduction technic to use.**")
        if st.session_state.otu_type == 'Read counts':
            beta_measure = st.radio("Pick one beta diversity measure", ('bray-curtis', 'gower', 'chao'))
        else:
            beta_measure = st.radio("Pick one beta diversity measure", ('bray-curtis', 'gower'))
        dim_red = st.radio("Pick one dimensionality reduction technique", ('PCoA', 'NMDS'))
        
        mcb_chosen = pd.DataFrame(index=mcb_f.index) 
        if 'species' in levels:
            mcb_chosen = mcb_chosen.join(mcb_s, how='inner', lsuffix='_1', rsuffix='_2') 
        if 'genus' in levels:
            mcb_chosen = mcb_chosen.join(mcb_g, how='inner', lsuffix='_3', rsuffix='_4')
        if 'family' in levels:
            mcb_chosen = mcb_chosen.join(mcb_f, how='inner', lsuffix='_5', rsuffix='_5')
        if 'class' in levels:
            mcb_chosen = mcb_chosen.join(mcb_c, how='inner', lsuffix='_6', rsuffix='_7')
        if 'phylum' in levels:
            mcb_chosen = mcb_chosen.join(mcb_p, how='inner', lsuffix='_8', rsuffix='_10')
        if 'kingdom' in levels:
            mcb_chosen = mcb_chosen.join(mcb_k, how='inner', lsuffix='_11', rsuffix='_12')
        if 'order' in levels:
            mcb_chosen = mcb_chosen.join(mcb_o, how='inner', lsuffix='_13', rsuffix='_14') 
        
        # Use Python implementation
        beta_measure_arg = beta_measure
        if beta_measure == 'bray-curtis':
            beta_measure_arg = 'bray'
        try:
            result = beta_dim_red(mcb_chosen, beta_measure_arg, dim_red)
            if dim_red == 'NMDS':
                bt = result['a']
                stress = result['b']
                st.success(f'NMDS stress value: {stress}')
            else:
                bt = result
            bt = pd.DataFrame(bt, columns=['PC1', 'PC2'])
            bt['bin_var'] = np.array(y).astype(str)
            fig = px.scatter(bt, x='PC1', y='PC2', color='bin_var', title = f'Beta Diversity using {beta_measure} and {dim_red}')
            fig.for_each_trace(lambda t: t.update(name = st.session_state.int_to_str_var[int(t.name)]))
            st.plotly_chart(fig, config=st.session_state.config)
        except Exception as e:
            st.error('Most probably, the error comes from your dataset having multiple samples with all 0 values in read counts. Consider removing these samples.') 
            raise e

        if st.session_state.otu_type == 'Read counts':
            st.divider()
            st.header('ANCOM-BC')
            st.write("This section is about showing results related to ANCOM BC calulcations.")
            st.warning('**Warning:** If the data is not formatted properly, you may get unusual results.')

            st.write("**The 3 dataframes input into ANCOM:** ")
            st.write(st.session_state.ancom_df)
            st.write(st.session_state.ancom_y)
            st.write(st.session_state.tax_tab)


            st.write("**Choose which taxa level to use for ANCOM:** ")
            level_input = st.radio('Taxa level to consider', all_levels[1:st.session_state.last_level])

            @st.cache_data
            def perform_ancom(dff, yy, tax_tabb, level):
                # Use Python implementation of differential abundance analysis
                results = perform_ancom_alternative(dff, yy, tax_tabb, level, method='mannwhitneyu')
                return results

            results = perform_ancom(st.session_state.ancom_df, st.session_state.ancom_y, st.session_state.tax_tab, level_input)
            labels = sorted(list(st.session_state.ancom_y['bin_var'].unique()))
            
            st.write('**ANCOM results:** ')
            st.write(f'Comparison of {labels[1]} vs {labels[0]} groups')
            st.write(results)
            #st.write(set(results.iloc[:, 0]).difference(set(list(st.session_state.tax_tab[level_input].unique()))))
            csv_res = results.to_csv(index=False)
            st.download_button("Export ANCOM results", csv_res, 'ancombc.csv', key='download15')

            results['pos_lfc'] = (results.iloc[:, 2] > 0)

            pos_results = results[results['pos_lfc'] == True]
            neg_results = results[results['pos_lfc'] == False]

            pos_results = pos_results.sort_values(by=pos_results.columns[2], ascending=False)
            neg_results = neg_results.sort_values(by=neg_results.columns[2], ascending=False)

            if len(results) < 20:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Positive Log Fold Change',
                    x=pos_results.iloc[:, 0], y=pos_results.iloc[:, 2],
                    marker_color = 'green',
                    error_y=dict(type='data', array=list(pos_results.iloc[:, 4]))
                ))
                fig.add_trace(go.Bar(
                    name='Negative Log Fold Change',
                    x=neg_results.iloc[:, 0], y=neg_results.iloc[:, 2],
                    marker_color = 'red',
                    error_y=dict(type='data', array=list(neg_results.iloc[:, 4]))
                ))
                fig.update_layout(title=f'ANCOM results for {level_input} level')
                st.plotly_chart(fig, config=st.session_state.config)
            
            else:
                st.write('**Too many micro-organisms to show everything, we\'ll show the top 10 positive and negative changes changes.**')

                len_pos = min(10, len(pos_results))
                len_neg = min(10, len(neg_results))

                pos_results = pos_results.iloc[:len_pos, :]
                neg_results = neg_results.iloc[:len_neg, :]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Positive Log Fold Change',
                    x=pos_results.iloc[:, 0], y=pos_results.iloc[:, 2],
                    marker_color = 'green',
                    error_y=dict(type='data', array=list(pos_results.iloc[:, 4]))
                ))
                fig.add_trace(go.Bar(
                    name='Negative Log Fold Change',
                    x=neg_results.iloc[:, 0], y=neg_results.iloc[:, 2],
                    marker_color = 'red',
                    error_y=dict(type='data', array=list(neg_results.iloc[:, 4]))
                ))
                fig.update_layout(title=f'ANCOM results for {level_input} level')
                st.plotly_chart(fig, config=st.session_state.config)


else:
    st.error('Please upload files first in the Upload data tab.')
#try: 
#except:    
#st.error('Please upload files first in the Upload data tab.')