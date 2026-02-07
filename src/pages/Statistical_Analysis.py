import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import umap
from plotly.subplots import make_subplots
from scipy.stats import kruskal, mannwhitneyu, ranksums, ttest_ind
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from st_pages import add_page_title
from streamlit_plotly_events import plotly_events

from src.utils.helpers import show_pvals
from src.utils.r_service import pandas2ri

st.set_page_config(layout="wide")
add_page_title()

pandas2ri.activate()
pio.templates.default = "plotly"

# Check if required session state exists
if 'proceed' not in st.session_state or not st.session_state.proceed:
    st.warning("⚠️ Please upload data first from the 'Upload data' page.")
    st.stop()

try:
    if st.session_state.proceed and st.session_state.otu_type == 'Read counts':
        if st.session_state.last_level >= st.session_state.level_to_int['kingdom']:
            mcb_k = st.session_state.mcb_k
        if st.session_state.last_level >= st.session_state.level_to_int['phylum']:
            mcb_p = st.session_state.mcb_p
        if st.session_state.last_level >= st.session_state.level_to_int['class']:
            mcb_c = st.session_state.mcb_c
        if st.session_state.last_level >= st.session_state.level_to_int['order']:
            mcb_o = st.session_state.mcb_o
        if st.session_state.last_level >= st.session_state.level_to_int['family']:
            mcb_f = st.session_state.mcb_f
        if st.session_state.last_level >= st.session_state.level_to_int['genus']:
            mcb_g = st.session_state.mcb_g
        if st.session_state.last_level >= st.session_state.level_to_int['species']:
            mcb_s = st.session_state.mcb_s

        y = st.session_state.y

        all_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
        
        #st.subheader('Raw data')
        st.write('The plot below shows taxonomcical diversity for each group.')
        st.write('Note that your dataset may have some unwanted data, such as "unclassified" or "unknown" microorganism in some taxa level. You can remove them from the analyzed data by adding these microorganisms to the lists below (each list corresponds to a taxonomic level).')
        col5, col6, col7 = st.columns(3)
        col8 = col9 = col10 = col11 = None
        if st.session_state.last_level >= 4:
            if st.session_state.last_level == 7:
                col8, col9, col10, col11 = st.columns(4)
            else:
                col8, col9, col10 = st.columns(3)

        funnels = []
        all_mcbs = []

        if st.session_state.last_level >= st.session_state.level_to_int['kingdom']:
            with col5:
                st.write('Kingdom')
                k_cols = mcb_k.columns.drop('bin_var')
                remove_k = st.multiselect('Remove from kingdom', k_cols, [])
                mcb_k = mcb_k.drop(columns=remove_k, axis=1)
            all_mcbs.append(mcb_k)
            mcb_k_funnel = ((mcb_k>0).groupby('bin_var').sum()>0).sum(axis=1).reset_index().astype(int).set_index('bin_var').rename(columns={0:'kingdom'})
            funnels.append(mcb_k_funnel)

        if st.session_state.last_level >= st.session_state.level_to_int['phylum']:
            with col6:
                st.write('Phylum')
                p_cols = mcb_p.columns.drop('bin_var')
                remove_p = st.multiselect('Remove from phylum', p_cols, [])
                mcb_p = mcb_p.drop(columns=remove_p, axis=1)
            all_mcbs.append(mcb_p)
            mcb_p_funnel = ((mcb_p>0).groupby('bin_var').sum()>0).sum(axis=1).reset_index().astype(int).set_index('bin_var').rename(columns={0:'phylum'})
            funnels.append(mcb_p_funnel)

        if st.session_state.last_level >= st.session_state.level_to_int['class']:
            
            with col7:
                st.write('Class')
                c_cols = mcb_c.columns.drop('bin_var')
                remove_c = st.multiselect('Remove from class', c_cols, [])
                mcb_c = mcb_c.drop(columns=remove_c, axis=1)
            all_mcbs.append(mcb_c)
            mcb_c_funnel = ((mcb_c>0).groupby('bin_var').sum()>0).sum(axis=1).reset_index().astype(int).set_index('bin_var').rename(columns={0:'class'})
            funnels.append(mcb_c_funnel)

        if st.session_state.last_level >= st.session_state.level_to_int['order']:
            with col8:
                st.write('Order')
                o_cols = mcb_o.columns.drop('bin_var')
                remove_o = st.multiselect('Remove from order', o_cols, [])
                mcb_o = mcb_o.drop(columns=remove_o, axis=1)
            all_mcbs.append(mcb_o)
            mcb_o_funnel = ((mcb_o>0).groupby('bin_var').sum()>0).sum(axis=1).reset_index().astype(int).set_index('bin_var').rename(columns={0:'order'})
            funnels.append(mcb_o_funnel)

        if st.session_state.last_level >= st.session_state.level_to_int['family']:
            with col9:
                st.write('Family')
                f_cols = mcb_f.columns.drop('bin_var')
                remove_f = st.multiselect('Remove from family', f_cols, [])
                mcb_f = mcb_f.drop(columns=remove_f, axis=1)
            all_mcbs.append(mcb_f)
            mcb_f_funnel = ((mcb_f>0).groupby('bin_var').sum()>0).sum(axis=1).reset_index().astype(int).set_index('bin_var').rename(columns={0:'family'})
            funnels.append(mcb_f_funnel)

        if st.session_state.last_level >= st.session_state.level_to_int['genus']:
            with col10:
                st.write('Genus')
                g_cols = mcb_g.columns.drop('bin_var')
                remove_g = st.multiselect('Remove from genus', g_cols, [])
                mcb_g = mcb_g.drop(columns=remove_g, axis=1)
            all_mcbs.append(mcb_g)
            mcb_g_funnel = ((mcb_g>0).groupby('bin_var').sum()>0).sum(axis=1).reset_index().astype(int).set_index('bin_var').rename(columns={0:'genus'})
            funnels.append(mcb_g_funnel)
        
        if st.session_state.last_level >= st.session_state.level_to_int['species']:
            with col11:
                st.write('Species')
                s_cols = mcb_s.columns.drop('bin_var')
                remove_s = st.multiselect('Remove from species', s_cols, [])
                mcb_s = mcb_s.drop(columns=remove_s, axis=1)
            all_mcbs.append(mcb_s)
            mcb_s_funnel = ((mcb_s>0).groupby('bin_var').sum()>0).sum(axis=1).reset_index().astype(int).set_index('bin_var').rename(columns={0:'species'})
            funnels.append(mcb_s_funnel)
            
        funnel_df = pd.concat(funnels, axis=1).reset_index()
        funnel_df = pd.melt(funnel_df, id_vars='bin_var', value_vars=all_levels[:st.session_state.last_level], var_name='taxa level', value_name='number')

        fig = px.funnel(funnel_df, x='number', y='taxa level', color='bin_var', title=f'Number of taxa present in each group: {st.session_state.int_to_str_var[0]} vs {st.session_state.int_to_str_var[1]}')
        fig.for_each_trace(lambda t: t.update(name = st.session_state.int_to_str_var[int(t.name)]))
        selected = plotly_events(fig, click_event=True)
        st.write('**To see which are the most abundant microorganisms and most frequent microorganisms by taxa, click on a taxonomical level in the plot above.**')
        st.caption('Frequency is the number of samples in which that microorganism is present, while abundance is the sum of the counts of this microorganism across all samples.')

        if selected:

            selected_category = selected[0]['y']
            elem0 = None
            elem1 = None

            if selected_category == 'kingdom':
                elem0 = mcb_k[mcb_k['bin_var']==0]
                elem1 = mcb_k[mcb_k['bin_var']==1]
            elif selected_category == 'phylum':
                elem0 = mcb_p[mcb_p['bin_var']==0]
                elem1 = mcb_p[mcb_p['bin_var']==1]
            elif selected_category == 'class':
                elem0 = mcb_c[mcb_c['bin_var']==0]
                elem1 = mcb_c[mcb_c['bin_var']==1]
            elif selected_category == 'order':
                elem0 = mcb_o[mcb_o['bin_var']==0]
                elem1 = mcb_o[mcb_o['bin_var']==1]
            elif selected_category == 'family':
                elem0 = mcb_f[mcb_f['bin_var']==0]
                elem1 = mcb_f[mcb_f['bin_var']==1]
            elif selected_category == 'genus':
                elem0 = mcb_g[mcb_g['bin_var']==0]
                elem1 = mcb_g[mcb_g['bin_var']==1]
            elif selected_category == 'species':
                elem0 = mcb_s[mcb_s['bin_var']==0]
                elem1 = mcb_s[mcb_s['bin_var']==1]


        
            st.write(f'--> Most abundant microorganisms at the {selected_category} level:')
            col3, col4 = st.columns(2, gap='small')
            abundance0 = elem0.drop(columns=['bin_var']).sum(axis=0).sort_values(ascending=False).reset_index().rename(columns={'index':'microorganism', 0:'Sum of read counts'})[:10]
            abundance1 = elem1.drop(columns=['bin_var']).sum(axis=0).sort_values(ascending=False).reset_index().rename(columns={'index':'microorganism', 0:'Sum of read counts'})[:10]
            with col3:
                fig = px.bar(abundance0, x='microorganism', y='Sum of read counts', title=f'Most abundant microorganisms at the {selected_category} level for {st.session_state.int_to_str_var[0]}')
                fig.update_layout(autosize=False, width=500, height=400)
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)

            with col4:
                # bar plot red
                fig = px.bar(abundance1, x='microorganism', y='Sum of read counts', title=f'Most abundant microorganisms at the {selected_category} level for {st.session_state.int_to_str_var[1]}')
                fig.update_layout(autosize=False, width=500, height=400)
                fig.update_traces(marker_color='red')
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)

            st.write(f'--> Most frequent microorganisms at the {selected_category} level:')

            col1, col2 = st.columns(2, gap='small')
            freq0 = elem0.drop(columns=['bin_var']).astype(bool).sum(axis=0).sort_values(ascending=False).reset_index().rename(columns={'index':'microorganism', 0:'frequency'})[:10]
            freq1 = elem1.drop(columns=['bin_var']).astype(bool).sum(axis=0).sort_values(ascending=False).reset_index().rename(columns={'index':'microorganism', 0:'frequency'})[:10]


            with col1:
                fig = px.bar(freq0, x='microorganism', y='frequency', title=f'Most frequent microorganisms at the {selected_category} level for {st.session_state.int_to_str_var[0]}')
                fig.update_layout(autosize=False, width=500, height=400)
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)
            
            with col2:
                fig = px.bar(freq1, x='microorganism', y='frequency', title=f'Most frequent microorganisms at the {selected_category} level for {st.session_state.int_to_str_var[1]}')
                fig.update_layout(autosize=False, width=500, height=400)
                fig.update_traces(marker_color='red')
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)


            # Dealing with less features, with relative abundance
            st.write(f'**Below, relative abundance plot of microorganisms at the selected taxonomical level: {selected_category}.**')
            st.write('Choose a threshold (%) for the minimum relative abundance of a microorganism to be considered. (e.g if 1%, each relative abundance which is more than 1% is considered. Then, we take top 20 microorganisms which satisfy the condition the most. The rest are put in "others"). ')
            st.write('**Note that it will sum up to 100% even if you removed some features.**')
            threshold = st.number_input('Insert a threshold (between 0.1% and 100%)', min_value=0.0001, max_value=100.0, value=0.5)
            st.success(f'You chose {threshold}% as the threshold.')

            # microorganisms with relative abundance > threshold
            combined = pd.concat([elem0, elem1], axis = 0)
            combined = combined.drop(columns=['bin_var'])
            sums = combined.sum(axis=1)
            rel_ab = combined.divide(sums, axis=0)*100      
            combined = rel_ab>threshold
            combined = combined.sum(axis=0).sort_values(ascending=False).reset_index().rename(columns={'index':'microorganism', 0:'frequency'})
            
            if len(combined) < 20:
                kept_microorganisms = list(combined[:len(combined)-1].microorganism)
                other_microorganisms = list(combined[len(combined)-1:].microorganism)
            else:
                kept_microorganisms = list(combined[:20].microorganism)
                other_microorganisms = list(combined[20:].microorganism)
            

            elem0_kept = elem0[kept_microorganisms].copy() 
            elem0_others = elem0[other_microorganisms]
            elem0_kept['others'] = elem0_others.sum(axis=1)

            elem1_kept = elem1[kept_microorganisms].copy()
            elem1_others = elem1[other_microorganisms]
            elem1_kept['others'] = elem1_others.sum(axis=1)

            elem0_kept_sums = elem0_kept.sum(axis=1)
            elem0_kept = elem0_kept.divide(elem0_kept_sums, axis=0)*100


            elem1_kept_sums = elem1_kept.sum(axis=1)
            elem1_kept = elem1_kept.divide(elem1_kept_sums, axis=0)*100

            choice_bool = False
            if len(elem0_kept) > 20 or len(elem1_kept) > 20:
                st.warning('WARNING: Too many samples to plot everything. Please select samples of interest if needed.')
                choice_bool = True

            col1, col2 = st.columns(2, gap='small')

            defaut_samples = min(len(elem0_kept), len(elem1_kept), 20)
                
            with col1:
                if choice_bool:
                    selected = st.multiselect(f'Samples of interest {st.session_state.int_to_str_var[0]}', elem0_kept.index.tolist())
                    if selected:
                        elem0_kept = elem0_kept.loc[selected]
                    else: 
                        elem0_kept = elem0_kept.iloc[:defaut_samples]
                
                fig = px.bar(elem0_kept, x=elem0_kept.index, y=elem0_kept.columns, title=f'Relative abundance of microorganisms for {st.session_state.int_to_str_var[0]}', color_discrete_sequence=px.colors.qualitative.Light24)
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)
            
            with col2:
                if choice_bool:
                    selected = st.multiselect(f'Samples of interest {st.session_state.int_to_str_var[1]}', elem1_kept.index.tolist())
                    if selected:
                        elem1_kept = elem1_kept.loc[selected]
                    else: 
                        elem1_kept = elem1_kept.iloc[:defaut_samples]
                fig = px.bar(elem1_kept, x=elem1_kept.index, y=elem1_kept.columns, title=f'Relative abundance of microorganisms for {st.session_state.int_to_str_var[1]}', color_discrete_sequence=px.colors.qualitative.Light24)       
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)

            #################### Stats tests ####################

            st.subheader('Statistical tests')
            st.write('In this subsection, you can perform statistical tests to see if there is a significant difference between the two groups (Tests are performed on original data, unlike the AI tab).')
            st.write('Choose the microorganisms you want to perform the test on. You can choose multiple microorganisms. The test will be performed on each of them.')
            st.warning('If an error shows, you probably must unselect some microorganisms.')
            microorganisms = st.multiselect('Microorganisms', elem0.columns.tolist(), default=elem0.columns.tolist()[:2])
            stat_tests = ttest_ind, mannwhitneyu, ranksums, kruskal
            stat_tests_names = ['t-test', 'Mann-Whitney U', 'Wilcoxon rank-sum', 'Kruskal-Wallis']
            try:
                pvals = pd.DataFrame(columns=stat_tests_names, index=microorganisms)
                for i, test in enumerate(stat_tests):
                    for microorganism in microorganisms:
                        try:
                            result = test(elem0[microorganism], elem1[microorganism])
                            pvals.loc[microorganism, stat_tests_names[i]] = "{:.2e}".format(result[1])
                        except ValueError as ve:
                            # Handle cases where test can't be performed (e.g., identical values)
                            if 'identical' in str(ve).lower():
                                pvals.loc[microorganism, stat_tests_names[i]] = "N/A"
                            else:
                                raise
                        
                #fig = px.imshow(pvals, text_auto=True, color_continuous_scale='Brwnyl', title='p-values of each test on original data (chosen features shown)')
                #st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)
                show_pvals(pvals, 'p-values of each test on original data (chosen features shown')
                csv_data = pvals.to_csv(index=False)
                st.download_button("Export p-values", csv_data, 'p-values.csv', key='download1')
            except Exception as e:
                st.error('There was a problem performing the statistical tests. Please try removing some microorganisms or check your data.')

            #################### DIM REDUCTION ####################   
            st.divider() 
            st.subheader('Dimensionality reduction and visual separation of the two groups')
            st.write('In this subsection, you can use dimensionality reduction techniques to visualize the separation of the two groups. You can choose between PCA, t-SNE and UMAP. You can also choose to use 2D or 3D for the visualization.')
            st.write('You can also also choose wether you want to perform the dimensionality reduction on the chosen taxa level or on all microorganisms (all taxa levels).')

            @st.cache_data
            def assemble_all_levels(mcb_levels): 
                assembled_data = pd.DataFrame(index=y.index)
                for level_data in mcb_levels:
                    to_add = level_data.drop(columns=['bin_var'])
                    assembled_data = pd.concat([assembled_data, to_add], axis=1, ignore_index=True)
                return assembled_data

            dim_red = st.radio('Dimensionality reduction technique', ['PCA', 't-SNE', 'UMAP'])
            dim_red_dim = st.radio('Dimensionality reduction dimension', ['2D', '3D'])
            level_or_global = st.radio('Level or global', (f'Selected taxa level: {selected_category}', 'All taxa levels'))
            
            data_unprocessed = assemble_all_levels(all_mcbs)

            if level_or_global == f'Selected taxa level: {selected_category}':
                data_unprocessed = pd.concat([elem0, elem1], axis=0)

            dim_red_func = None
            if dim_red == 'PCA':
                dim_red_func = PCA
            elif dim_red == 't-SNE':
                dim_red_func = TSNE
            elif dim_red == 'UMAP':
                dim_red_func = umap.UMAP
            
            y_names = y.replace({0: str(st.session_state.int_to_str_var[0]), 1: str(st.session_state.int_to_str_var[1])})
            y_colors = y.replace({0: 'red', 1: 'blue'})
            if dim_red_dim == '2D':
                dim_red_func = dim_red_func(n_components=2)
                projections = dim_red_func.fit_transform(data_unprocessed)
                fig = px.scatter(x=projections[:, 0], y=projections[:, 1], color=y_names, title=f'{dim_red_dim} projection of the data using {dim_red}', color_discrete_sequence=px.colors.qualitative.Dark24)
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)
            
            elif dim_red_dim == '3D':
                dim_red_func = dim_red_func(n_components=3)
                projections = dim_red_func.fit_transform(data_unprocessed)

                fig = px.scatter_3d(projections, x=projections[:, 0], y=projections[:, 1], z=projections[:, 2], color=y_names, title=f'{dim_red_dim} projection of the data using {dim_red}', color_discrete_sequence=px.colors.qualitative.Dark24)
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)

                st.write('For the 3D part, you can also plot the 3 compenents 2 by 2 to see the separation of the two groups in 2D (using the 3D components).')

                # Plot the 3 subplots in 1 plot
                fig = make_subplots(rows=1, cols=3, subplot_titles=['1st and 2nd components', '1st and 3rd components', '2nd and 3rd components'])
                fig.add_trace(go.Scatter(x=projections[:, 0], y=projections[:, 1], mode='markers', marker=dict(color=y_colors)), row=1, col=1)
                fig.add_trace(go.Scatter(x=projections[:, 0], y=projections[:, 2], mode='markers', marker=dict(color=y_colors)), row=1, col=2)
                fig.add_trace(go.Scatter(x=projections[:, 1], y=projections[:, 2], mode='markers', marker=dict(color=y_colors)), row=1, col=3)
                fig.update_layout(title='2D projections of the 3D components', showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)



            

             



###############################

    elif st.session_state.proceed and st.session_state.otu_type == 'Relative abundance':
        
        if st.session_state.last_level >= st.session_state.level_to_int['kingdom']:
            mcb_k = st.session_state.mcb_k
        if st.session_state.last_level >= st.session_state.level_to_int['phylum']:
            mcb_p = st.session_state.mcb_p
        if st.session_state.last_level >= st.session_state.level_to_int['class']:
            mcb_c = st.session_state.mcb_c
        if st.session_state.last_level >= st.session_state.level_to_int['order']:
            mcb_o = st.session_state.mcb_o
        if st.session_state.last_level >= st.session_state.level_to_int['family']:
            mcb_f = st.session_state.mcb_f
        if st.session_state.last_level >= st.session_state.level_to_int['genus']:
            mcb_g = st.session_state.mcb_g
        if st.session_state.last_level >= st.session_state.level_to_int['species']:
            mcb_s = st.session_state.mcb_s

        y = st.session_state.y

        all_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

        
        #st.subheader('Raw data')
        st.write('The plot below shows taxonomcical diversity for each group.')
        st.write('Note that your dataset may have some unwanted data, such as "unclassified" or "unknown" microorganism in some taxa level. You can remove them from the analyzed data by adding these microorganisms to the lists below (each list corresponds to a taxonomic level).')
        
        col5, col6, col7 = st.columns(3)
        if st.session_state.last_level >= 4:
            if st.session_state.last_level == 7:
                col8, col9, col10, col11 = st.columns(4)
            else:
                col8, col9, col10 = st.columns(3)

        funnels = []
        all_mcbs = []

        if st.session_state.last_level >= st.session_state.level_to_int['kingdom']:
            with col5:
                st.write('Kingdom')
                k_cols = mcb_k.columns.drop('bin_var')
                remove_k = st.multiselect('Remove from kingdom', k_cols, [])
                mcb_k = mcb_k.drop(columns=remove_k, axis=1)
            all_mcbs.append(mcb_k)
            mcb_k_funnel = ((mcb_k>0).groupby('bin_var').sum()>0).sum(axis=1).reset_index().astype(int).set_index('bin_var').rename(columns={0:'kingdom'})
            funnels.append(mcb_k_funnel)

        if st.session_state.last_level >= st.session_state.level_to_int['phylum']:
            with col6:
                st.write('Phylum')
                p_cols = mcb_p.columns.drop('bin_var')
                remove_p = st.multiselect('Remove from phylum', p_cols, [])
                mcb_p = mcb_p.drop(columns=remove_p, axis=1)
            all_mcbs.append(mcb_p)
            mcb_p_funnel = ((mcb_p>0).groupby('bin_var').sum()>0).sum(axis=1).reset_index().astype(int).set_index('bin_var').rename(columns={0:'phylum'})
            funnels.append(mcb_p_funnel)

        if st.session_state.last_level >= st.session_state.level_to_int['class']:
            
            with col7:
                st.write('Class')
                c_cols = mcb_c.columns.drop('bin_var')
                remove_c = st.multiselect('Remove from class', c_cols, [])
                mcb_c = mcb_c.drop(columns=remove_c, axis=1)
            all_mcbs.append(mcb_c)
            mcb_c_funnel = ((mcb_c>0).groupby('bin_var').sum()>0).sum(axis=1).reset_index().astype(int).set_index('bin_var').rename(columns={0:'class'})
            funnels.append(mcb_c_funnel)

        if st.session_state.last_level >= st.session_state.level_to_int['order']:
            with col8:
                st.write('Order')
                o_cols = mcb_o.columns.drop('bin_var')
                remove_o = st.multiselect('Remove from order', o_cols, [])
                mcb_o = mcb_o.drop(columns=remove_o, axis=1)
            all_mcbs.append(mcb_o)
            mcb_o_funnel = ((mcb_o>0).groupby('bin_var').sum()>0).sum(axis=1).reset_index().astype(int).set_index('bin_var').rename(columns={0:'order'})
            funnels.append(mcb_o_funnel)

        if st.session_state.last_level >= st.session_state.level_to_int['family']:
            with col9:
                st.write('Family')
                f_cols = mcb_f.columns.drop('bin_var')
                remove_f = st.multiselect('Remove from family', f_cols, [])
                mcb_f = mcb_f.drop(columns=remove_f, axis=1)
            all_mcbs.append(mcb_f)
            mcb_f_funnel = ((mcb_f>0).groupby('bin_var').sum()>0).sum(axis=1).reset_index().astype(int).set_index('bin_var').rename(columns={0:'family'})
            funnels.append(mcb_f_funnel)

        if st.session_state.last_level >= st.session_state.level_to_int['genus']:
            with col10:
                st.write('Genus')
                g_cols = mcb_g.columns.drop('bin_var')
                remove_g = st.multiselect('Remove from genus', g_cols, [])
                mcb_g = mcb_g.drop(columns=remove_g, axis=1)
            all_mcbs.append(mcb_g)
            mcb_g_funnel = ((mcb_g>0).groupby('bin_var').sum()>0).sum(axis=1).reset_index().astype(int).set_index('bin_var').rename(columns={0:'genus'})
            funnels.append(mcb_g_funnel)
        
        if st.session_state.last_level >= st.session_state.level_to_int['species']:
            with col11:
                st.write('Species')
                s_cols = mcb_s.columns.drop('bin_var')
                remove_s = st.multiselect('Remove from species', s_cols, [])
                mcb_s = mcb_s.drop(columns=remove_s, axis=1)
            all_mcbs.append(mcb_s)
            mcb_s_funnel = ((mcb_s>0).groupby('bin_var').sum()>0).sum(axis=1).reset_index().astype(int).set_index('bin_var').rename(columns={0:'species'})
            funnels.append(mcb_s_funnel)
            
        funnel_df = pd.concat(funnels, axis=1).reset_index()
        funnel_df = pd.melt(funnel_df, id_vars='bin_var', value_vars=all_levels[:st.session_state.last_level], var_name='taxa level', value_name='number')


        fig = px.funnel(funnel_df, x='number', y='taxa level', color='bin_var', title=f'Number of taxa present in each group: {st.session_state.int_to_str_var[0]} vs {st.session_state.int_to_str_var[1]}')
        fig.for_each_trace(lambda t: t.update(name = st.session_state.int_to_str_var[int(t.name)]))
        selected = plotly_events(fig, click_event=True)
        st.write('**To see which are the most abundant microorganisms and most frequent microorganisms by taxa, click on a taxonomical level in the plot above.**')
        st.caption('Frequency is the number of samples in which that microorganism is present, while abundance is the sum of the counts of this microorganism across all samples.')
        if selected:
            selected_category = selected[0]['y']
            elem0 = None
            elem1 = None

            if selected_category == 'kingdom':
                elem0 = mcb_k[mcb_k['bin_var']==0]
                elem1 = mcb_k[mcb_k['bin_var']==1]
            elif selected_category == 'phylum':
                elem0 = mcb_p[mcb_p['bin_var']==0]
                elem1 = mcb_p[mcb_p['bin_var']==1]
            elif selected_category == 'class':
                elem0 = mcb_c[mcb_c['bin_var']==0]
                elem1 = mcb_c[mcb_c['bin_var']==1]
            elif selected_category == 'order':
                elem0 = mcb_o[mcb_o['bin_var']==0]
                elem1 = mcb_o[mcb_o['bin_var']==1]
            elif selected_category == 'family':
                elem0 = mcb_f[mcb_f['bin_var']==0]
                elem1 = mcb_f[mcb_f['bin_var']==1]
            elif selected_category == 'genus':
                elem0 = mcb_g[mcb_g['bin_var']==0]
                elem1 = mcb_g[mcb_g['bin_var']==1]
            elif selected_category == 'species':
                elem0 = mcb_s[mcb_s['bin_var']==0]
                elem1 = mcb_s[mcb_s['bin_var']==1]


            st.write(f'--> Most frequent microorganisms at the {selected_category} level:')

            col1, col2 = st.columns(2, gap='small')
            freq0 = elem0.drop(columns=['bin_var']).astype(bool).sum(axis=0).sort_values(ascending=False).reset_index().rename(columns={'index':'microorganism', 0:'frequency'})[:10]
            freq1 = elem1.drop(columns=['bin_var']).astype(bool).sum(axis=0).sort_values(ascending=False).reset_index().rename(columns={'index':'microorganism', 0:'frequency'})[:10]

            with col1:
                fig = px.bar(freq0, x='microorganism', y='frequency', title=f'Most frequent microorganisms at the {selected_category} level for {st.session_state.int_to_str_var[0]}')
                fig.update_layout(autosize=False, width=500, height=400)
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)
            
            with col2:
                fig = px.bar(freq1, x='microorganism', y='frequency', title=f'Most frequent microorganisms at the {selected_category} level for {st.session_state.int_to_str_var[1]}')
                fig.update_layout(autosize=False, width=500, height=400)
                fig.update_traces(marker_color='red')
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)

            # Dealing with less features, with relative abundance
            st.write(f'**Below, relative abundance plot of microorganisms at the selected taxonomical level: {selected_category}.**')
            st.write('Choose a threshold (%) for the minimum relative abundance of a microorganism to be considered. (e.g if 1%, each relative abundance which is more than 1% is considered. Then, we take top 20 microorganisms which satisfy the condition (in terms of counts). The rest of microorganisms are put in "others"). ')
            threshold = st.number_input('Insert a threshold (between 0.0001% and 100%)', min_value=0.0001, max_value=100.0, value=0.5)
            st.success(f'You chose {threshold}% as the threshold.')

            combined = pd.concat([elem0, elem1], axis = 0)
            combined = combined.drop(columns=['bin_var'])
            combined = combined>threshold
            combined = combined.sum(axis=0).sort_values(ascending=False).reset_index().rename(columns={'index':'microorganism', 0:'frequency'})
            
            if len(combined) < 20:
                kept_microorganisms = list(combined[:len(combined)-1].microorganism)
                other_microorganisms = list(combined[len(combined)-1:].microorganism)
            else:
                kept_microorganisms = list(combined[:20].microorganism)
                other_microorganisms = list(combined[20:].microorganism)
            

            elem0_kept = elem0[kept_microorganisms] 
            elem0_others = elem0[other_microorganisms]
            elem0_kept['others'] = elem0_others.sum(axis=1)

            elem1_kept = elem1[kept_microorganisms]
            elem1_others = elem1[other_microorganisms]
            elem1_kept['others'] = elem1_others.sum(axis=1)

            choice_bool = False
            if len(elem0_kept) > 20 or len(elem1_kept) > 20:
                st.warning('WARNING: Too many samples to plot everything. Please select samples of interest if needed.')
                choice_bool = True

            col1, col2 = st.columns(2, gap='small')

            defaut_samples = min(len(elem0_kept), len(elem1_kept), 20)
                
            with col1:
                if choice_bool:
                    selected = st.multiselect(f'Samples of interest {st.session_state.int_to_str_var[0]}', elem0_kept.index.tolist())
                    if selected:
                        elem0_kept = elem0_kept.loc[selected]
                    else: 
                        elem0_kept = elem0_kept.iloc[:defaut_samples]
                fig = px.bar(elem0_kept, x=elem0_kept.index, y=elem0_kept.columns, title=f'Relative abundance of microorganisms for {st.session_state.int_to_str_var[0]}', color_discrete_sequence=px.colors.qualitative.Light24)
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)
            
            with col2:
                if choice_bool:
                    selected = st.multiselect(f'Samples of interest {st.session_state.int_to_str_var[1]}', elem1_kept.index.tolist())
                    if selected:
                        elem1_kept = elem1_kept.loc[selected]
                    else: 
                        elem1_kept = elem1_kept.iloc[:defaut_samples]
                fig = px.bar(elem1_kept, x=elem1_kept.index, y=elem1_kept.columns, title=f'Relative abundance of microorganisms for {st.session_state.int_to_str_var[1]}', color_discrete_sequence=px.colors.qualitative.Light24)       
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)

            #################### Stats tests ####################

            st.subheader('Statistical tests')
            st.write('In this subsection, you can perform statistical tests to see if there is a significant difference between the two groups (Tests are performed on original data, unlike the AI tab).')
            st.write('Choose the microorganisms you want to perform the test on. You can choose multiple microorganisms. The test will be performed on each of them.')
            st.warning('If an error shows, you probably must unselect some microorganisms.')
            microorganisms = st.multiselect('Microorganisms', elem0.columns.tolist(), default=elem0.columns.tolist()[:2])
            stat_tests = ttest_ind, mannwhitneyu, ranksums, kruskal
            stat_tests_names = ['t-test', 'Mann-Whitney U', 'Wilcoxon rank-sum', 'Kruskal-Wallis']
            try:
                pvals = pd.DataFrame(columns=stat_tests_names, index=microorganisms)
                for i, test in enumerate(stat_tests):
                    for microorganism in microorganisms:
                        pvals.loc[microorganism, stat_tests_names[i]] = "{:.2e}".format(test(elem0[microorganism], elem1[microorganism])[1])
                        
                #fig = px.imshow(pvals, text_auto=True, color_continuous_scale='Brwnyl', title='p-values of each test on original data (chosen features shown)')
                #st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)

                show_pvals(pvals, 'p-values of each test on original data (chosen features shown')
            except Exception as e:
                st.error('There must be a problem with one of the microorganism on which one of the tests can\'t be applied. Please try to remove the problematic microorgism.')
                raise e

            csv_data = pvals.to_csv(index=False)
            st.download_button("Export p-values", csv_data, 'p-values.csv', key='download1')

        
            #################### DIM REDUCTION ####################   
            st.divider() 
            st.subheader('Dimensionality reduction and visual separation of the two groups')
            st.write('In this subsection, you can use dimensionality reduction techniques to visualize the separation of the two groups. You can choose between PCA, t-SNE and UMAP. You can also choose to use 2D or 3D for the visualization.')
            st.write('You can also also choose wether you want to perform the dimensionality reduction on the chosen taxa level or on all microorganisms (all taxa levels).')

            @st.cache_data
            def assemble_all_levels(levels): 
                
                data_unprocessed = pd.DataFrame(index=y.index)
                for level in levels:
                    to_add = level.drop(columns=['bin_var'])
                    data_unprocessed = pd.concat([data_unprocessed, to_add], axis=1, ignore_index=True)
                return data_unprocessed

            dim_red = st.radio('Dimensionality reduction technique', ['PCA', 't-SNE', 'UMAP'])
            dim_red_dim = st.radio('Dimensionality reduction dimension', ['2D', '3D'])
            level_or_global = st.radio('Level or global', (f'Selected taxa level: {selected_category}', 'All taxa levels'))

            data_unprocessed = assemble_all_levels(all_mcbs)

            if level_or_global == f'Selected taxa level: {selected_category}':
                data_unprocessed = pd.concat([elem0, elem1], axis=0) 

            if dim_red == 'PCA':
                dim_red_func = PCA
            elif dim_red == 't-SNE':
                dim_red_func = TSNE
            elif dim_red == 'UMAP':
                dim_red_func = umap.UMAP
            
            y_names = y.replace({0: str(st.session_state.int_to_str_var[0]), 1: str(st.session_state.int_to_str_var[1])})
            y_colors = y.replace({0: 'red', 1: 'blue'})
            if dim_red_dim == '2D':
                dim_red_func = dim_red_func(n_components=2)
                projections = dim_red_func.fit_transform(data_unprocessed)
                fig = px.scatter(x=projections[:, 0], y=projections[:, 1], color=y_names, title=f'{dim_red_dim} projection of the data using {dim_red}', color_discrete_sequence=px.colors.qualitative.Dark24)
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)
            
            elif dim_red_dim == '3D':
                dim_red_func = dim_red_func(n_components=3)
                projections = dim_red_func.fit_transform(data_unprocessed)

                fig = px.scatter_3d(projections, x=projections[:, 0], y=projections[:, 1], z=projections[:, 2], color=y_names, title=f'{dim_red_dim} projection of the data using {dim_red}', color_discrete_sequence=px.colors.qualitative.Dark24)
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)

                st.write('For the 3D part, you can also plot the 3 compenents 2 by 2 to see the separation of the two groups in 2D (using the 3D components).')
 
                # Plot the 3 subplots in 1 plot
                fig = make_subplots(rows=1, cols=3, subplot_titles=['1st and 2nd components', '1st and 3rd components', '2nd and 3rd components'])
                fig.add_trace(go.Scatter(x=projections[:, 0], y=projections[:, 1], mode='markers', marker=dict(color=y_colors)), row=1, col=1)
                fig.add_trace(go.Scatter(x=projections[:, 0], y=projections[:, 2], mode='markers', marker=dict(color=y_colors)), row=1, col=2)
                fig.add_trace(go.Scatter(x=projections[:, 1], y=projections[:, 2], mode='markers', marker=dict(color=y_colors)), row=1, col=3)
                fig.update_layout(title='2D projections of the 3D components', showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)

    else:
        st.error('Upload data to start the analysis.')

except Exception as e:
    raise e
    





