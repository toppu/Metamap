import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy.stats import kruskal, mannwhitneyu, ranksums, ttest_ind
from st_pages import add_page_title

from src.utils.helpers import show_pvals
from src.utils.r_service import get_r_connection_pool, pandas2ri
from src.utils.r_service import globalenv as robjects

add_page_title()

pandas2ri.activate()
pio.templates.default = "plotly"

# Check if required session state exists
if 'proceed' not in st.session_state or not st.session_state.proceed:
    st.warning("⚠️ Please upload data first from the 'Upload data' page.")
    st.stop()

if "proceed" in st.session_state.keys() and st.session_state.proceed:
    
    # Load data once for all tabs
    all_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    
    # Prepare data from session state
    if st.session_state.last_level >= st.session_state.level_to_int['kingdom']:
        mcb_k = st.session_state.mcb_k.drop(columns=['bin_var'])
    if st.session_state.last_level >= st.session_state.level_to_int['phylum']:
        mcb_p = st.session_state.mcb_p.drop(columns=['bin_var'])
    if st.session_state.last_level >= st.session_state.level_to_int['class']:
        mcb_c = st.session_state.mcb_c.drop(columns=['bin_var'])
    if st.session_state.last_level >= st.session_state.level_to_int['order']:
        mcb_o = st.session_state.mcb_o.drop(columns=['bin_var'])
    if st.session_state.last_level >= st.session_state.level_to_int['family']:
        mcb_f = st.session_state.mcb_f.drop(columns=['bin_var'])
    if st.session_state.last_level >= st.session_state.level_to_int['genus']:
        mcb_g = st.session_state.mcb_g.drop(columns=['bin_var'])
    if st.session_state.last_level >= st.session_state.level_to_int['species']:
        mcb_s = st.session_state.mcb_s.drop(columns=['bin_var'])
    
    y = st.session_state.y
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["🔢 Alpha Diversity", "🗺️ Beta Diversity", "📊 Differential Abundance (ANCOM-BC)"])
    
    # =========================================================================
    # TAB 1: ALPHA DIVERSITY
    # =========================================================================
    with tab1:
        st.write("**Choose which taxa levels to plot (default: Phyla)**")

        levels = st.multiselect('Taxa levels to plot', all_levels[:st.session_state.last_level], default=['phylum'], key='alpha_levels')
        
        alpha_measure = st.radio("Pick one alpha diversity measure", ('Shannon', 'Simpson'))
        alpha_data = pd.DataFrame(y.copy())
        
        # Get the alpha diversity function name
        if alpha_measure == 'Simpson':
            alpha_func = 'simpson'
        elif alpha_measure == 'Shannon':
            alpha_func = 'shannon'
        
        # Calculate alpha diversity for each selected level
        # Helper function to convert R result to numpy array
        def r_to_numeric(r_result):
            if isinstance(r_result, dict):
                # If it's a dict (named vector), extract values
                return np.array(list(r_result.values()), dtype=float)
            elif hasattr(r_result, '__iter__') and not isinstance(r_result, str):
                # If it's array-like, convert directly
                return np.array(list(r_result), dtype=float)
            else:
                # Single value
                return np.array([r_result], dtype=float)
        
        if 'species' in levels:
            result = robjects[alpha_func](mcb_s)
            alpha_data['species'] = r_to_numeric(result)
        if 'genus' in levels:
            result = robjects[alpha_func](mcb_g)
            alpha_data['genus'] = r_to_numeric(result)
        if 'family' in levels:
            result = robjects[alpha_func](mcb_f)
            alpha_data['family'] = r_to_numeric(result)
        if 'order' in levels:
            result = robjects[alpha_func](mcb_o)
            alpha_data['order'] = r_to_numeric(result)
        if 'class' in levels:
            result = robjects[alpha_func](mcb_c)
            alpha_data['class'] = r_to_numeric(result)
        if 'phylum' in levels:
            result = robjects[alpha_func](mcb_p)
            alpha_data['phylum'] = r_to_numeric(result)
        if 'kingdom' in levels:
            result = robjects[alpha_func](mcb_k)
            alpha_data['kingdom'] = r_to_numeric(result)
        
        fig = px.box(alpha_data, y=levels, color='bin_var', title=f'Alpha Diversity using {alpha_measure}')
        fig.update_traces(marker=dict(size=4, opacity=0.5), boxpoints='all', jitter=0.3, pointpos=0)
        # Update trace names if int_to_str_var is available
        if 'int_to_str_var' in st.session_state:
            fig.for_each_trace(lambda t: t.update(name = st.session_state.int_to_str_var[int(t.name)]))
        st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)

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

    # =========================================================================
    # TAB 2: BETA DIVERSITY
    # =========================================================================
    with tab2:
        st.write("**Choose which taxa levels to consider for the analysis (dimensionality reduction) (default: all levels)**")
        levels_beta = st.multiselect('Taxa levels to consider', all_levels[:st.session_state.last_level], default=all_levels[:st.session_state.last_level], key='beta_levels')

        st.write("**Second, choose which beta diversity measure and with dimensionality reduction technic to use.**")
        if st.session_state.otu_type == 'Read counts':
            beta_measure = st.radio("Pick one beta diversity measure", ('bray-curtis', 'gower', 'chao'))
        else:
            beta_measure = st.radio("Pick one beta diversity measure", ('bray-curtis', 'gower'))
        dim_red = st.radio("Pick one dimensionality reduction technique", ('PCoA', 'NMDS'))
        
        mcb_chosen = pd.DataFrame(index=mcb_f.index) 
        if 'species' in levels_beta:
            mcb_chosen = mcb_chosen.join(mcb_s, how='inner', lsuffix='_1', rsuffix='_2') 
        if 'genus' in levels_beta:
            mcb_chosen = mcb_chosen.join(mcb_g, how='inner', lsuffix='_3', rsuffix='_4')
        if 'family' in levels_beta:
            mcb_chosen = mcb_chosen.join(mcb_f, how='inner', lsuffix='_5', rsuffix='_5')
        if 'class' in levels_beta:
            mcb_chosen = mcb_chosen.join(mcb_c, how='inner', lsuffix='_6', rsuffix='_7')
        if 'phylum' in levels_beta:
            mcb_chosen = mcb_chosen.join(mcb_p, how='inner', lsuffix='_8', rsuffix='_10')
        if 'kingdom' in levels_beta:
            mcb_chosen = mcb_chosen.join(mcb_k, how='inner', lsuffix='_11', rsuffix='_12')
        if 'order' in levels_beta:
            mcb_chosen = mcb_chosen.join(mcb_o, how='inner', lsuffix='_13', rsuffix='_14')
        
        # Show dataset size info
        st.write(f"Selected data: {mcb_chosen.shape[0]} samples × {mcb_chosen.shape[1]} features")
        
        # Check for problematic samples (all zeros)
        row_sums = mcb_chosen.sum(axis=1)
        zero_samples = (row_sums == 0).sum()
        if zero_samples > 0:
            st.warning(f'⚠️ Found {zero_samples} sample(s) with all zero values. These may cause issues with beta diversity calculations.')
        
        beta_measure_arg = beta_measure
        if beta_measure == 'bray-curtis':
            beta_measure_arg = 'bray'
        
        try:
            # Use file-based approach to avoid pyRserve timeout issues
            input_path = '/app/data/temp_beta_input.csv'
            output_path = '/app/data/temp_beta_output.csv'
            status_path = '/app/data/temp_beta_status.txt'
            
            # Save input data
            mcb_chosen.to_csv(input_path, index=False)
            
            # Remove old files
            for path in [output_path, status_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            # Write initial status
            with open(status_path, 'w', encoding='utf-8') as f:
                f.write("RUNNING")
            
            # Run beta diversity in thread - R will update status when done
            def run_beta():
                try:
                    pool = get_r_connection_pool()
                    with pool.get_connection() as conn:
                        r_code = f"beta_dim_red_from_file('{input_path}', '{beta_measure_arg}', '{dim_red}', '{output_path}', '{status_path}')"
                        # Use eval() instead of eval_void() - R function returns TRUE/FALSE
                        # Any exception means R couldn't start, otherwise R writes status
                        try:
                            conn.eval(r_code)
                        except Exception:  # pylint: disable=broad-except
                            pass  # Ignore all errors - R writes status file
                except Exception:  # pylint: disable=broad-except
                    pass  # Ignore connection errors
            
            thread = threading.Thread(target=run_beta, daemon=True)
            thread.start()
            
            # Poll status with progress
            max_wait = 300  # 5 minutes for large datasets
            wait_interval = 2
            elapsed = 0
            
            with st.spinner(f'Computing beta diversity ({beta_measure}) with {dim_red}... This may take 1-2 minutes for large datasets.'):
                while elapsed < max_wait:
                    if os.path.exists(status_path):
                        with open(status_path, 'r', encoding='utf-8') as f:
                            status = f.read().strip()
                        
                        if status == "SUCCESS":
                            # Read results
                            result_df = pd.read_csv(output_path)
                            
                            if dim_red == 'NMDS':
                                stress = result_df['stress'].iloc[0]
                                # NMDS uses MDS1, MDS2 columns
                                bt = result_df[['MDS1', 'MDS2']].values
                                st.success(f'NMDS stress value: {stress:.4f}')
                            else:
                                # PCoA uses V1, V2 columns
                                bt = result_df[['V1', 'V2']].values
                            
                            bt = pd.DataFrame(bt, columns=['PC1', 'PC2'])
                            bt['bin_var'] = np.array(y).astype(str)
                            fig = px.scatter(bt, x='PC1', y='PC2', color='bin_var', 
                                           title=f'Beta Diversity using {beta_measure} and {dim_red}')
                            # Update trace names if int_to_str_var is available
                            if 'int_to_str_var' in st.session_state:
                                fig.for_each_trace(lambda t: t.update(name=st.session_state.int_to_str_var[int(t.name)]))
                            st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)
                            break
                        elif status.startswith("ERROR:"):
                            raise RuntimeError(f"Beta diversity failed: {status}")
                    
                    time.sleep(wait_interval)
                    elapsed += wait_interval
                else:
                    raise TimeoutError(f"Beta diversity calculation timed out after {max_wait} seconds")
                    
        except Exception as e:
            error_msg = str(e)
            st.error(f'Beta diversity calculation failed: {error_msg}')
            
            # Provide helpful suggestions
            if 'timeout' in error_msg.lower() or 'TimeoutError' in str(type(e)):
                st.warning('⚠️ The calculation took too long. Try:\n'
                          '- Using fewer taxonomic levels\n'
                          '- Using PCoA instead of NMDS (faster)\n'
                          '- Filtering samples with low read counts')
            elif 'all 0' in error_msg.lower() or 'zero' in error_msg.lower():
                st.warning('💡 Remove samples with all zero values before running beta diversity.')
            else:
                st.warning('💡 Common causes:\n'
                          '- Samples with all zero values\n'
                          '- Very sparse data matrix\n'
                          '- Consider removing low-abundance samples')
            raise e

    # =========================================================================
    # TAB 3: DIFFERENTIAL ABUNDANCE (ANCOM-BC)
    # =========================================================================
    with tab3:
        if st.session_state.otu_type == 'Read counts':
            st.write("This section is about showing results related to ANCOM BC calulcations.")
            st.warning('**Warning:** If the data is not formatted properly, you may get unusual results.')

            st.write("**The 3 dataframes input into ANCOM:** ")
            st.write(st.session_state.ancom_df)
            st.write(st.session_state.ancom_y)
            st.write(st.session_state.tax_tab)
            
            # Show data size information
            n_samples = st.session_state.ancom_df.shape[1] - 1  # Exclude taxonomy column
            n_taxa = st.session_state.ancom_df.shape[0]
            st.info(f"📊 Data size: {n_samples} samples × {n_taxa} taxa")
            # if n_taxa > 500:
            #     st.warning(f"⚠️ Large dataset detected ({n_taxa} taxa). ANCOM-BC several minutes to complete.")
            # elif n_taxa > 1000:
            #     st.warning(f"⚠️ Very large dataset ({n_taxa} taxa). ANCOM-BC may take 3-8 minutes or more. Consider using a higher taxonomic level (e.g., family instead of genus) to reduce computation time.")

            st.write("**Choose which taxa level to use for ANCOM:** ")
            level_input = st.radio('Taxa level to consider', all_levels[1:st.session_state.last_level])
            
            st.write("**💡 Tip:** Higher taxonomic levels (e.g., Phylum, Class) have fewer taxa and run faster than lower levels (e.g., Genus, Species).")
            
            # Add feature filtering options
            st.write("**⚡ ANCOM-BC Filtering Options:**")
            st.info("🔍 ANCOM-BC will filter taxa internally based on prevalence. Higher values = fewer taxa, faster analysis.")
            
            col1, col2 = st.columns(2)
            with col1:
                prv_cut_percent = st.slider('ANCOM-BC prevalence filter (%)', 1, 50, 10, 1,
                                            help='Taxa must appear in at least this % of samples. ANCOM-BC applies this filter internally. Recommended: 10-15%.')
            with col2:
                max_samples = st.number_input('Max samples (0=all)', min_value=0, max_value=1000, value=0, step=50,
                                             help='Limit analysis to first N samples. Use to speed up very large datasets.')
            
            # st.info(f"ℹ️ All {n_taxa} taxa will be sent to ANCOM-BC, which will filter internally using {prv_cut_percent}% prevalence threshold")

            # Add button to trigger ANCOM
            run_ancom_button = st.button('🚀 Run ANCOM-BC Analysis', type='primary', help='Click to start the differential abundance analysis')

            @st.cache_data
            def perform_ancom(dff, yy, tax_tabb, level, prv_cut_pct=10, max_samp=0):  # pylint: disable=redefined-outer-name
                # Make copies to avoid modifying cached data
                dff = dff.copy()
                yy = yy.copy()
                tax_tabb = tax_tabb.copy()
                
                # Limit samples if requested
                if max_samp > 0 and len(yy) > max_samp:
                    st.info(f"🔄 Subsampling from {len(yy)} to {max_samp} samples")
                    # Keep sample IDs from yy and filter dff to match
                    sample_ids = yy.index[:max_samp]
                    yy = yy.iloc[:max_samp].copy()
                    # Filter columns in dff (samples are columns, except first which might be taxonomy)
                    if 'taxonomy' in dff.columns:
                        dff = dff[['taxonomy'] + [col for col in sample_ids if col in dff.columns]].copy()
                    else:
                        dff = dff[[col for col in sample_ids if col in dff.columns]].copy()
                
                # No Python pre-filtering - send all taxa to ANCOM-BC
                # ANCOM-BC will filter internally using prv_cut parameter
                
                # Display dataset dimensions
                n_samples = len(yy)
                n_taxa = len(dff)
                complexity = n_taxa * n_samples
                # st.info(f"📏 Sending full dataset to ANCOM-BC: {n_taxa} taxa × {n_samples} samples (complexity: {complexity:,})")

                # Warn about performance with large datasets
                # if complexity > 500000:
                #     st.error(f"⚠️ Dataset extremely large ({n_taxa} taxa × {n_samples} samples = {complexity:,}). This may take several minutes or timeout. Consider using a higher prevalence filter.")
                # elif complexity > 200000:
                #     st.warning(f"⚠️ Large dataset ({n_taxa} taxa × {n_samples} samples = {complexity:,}). Expected time: 10-20 minutes. Consider increasing prevalence filter for faster results.")
                # elif n_samples > 300:
                #     st.warning(f"⚠️ Very large sample size ({n_samples} samples) will result in long computation time (5-10 minutes).")
                # elif n_samples > 200:
                #     st.warning(f"⚠️ Large sample size ({n_samples} samples) may take 1-5 minutes.")
                
                # Save dataframes to CSV files to avoid passing large data through pyRserve
                # Use absolute paths that are accessible from both containers
                # Save with index=True to preserve the row index as first column
                dff.to_csv('/app/data/temp_df.csv', index=True)
                yy.to_csv('/app/data/temp_y.csv', index=True)
                tax_tabb.to_csv('/app/data/temp_tab.csv', index=True)

                # Output paths
                output_path = '/app/data/temp_ancom_results.csv'
                status_path = '/app/data/temp_ancom_status.txt'
                
                # Remove old files if they exist
                for path in [output_path, status_path]:
                    if os.path.exists(path):
                        os.remove(path)
                
                # Write initial status
                with open(status_path, 'w', encoding='utf-8') as f:
                    f.write("RUNNING")
                
                # Convert prv_cut_pct to decimal for R (prv_cut)
                prv_cut_r = prv_cut_pct / 100.0  # e.g., 10% -> 0.10
                
                # Call R function in a separate thread to avoid blocking
                def run_ancom():
                    try:
                        pool = get_r_connection_pool()
                        with pool.get_connection() as conn:
                            r_code = f"perform_ancom_from_files('/app/data/temp_df.csv', '/app/data/temp_y.csv', '/app/data/temp_tab.csv', '{level}', '{output_path}', '{status_path}', {prv_cut_r})"
                            try:
                                # Use eval() - R function returns TRUE/FALSE
                                conn.eval(r_code)
                            except Exception as e:  # pylint: disable=broad-except
                                # Expected: R process may return before completing since it writes to file
                                # Only log if it's not a timeout/connection issue
                                if "timeout" not in str(e).lower() and "connection" not in str(e).lower():
                                    print(f"R call returned early (expected for background process): {e}", file=sys.stderr)
                    except Exception as e:  # pylint: disable=broad-except
                        # Connection pool errors are expected for long-running processes
                        if "timeout" not in str(e).lower():
                            print(f"Python connection note: {e}", file=sys.stderr)
                
                # Start R function in background thread
                thread = threading.Thread(target=run_ancom, daemon=True)
                thread.start()
                
                # Poll the status file
                # Timeout scales with complexity: longer for large datasets without pre-filtering
                base_timeout = 150  # 2.5 minutes base
                extra_timeout = (complexity // 100000) * 150  # Add 2.5min per 100k complexity
                max_wait = min(base_timeout + extra_timeout, 300)  # Cap at 5 minutes
                wait_interval = 2  # Check every 2 seconds
                elapsed = 0
                
                # Return the status path and timing info for external polling
                return {
                    'status_path': status_path,
                    'output_path': output_path,
                    'max_wait': max_wait,
                    'wait_interval': wait_interval,
                    'n_samples': n_samples
                }

            # Only run ANCOM when button is clicked
            if run_ancom_button:
                # Calculate dynamic estimated time based on dataset size
                complexity = n_taxa * n_samples
                if complexity < 10000:
                    estimated_time = "10-20 seconds"
                elif complexity < 50000:
                    estimated_time = "20-60 seconds"
                elif complexity < 100000:
                    estimated_time = "1-2 minutes"
                elif complexity < 200000:
                    estimated_time = "2-3.5 minutes"
                elif complexity < 500000:
                    estimated_time = "3.5-6 minutes"
                else:
                    estimated_time = "6-10 minutes"
                
                # Start ANCOM process (returns polling info)
                polling_info = perform_ancom(st.session_state.ancom_df, st.session_state.ancom_y, st.session_state.tax_tab, level_input, prv_cut_percent, max_samples)
                
                # Poll for results with cancel button (outside cached function)
                status_path = polling_info['status_path']
                output_path = polling_info['output_path']
                max_wait = polling_info['max_wait']
                wait_interval = polling_info['wait_interval']
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                cancel_placeholder = st.empty()
                
                elapsed = 0
                results = None
                cancelled = False
                
                while elapsed < max_wait and not cancelled:
                    # Check for cancel button
                    if cancel_placeholder.button('🛑 Cancel Analysis', key=f'cancel_ancom_{elapsed}'):
                        cancelled = True
                        with open(status_path, 'w', encoding='utf-8') as f:
                            f.write("CANCELLED")
                        progress_bar.empty()
                        status_text.empty()
                        cancel_placeholder.empty()
                        st.warning("⚠️ Analysis cancelled by user.")
                        break
                    
                    if os.path.exists(status_path):
                        with open(status_path, 'r', encoding='utf-8') as f:
                            status = f.read().strip()
                        
                        if status == "SUCCESS":
                            progress_bar.progress(1.0)
                            status_text.success("✅ ANCOM-BC completed!")
                            cancel_placeholder.empty()
                            if os.path.exists(output_path):
                                results = pd.read_csv(output_path)
                                break
                            else:
                                st.error("ANCOM completed but output file not found")
                                break
                        elif status.startswith("ERROR:"):
                            progress_bar.empty()
                            status_text.empty()
                            cancel_placeholder.empty()
                            st.error(f"ANCOM-BC failed: {status}")
                            break
                        elif status == "CANCELLED":
                            progress_bar.empty()
                            status_text.empty()
                            cancel_placeholder.empty()
                            break
                    
                    # Update progress
                    progress = elapsed / max_wait
                    progress_bar.progress(progress)
                    mins_elapsed = elapsed // 60
                    secs_elapsed = elapsed % 60
                    mins_remaining = (max_wait - elapsed) // 60
                    status_text.info(f"⏳ Running ANCOM-BC... {mins_elapsed}m {secs_elapsed}s elapsed (est. {mins_remaining}m remaining)")
                    
                    time.sleep(wait_interval)
                    elapsed += wait_interval
                
                if elapsed >= max_wait and results is None:
                    progress_bar.empty()
                    status_text.empty()
                    cancel_placeholder.empty()
                    st.error(f"ANCOM-BC timed out after {max_wait//60} minutes. Try filtering more aggressively.")
                
                # Display results if successful
                if results is not None:
                    labels = sorted(list(st.session_state.ancom_y['bin_var'].unique()))
                    
                    st.write('**ANCOM results:** ')
                    st.write(f'Comparison of {labels[1]} vs {labels[0]} groups')
                    
                    # Validate results
                    if len(results) == 0:
                        st.error("⚠️ ANCOM-BC returned no results. The analysis may not have completed properly.")
                    else:
                        st.success(f"✅ Analysis returned {len(results)} taxa with differential abundance results")
                    
                    st.write(results)
                    #st.write(set(results.iloc[:, 0]).difference(set(list(st.session_state.tax_tab[level_input].unique()))))
                    csv_res = results.to_csv(index=False)
                    st.download_button("Export ANCOM results", csv_res, 'ancombc.csv', key='download15')
                    
                    # Process and visualize results
                    results['pos_lfc'] = (results.iloc[:, 2] > 0)

                    pos_results = results[results['pos_lfc']]
                    neg_results = results[~results['pos_lfc']]

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
                        st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)
                    
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
                        st.plotly_chart(fig, use_container_width=True, config=st.session_state.config)
            else:
                st.info('👆 Click the button above to start ANCOM-BC analysis')
        else:
            st.info('ℹ️ ANCOM-BC analysis is only available for read count data. Your data appears to be relative abundance.')

else:
    st.error('Please upload files first in the Upload data tab.')
