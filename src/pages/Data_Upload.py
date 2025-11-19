import sys

import pandas as pd
import streamlit as st
from st_pages import add_page_title
from streamlit_plotly_events import plotly_events
from streamlit_tags import st_tags

sys.path.insert(0, './src/utils')
from helpers import *

add_page_title()

st.session_state.config = {
    'toImageButtonOptions': {
        'format': 'svg',  # one of png, svg, jpeg, webp
        'filename': 'custom_image',
        'height': 500,
        'width': 700,
        'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
    }
}

st.write('**You can either use a sample data or upload your own data.**')
data_option = st.radio('Choose an option:', ('Sample data', 'Upload your own data'))

if data_option == 'Upload your own data':

    try:
        #st.session_state.metadata_raw, st.session_state.mcb_raw, st.session_state.proceed = load_data()
        metadata_raw, mcb_raw, proceed, otu_separator, otu_type, int_to_str_var, level_to_int, last_level = load_data()
        
        if proceed:
            st.session_state.metadata_raw, st.session_state.mcb_raw, st.session_state.proceed, st.session_state.otu_separator, st.session_state.otu_type, st.session_state.int_to_str_var, st.session_state.level_to_int, st.session_state.last_level = metadata_raw, mcb_raw, proceed, otu_separator, otu_type, int_to_str_var, level_to_int, last_level
        else:
            st.session_state.proceed = False

        if 'proceed' in st.session_state.keys() and st.session_state.proceed:
            
            mcb_to_taxa.clear()
            mcbs = mcb_to_taxa(st.session_state.mcb_raw, st.session_state.metadata_raw, otu_separator, st.session_state.level_to_int, st.session_state.last_level)
            
            if st.session_state.last_level >= st.session_state.level_to_int['kingdom']:
                st.session_state.mcb_k = mcbs[st.session_state.level_to_int['kingdom']-1]
            if st.session_state.last_level >= st.session_state.level_to_int['phylum']:
                st.session_state.mcb_p = mcbs[st.session_state.level_to_int['phylum']-1]
                st.write('**Chunk of data uploaded (phyla data):**')
                st.write(st.session_state.mcb_p.head())     
            if st.session_state.last_level >= st.session_state.level_to_int['class']:
                st.session_state.mcb_c = mcbs[st.session_state.level_to_int['class']-1]
            if st.session_state.last_level >= st.session_state.level_to_int['order']:
                st.session_state.mcb_o = mcbs[st.session_state.level_to_int['order']-1]
            if st.session_state.last_level >= st.session_state.level_to_int['family']:
                st.session_state.mcb_f = mcbs[st.session_state.level_to_int['family']-1]
            if st.session_state.last_level >= st.session_state.level_to_int['genus']:
                st.session_state.mcb_g = mcbs[st.session_state.level_to_int['genus']-1]
            if st.session_state.last_level >= st.session_state.level_to_int['species']:
                st.session_state.mcb_s = mcbs[st.session_state.level_to_int['species']-1]

            st.session_state.y = st.session_state.mcb_k['bin_var']

            # ANCOM stuff
            st.session_state.ancom_df = st.session_state.mcb_raw.set_index('taxonomy') # assays
            st.session_state.ancom_df.columns = [f'sample{i}' for i in range(len(st.session_state.ancom_df.columns)) if st.session_state.ancom_df.columns[i] != 'taxonomy']

            st.session_state.ancom_y = pd.DataFrame(st.session_state.y).reset_index()
            st.session_state.ancom_y['id'] = [f'sample{i}' for i in range(len(st.session_state.ancom_y.id))]
            st.session_state.ancom_y.bin_var = st.session_state.ancom_y.bin_var.replace({0: st.session_state.int_to_str_var[0], 1: st.session_state.int_to_str_var[1]})
            st.session_state.ancom_y = st.session_state.ancom_y.set_index('id')

            common_ids = set(st.session_state.ancom_df.columns).intersection(set(st.session_state.ancom_y.index))
            st.session_state.ancom_df = st.session_state.ancom_df[common_ids]
            st.session_state.ancom_y = st.session_state.ancom_y.loc[common_ids]

            tax_tab = pd.DataFrame(st.session_state.mcb_raw['taxonomy'], columns = ['taxonomy']) # tax_tab
            all_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
            tax_tab[all_levels[:st.session_state.last_level]] = pd.DataFrame(tax_tab['taxonomy'].str.split(st.session_state.otu_separator, expand = True))
            st.session_state.tax_tab = tax_tab.set_index('taxonomy')

            proceed = False

    except Exception as e:
        st.warning('If you\'re trying to upload new data, make sure to choose the right separator, the right type of data, and the right last taxa level.')
        raise e

else:
    load_sample = st.button('Click to load sample metadata file and OTU table')
    if load_sample:
        metadata_raw = pd.read_csv('./data/perfect_metadata_vienna.csv')
        metadata_raw = metadata_raw[['bin_var', 'id']].astype({'id': str}).set_index('id')
        #metadata_raw['bin_var'] = metadata_raw['id'].replace({''})
        mcb_raw = pd.read_csv('./data/perfect_mcb_vienna.csv')
        st.success('Sample data and metadata uploaded successfully.')
        st.write('**Metadata uploaded:**')
        st.write(metadata_raw)
        st.write('**Microbiota data table uploaded:**')
        st.write(mcb_raw)
        otu_separator = ';'
        otu_type = 'Read counts' 
        proceed = True
        int_to_str_var = {0: 'Meat', 1: 'Surface'}
        metadata_raw['bin_var'] = metadata_raw['bin_var'].replace({'Meat': 0, 'Surface': 1})
        level_to_int = {'kingdom': 1, 'phylum': 2, 'class': 3, 'order': 4, 'family': 5, 'genus': 6, 'species': 7}
        last_level = 'species'
        last_level = level_to_int[last_level]

        st.session_state.metadata_raw, st.session_state.mcb_raw, st.session_state.proceed, st.session_state.otu_separator, st.session_state.otu_type, st.session_state.int_to_str_var, st.session_state.level_to_int, st.session_state.last_level = metadata_raw, mcb_raw, proceed, otu_separator, otu_type, int_to_str_var, level_to_int, last_level

        if 'proceed' in st.session_state.keys() and st.session_state.proceed:
            mcb_to_taxa.clear()
            mcbs = mcb_to_taxa(st.session_state.mcb_raw, st.session_state.metadata_raw, otu_separator, st.session_state.level_to_int, st.session_state.last_level)

            if st.session_state.last_level >= st.session_state.level_to_int['kingdom']:
                st.session_state.mcb_k = mcbs[st.session_state.level_to_int['kingdom']-1]
            if st.session_state.last_level >= st.session_state.level_to_int['phylum']:
                st.session_state.mcb_p = mcbs[st.session_state.level_to_int['phylum']-1] 
            if st.session_state.last_level >= st.session_state.level_to_int['class']:
                st.session_state.mcb_c = mcbs[st.session_state.level_to_int['class']-1]
            if st.session_state.last_level >= st.session_state.level_to_int['order']:
                st.session_state.mcb_o = mcbs[st.session_state.level_to_int['order']-1]
            if st.session_state.last_level >= st.session_state.level_to_int['family']:
                st.session_state.mcb_f = mcbs[st.session_state.level_to_int['family']-1]
            if st.session_state.last_level >= st.session_state.level_to_int['genus']:
                st.session_state.mcb_g = mcbs[st.session_state.level_to_int['genus']-1]
            if st.session_state.last_level >= st.session_state.level_to_int['species']:
                st.session_state.mcb_s = mcbs[st.session_state.level_to_int['species']-1]

            st.session_state.y = st.session_state.mcb_k['bin_var']

            # ANCOM stuff
            st.session_state.ancom_df = st.session_state.mcb_raw.set_index('taxonomy') # assays
            st.session_state.ancom_df.columns = [f'sample{i}' for i in range(len(st.session_state.ancom_df.columns)) if st.session_state.ancom_df.columns[i] != 'taxonomy']

            st.session_state.ancom_y = pd.DataFrame(st.session_state.y).reset_index()
            st.session_state.ancom_y['id'] = [f'sample{i}' for i in range(len(st.session_state.ancom_y.id))]
            st.session_state.ancom_y.bin_var = st.session_state.ancom_y.bin_var.replace({0: st.session_state.int_to_str_var[0], 1: st.session_state.int_to_str_var[1]})
            st.session_state.ancom_y = st.session_state.ancom_y.set_index('id')

            common_ids = set(st.session_state.ancom_df.columns).intersection(set(st.session_state.ancom_y.index))
            st.session_state.ancom_df = st.session_state.ancom_df[common_ids]
            st.session_state.ancom_y = st.session_state.ancom_y.loc[common_ids]

            tax_tab = pd.DataFrame(st.session_state.mcb_raw['taxonomy'], columns = ['taxonomy']) # tax_tab
            all_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
            tax_tab[all_levels[:st.session_state.last_level]] = pd.DataFrame(tax_tab['taxonomy'].str.split(st.session_state.otu_separator, expand = True))
            st.session_state.tax_tab = tax_tab.set_index('taxonomy')

            proceed = False

