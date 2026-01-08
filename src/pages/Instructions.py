import pandas as pd
import streamlit as st
from st_pages import add_page_title

add_page_title()

st.write('**If some specific technique (like ANCOMBC) is used in the website, you should be able to find it on the references in the paper.**')
st.write('**Important**: Please wait until a task is executed before initiating new tasks to prevent website from crashing. Tip: Pay attention to the “Running” icon at the top right side of the webpage to know the status of a task.')

st.divider()
st.header('Upload data')
st.write('The microbiota data and metadata tables should be separately uploaded to start an analysis. The microbiota data can be uploaded by going to the "Upload data" tab in the "Explore" section.')
st.write('**Note:** The data needs to follow a specific format. Microbiota data must have only 2 types of columns: a single column named "taxonomy" which contains the specific taxa (kingdom, phylum...), and all the other columns should the sample IDs. These IDs should be identical to the ones in the metadata file in the "id" column to allow the website to associate the microbiota profile of a sample with its metadata.')
st.write('As it is not always possible to obtain the highest possible taxonomic resolution, e.g. up to species level in all instances, the website allows the possibility to input the taxonomic unit (e.g. family, genus, or species). Please use the smallest taxonomic unit present in any given sample.  The separator used in the "taxonomy" column in the microbiota data file should also be identified to allow the website to properly parse the data. Finally, either read counts or relative abundance (specific processing/analysis needs to be performed for each) can be inputted.')
st.write('For the **metadata file** it must have a column named "id" with sample IDs, and another column named "bin_var" which contains the condition/phenomena of interest (should only have 2 possible values, like positive/negative or present/not present...).')
st.write('Finally, when you upload the metadata file, you need to assign the 2 variables in the "bin_var" column to 0 and 1. This is needed to compute some metrics like recall and specificity in the AI analysis.')
st.write('Here is how the data files could look like (metadata on the left, microbiota data on the right):')
col2, col1 = st.columns(2, gap='large')
with col1:
    mcb = pd.read_csv('./data/perfect_mcb_vienna.csv')
    st.write(mcb)
with col2:
    meta = pd.read_csv('./data/perfect_metadata_vienna.csv')
    st.write(meta)

st.divider()

st.header('Statistical analysis')
st.write('This tab is for doing exploratory analysis for a specific taxa level.')
st.write('It is highly recommended to remove taxa with labels such as NA, ambiguous etc. that is deemed not to be essential for the analysis. You can do this as the first step in this tab if you want.')
st.write('Then, you can choose a taxa level from the funnel plot by clicking on it (on the line on the plot), and the rest of the analysis will show up.')
st.write('The rest should be self-explanatory and further instructions are given throughout the analysis. The final part of the analysis leverages dimensionality reduction techniques to determine if there is a separation of the two groups. Even if the rest of the analysis in this page concerns only the chosen taxa level, we have provided the option in this last part to apply dimensionality reduction on all levels.')

st.divider()
st.header('Ecological measures')
st.write('The instructions in this tab should be self-explanatory. For the used techniques like ANCOMBC, you can look up the references that are given in the original paper.')

st.divider()
st.header('AI predictions')
st.write('AI predictions allows the identification of features that predicts a certain condition and its control. Below flowchart provides a high level overview of the process steps used in the AI predictions tab: ')
import os

if os.path.exists('./ims/pipeline.png'):
    st.image('./ims/pipeline.png')
st.write('The instructions are given step by step in the tab itself, and the explanations of most techniques can be found in the paper.')
st.write('However, the following are some specific information. First, you have to determine the cutoffs to prune the data given the number of zero-values and data variance. Then, you can choose the taxa levels that may be of interest to you in the "Feature selection" part in the tab. In this part, you\'ll see "phyla ratios", which are the ratios of all phyla in the dataset. Ratio between phyla have been shown to useful and thus have been included in the predictions even though it is not part of the original dataset.')
st.write('Attention: Because of the way the website has been coded, you have to press "Run AI now" when you want to run the AI algorithms and see the results. If you decide to change some parameters in the previous steps and run again, remember to select "Don\'t run AI predictions for now" before changing the parameters, otherwise each change will trigger a new AI run, which can crash the website.')

st.divider()
st.write('**Disclaimer**: Our platform is designed for exploratory data analysis. We do not guarantee the accuracy or applicability of the results for specific purposes. Always consult with subject matter experts and/or statisticians when interpreting the results.')
st.write('If you encounter any issues or need further assistance, please contact taha.zakariya99@gmail.com or ShaillayKumar.Dogra@rd.nestle.com and for general questions contact balamurugan.jagadeesan@rdls.nestle.com')

st.divider()
st.header('Tutorials')
st.write('We provide some tutorials to help you some technical parts of the website:')
st.write('1. ANCOMBC (Ecological measures tab)')
st.write('http://www.bioconductor.org/packages/release/bioc/vignettes/ANCOMBC/inst/doc/ANCOMBC.html')
st.write('https://github.com/FrederickHuangLin/ANCOM-BC-Code-Archive')
st.write('2. SHAP impact (AI predictions tab)')
st.write('https://www.youtube.com/watch?v=-taOhqkiuIo&t=169s')
st.write('https://christophm.github.io/interpretable-ml-book/shap.html')
st.write('https://github.com/slundberg/shap')

st.divider()
st.header('References')
st.write('1) Sample data -Zwirzitz B, Wetzels SU, Dixon ED, et al. "The sources and transmission routes of microbial populations throughout a meat processing facility"  NPJ Biofilms Microbiomes. 2020;6(1):26. - https://www.nature.com/articles/s41522-020-0136-z')
st.write('2) Vegan package of R -Oksanen J, Simpson G, Blanchet F, Kindt R, Legendre P, Minchin P, O\'Hara R, Solymos P, Stevens M, Szoecs E, Wagner H, Barbour M, Bedward M, Bolker B, Borcard D, Carvalho G, Chirico M, De Caceres M, Durand S, Evangelista H, FitzJohn  R, Friendly M, Furneaux B, Hannigan G, Hill M, Lahti L, McGlinn D, Ouellette M, Ribeiro Cunha E, Smith T, Stier A, Ter Braak C, Weedon J (2022). _vegan: Community Ecology Package_. R package version 2.6-4, - https://CRAN.R-project.org/package=vegan')
st.write('3) ANCOMBC -Kaul A, Mandal S, Davidov O, Peddada SD (2017). “Analysis of microbiome data in the presence of excess zeros.” Frontiers in microbiology, 8, 2114. - https://www.frontiersin.org/articles/10.3389/fmicb.2017.02114/full Lin H, Peddada SD (2020). “Analysis of compositions of microbiomes with bias correction.” Nature communications, 11(1), 1–11. - https://www.nature.com/articles/s41467-020-17041-7')
st.write('4) SHAP -Lundberg, S.M., Erion, G., Chen, H. et al. "From local explanations to global understanding with explainable AI for trees." Nat Mach Intell 2, 56–67 (2020) - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7326367/')