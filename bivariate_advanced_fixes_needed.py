import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import stats
from scipy.stats import pearsonr
import base64

# Function to calculate statistics
def calculate_statistics(selected_columns):
    # Core statistics
    stats = selected_columns.agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    
    # Additional statistics
    mode_values = selected_columns.mode().iloc[0]  # First mode if multiple exist
    percentile_25 = selected_columns.quantile(0.25)
    percentile_75 = selected_columns.quantile(0.75)
    null_values = selected_columns.isnull().sum()
    null_percentage = (null_values / len(selected_columns)) * 100
    coefficient_variation = (selected_columns.std() / selected_columns.mean()) * 100
    skewness = selected_columns.skew()
    kurtosis = selected_columns.kurt()

    # IQR and outlier calculations
    IQR = percentile_75 - percentile_25
    lower_bound = percentile_25 - 1.5 * IQR
    upper_bound = percentile_75 + 1.5 * IQR
    outliers = ((selected_columns < lower_bound) | (selected_columns > upper_bound)).sum()
    percentage_outliers = (outliers / len(selected_columns)) * 100

    # Combine all statistics with unique indices
    additional_stats = pd.DataFrame({
        'Mode': mode_values,
        '25th Percentile': percentile_25,
        '75th Percentile': percentile_75,
        'Null Values': null_values,
        '% Null Values': null_percentage,
        'No. of Outliers': outliers,
        '% Outliers': percentage_outliers,
        'Coefficient of Variation': coefficient_variation,
        'Skewness': skewness,
        'Kurtosis': kurtosis
    }).T

    # Concatenate stats and additional_stats
    final_stats = pd.concat([stats, additional_stats], axis=0)
    return final_stats

# Function to remove outliers from a column
def remove_outliers_from_column(df, column_name):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

# Function to create box plot
def create_box_plot(data, columns):
    fig = px.box(data, y=columns, title="Box Plot Comparison")
    fig.update_traces(boxmean=True)
    fig.update_layout(
        height=100 + (len(columns) * 100),  # Dynamic height based on number of columns
        showlegend=False,
        yaxis_title="Value",
        boxmode="group"
    )
    return fig

# Function to create histogram with distribution curve
def create_histogram(data, column):
    # Create histogram data
    hist_data = [data[column].dropna()]
    group_labels = [column]
    
    # Create distplot with curve
    fig = ff.create_distplot(
        hist_data, 
        group_labels,
        bin_size=(data[column].max() - data[column].min()) / 30,  # Automatic bin size
        show_rug=False
    )
    
    fig.update_layout(
        title=f"Distribution Plot for {column}",
        xaxis_title="Values",
        yaxis_title="Density",
        height=400
    )
    return fig

# Function to detect table type
def detect_table_type(df):
    # Check if the table is a Cathode table
    gdl_columns = [col for col in df.columns if "gdl" in col.lower()]
    cl_columns = [col for col in df.columns if "cl" in col.lower() or "catalyst" in col.lower()]
    
    if gdl_columns and cl_columns:
        return "Cathode"
    else:
        return "Other"

# Function to create a download link for Plotly figures
def download_plotly_fig(fig, filename):
    html = fig.to_html(full_html=False)
    return html

def create_categorical_visualizations(data, column):
    # Calculate value counts and percentages
    value_counts = data[column].value_counts()
    percentages = (value_counts / len(data)) * 100
    stats_df = pd.DataFrame({
        'Count': value_counts,
        'Percentage (%)': percentages.round(2)
    })
    stats_df.index.name = 'index'
    
    # Create bar plot
    fig = px.bar(
        stats_df.reset_index(),
        x='index',
        y='Count',
        title=f"Value Counts for {column}",
        labels={'index': column, 'Count': 'Count'},
        text='Count'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count",
        height=500
    )
    
    return stats_df, fig

# Function to apply data filters
def apply_data_filters(df):
    st.header("Data Filtering")
    
    # Create expandable filter section
    with st.expander("🔍 Filter Data", expanded=False):
        # Get all column names
        all_columns = df.columns.tolist()
        
        # Initialize filtered_df with the original dataframe
        filtered_df = df.copy()
        
        # NaN handling option
        nan_handling = st.radio(
            "Handle NaN/missing values of the uploaded data:",
            ["Include rows with NaN values", "Exclude rows with NaN values", "Only show rows with NaN values"],
            index=0,  # Default: Include NaN values
            key=f"nan_handling_{st.session_state.page}"
        )
        
        # Allow multiple columns to filter
        columns_to_filter = st.multiselect(
            "Select columns to filter:",
            all_columns,
            key=f"filter_columns_{st.session_state.page}"
        )
        
        # Create filters for each selected column
        filters = {}
        for column in columns_to_filter:
            col_type = df[column].dtype
            
            if col_type in ['object', 'category']:
                # Categorical filter
                # Get unique values, handling NaN appropriately
                unique_values = df[column].dropna().unique().tolist()
                
                # Add a "NaN/Missing" option if Include or Only show NaN is selected
                if nan_handling != "Exclude rows with NaN values" and df[column].isna().any():
                    unique_values_with_nan = unique_values.copy()
                    unique_values_with_nan.append("NaN/Missing")
                    
                    selected_values = st.multiselect(
                        f"Select values for '{column}':",
                        unique_values_with_nan,
                        default=unique_values_with_nan if nan_handling != "Only show rows with NaN values" else ["NaN/Missing"],
                        key=f"filter_{column}_{st.session_state.page}"
                    )
                    
                    # Store selection without the NaN placeholder
                    filters[column] = [val for val in selected_values if val != "NaN/Missing"]
                    
                    # Store whether NaN was selected
                    filters[f"{column}_include_nan"] = "NaN/Missing" in selected_values
                else:
                    selected_values = st.multiselect(
                        f"Select values for '{column}':",
                        unique_values,
                        default=unique_values if nan_handling != "Only show rows with NaN values" else [],
                        key=f"filter_{column}_{st.session_state.page}"
                    )
                    filters[column] = selected_values
                    filters[f"{column}_include_nan"] = nan_handling == "Only show rows with NaN values"
            else:
                # Numerical filter (slider)
                min_val = float(df[column].min())
                max_val = float(df[column].max())
                step = (max_val - min_val) / 100 if (max_val - min_val) > 0 else 0.1
                
                selected_range = st.slider(
                    f"Select range for '{column}':",
                    min_val,
                    max_val,
                    (min_val, max_val),
                    step=step,
                    key=f"filter_{column}_{st.session_state.page}"
                )
                filters[column] = selected_range
                
                # Add NaN checkbox for numerical columns
                include_nan_numeric = st.checkbox(
                    f"Include NaN/missing values for '{column}'",
                    value=nan_handling == "Include rows with NaN values" or nan_handling == "Only show rows with NaN values",
                    key=f"include_nan_{column}_{st.session_state.page}"
                )
                filters[f"{column}_include_nan"] = include_nan_numeric
        
        # Add reset filters button
        if st.button("Reset All Filters"):
            filters = {}
            filtered_df = df.copy()
        
        # Apply filters
        if filters:
            # Apply global NaN handling if no column-specific filters
            # if not columns_to_filter:
            if nan_handling == "Exclude rows with NaN values":
                filtered_df = filtered_df.dropna()
            elif nan_handling == "Only show rows with NaN values":
                # Keep only rows with at least one NaN
                filtered_df = filtered_df[filtered_df.isna().any(axis=1)]
            
            # Apply column-specific filters
            for column, filter_value in filters.items():
                # Skip the NaN indicator entries
                if column.endswith("_include_nan"):
                    continue
                
                col_type = df[column].dtype
                include_nan = filters.get(f"{column}_include_nan", nan_handling == "Include rows with NaN values")
                
                if col_type in ['object', 'category']:
                    # For categorical columns, filter based on selected values
                    if include_nan:
                        # Include both selected values and NaN
                        filtered_df = filtered_df[
                            filtered_df[column].isin(filter_value) | filtered_df[column].isna()
                        ]
                    else:
                        # Only include selected values, exclude NaN
                        filtered_df = filtered_df[filtered_df[column].isin(filter_value)]
                else:
                    # For numerical columns, apply range filter
                    if include_nan:
                        # Include values in range or NaN
                        filtered_df = filtered_df[
                            ((filtered_df[column] >= filter_value[0]) & 
                            (filtered_df[column] <= filter_value[1])) | 
                            filtered_df[column].isna()
                        ]
                    else:
                        # Only include values in range, exclude NaN
                        filtered_df = filtered_df[
                            (filtered_df[column] >= filter_value[0]) & 
                            (filtered_df[column] <= filter_value[1])
                        ]
            
            # Handle "Only show NaN values" for all columns
            if nan_handling == "Only show rows with NaN values" and not columns_to_filter:
                filtered_df = filtered_df[filtered_df.isna().any(axis=1)]
    
    # # Show filter summary and filtered data
    # if 'filters' in locals() and filters:
    #     st.success(f"Filter applied! Remaining rows: {len(filtered_df)} (from original {len(df)})")
    #     st.write("Filtered Dataset Preview:")
    #     st.dataframe(filtered_df.head())
    # else:
    #     st.info("No filters applied. Showing full dataset.")
    
    # return filtered_df

# Show filter summary and filtered data
    original_rows = len(df)
    filtered_rows = len(filtered_df)

    # Check if any filtering was applied (global or column-specific)
    global_filter_applied = (nan_handling != "Include NaN values")
    column_filters_applied = 'filters' in locals() and filters and columns_to_filter

    if global_filter_applied or column_filters_applied:
        # Build filter description
        filter_descriptions = []
        
        if global_filter_applied:
            if nan_handling == "Exclude NaN values":
                filter_descriptions.append("Excluded rows with NaN values")
            elif nan_handling == "Only show NaN values":
                filter_descriptions.append("Showing only rows with NaN values")
        
        if column_filters_applied:
            filter_descriptions.append(f"Column-specific filters on {len(columns_to_filter)} column(s)")
        
        # Display results
        if filtered_rows != original_rows:
            st.success(f"Filters applied! Remaining rows: {filtered_rows} (from original {original_rows})")
            st.write("**Applied filters:**")
            for desc in filter_descriptions:
                st.write(f"• {desc}")
        else:
            st.info(f"Filters applied but no rows were removed. Total rows: {filtered_rows}")
            st.write("**Applied filters:**")
            for desc in filter_descriptions:
                st.write(f"• {desc}")
        
        st.write("**Filtered Dataset Preview:**")
        st.dataframe(filtered_df.head())
    else:
        st.info(f"No filters applied. Showing full dataset ({original_rows} rows).")

    return filtered_df
    
def reset_all_changes():
    """Reset all changes made to the dataframe"""
    if 'original_df' in st.session_state:
        st.session_state.df = st.session_state.original_df.copy()
        st.success("All changes have been reset to the original dataset!")
        st.rerun()
    else:
        st.warning("No original dataset found to reset to.")

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Univariate Analysis"

# Navigation buttons
st.sidebar.title("Type of Analysis")
if st.sidebar.button("Univariate Analysis"):
    st.session_state.page = "Univariate Analysis"
if st.sidebar.button("Bivariate Analysis"):
    st.session_state.page = "Bivariate Analysis"

# Univariate Analysis Page
if st.session_state.page == "Univariate Analysis":
    st.title("Univariate Analysis")
    # File Uploader
    uploaded_file = st.file_uploader("Upload a dataset for Univariate Analysis", type=['csv','xlsx'], accept_multiple_files=False, key="univariate_upload")

    if uploaded_file:
        # Read the uploaded Excel file into a DataFrame
        try:
            df = pd.read_excel(uploaded_file)
        except:
            df = pd.read_csv(uploaded_file)
        
        n_rows, n_columns = df.shape
        
        # Display the uploaded dataset
        st.write("Uploaded Dataset:")
        st.dataframe(df, hide_index=True)
        st.write(f"Rows: {n_rows}, Columns: {n_columns}")
        
        # Apply data filters
        df = apply_data_filters(df)
        
        # Detect table type
        table_type = detect_table_type(df)
        st.write(f"Detected Table Type: **{table_type}**")
        
        # Perform analysis based on table type
        if table_type == "Cathodes":
            # GDL Analysis
            st.header("GDL Analysis")
            gdl_columns = [col for col in df.columns if "gdl" in col.lower()]
            if gdl_columns:
                gdl_parameters_df = df[gdl_columns]
                gdl_numerical_df = gdl_parameters_df.select_dtypes(include=['number'])
                if not gdl_numerical_df.empty:
                    st.write("Statistical Analysis for GDL Parameters:")
                    gdl_stats = calculate_statistics(gdl_numerical_df)
                    st.dataframe(gdl_stats.T)
                else:
                    st.warning("No numerical GDL columns found in the dataset.")
            else:
                st.warning("No GDL columns found in the dataset.")

            # CL Analysis
            st.header("CL Analysis")
            cl_columns = [col for col in df.columns if "cl" in col.lower() or "catalyst" in col.lower()]
            if cl_columns:
                cl_parameters_df = df[cl_columns]
                cl_numerical_df = cl_parameters_df.select_dtypes(include=['number'])
                if not cl_numerical_df.empty:
                    st.write("Statistical Analysis for CL Parameters:")
                    cl_stats = calculate_statistics(cl_numerical_df)
                    st.dataframe(cl_stats.T)
                else:
                    st.warning("No numerical CL columns found in the dataset.")
            else:
                st.warning("No CL columns found in the dataset.")
        else:
            # General Numerical Analysis for Anode, Cell Test, etc.
            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
            if numerical_columns:
                st.write("Statistical Analysis for Numerical Columns:")
                numerical_stats = calculate_statistics(df[numerical_columns])
                st.dataframe(numerical_stats.T)
            else:
                st.warning("No numerical columns found in the dataset.")

        # Data Visualization Section
        st.header("Plots")
        
        # Combine all numerical columns
        all_numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if all_numerical_columns:
            # Column selection
            st.subheader("Select Columns for Visualization")
            selected_columns = st.multiselect(
                "Choose columns to visualize",
                all_numerical_columns,
                default=all_numerical_columns[:2] if len(all_numerical_columns) >= 2 else all_numerical_columns
            )

            if selected_columns:
                # Box Plot
                st.subheader("Box Plot")
                box_plot = create_box_plot(df, selected_columns)
                st.plotly_chart(box_plot, use_container_width=True)
                st.download_button(label = "Download as HTML",
                                    data = download_plotly_fig(box_plot, "Boxplot_plot"),
                                    file_name=f"Boxplot_univariate.html",
                                    mime= "text/html")
                
                # Histograms with Distribution Curves
                st.subheader("Distribution Analysis")
                for column in selected_columns:
                    hist_plot = create_histogram(df, column)
                    st.plotly_chart(hist_plot, use_container_width=True)
                    st.download_button(label = "Download as HTML",
                                       data = download_plotly_fig(hist_plot, "histogram_plot"),
                                       file_name=f"Histogram - {column}.html",
                                       mime= "text/html")
            else:
                st.warning("Please select at least one column for visualization.")
        else:
            st.warning("No numerical columns found in the dataset for visualization.")
        
        # Categorical Analysis Section
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_columns:
            st.header("Categorical Analysis")
            st.write("Statistical Analysis for Categorical Columns:")
            
            # Display categorical column statistics
            for col in categorical_columns:
                with st.expander(f"📊 {col} Analysis", expanded=False):
                    # Calculate and display basic stats
                    tot_count = df[col].count()
                    unique_count = df[col].nunique()
                    null_count = df[col].isnull().sum()
                    null_percentage = (null_count / len(df)) * 100
                    
                    st.write(f'**Total Entries:** {tot_count}')
                    st.write(f"**Unique Values:** {unique_count}")
                    st.write(f"**Null Values:** {null_count} ({null_percentage:.2f}%)")
                    
                    # Create and display visualizations
                    if unique_count <= 20:  # Only show for columns with reasonable unique values
                        stats_df, cat_fig = create_categorical_visualizations(df, col)
                        st.dataframe(stats_df)
                        st.plotly_chart(cat_fig, use_container_width=True)
                        st.download_button(
                            label="Download as HTML",
                            data=download_plotly_fig(cat_fig, f"categorical_plot_{col}"),
                            file_name=f"Categorical_plot_{col}.html",
                            mime="text/html"
                        )
                    else:
                        st.warning(f"Too many unique values ({unique_count}) to display effectively.")
                    
                    # Show value counts table for columns with moderate unique counts
                    if unique_count <= 50:
                        st.write("Value Counts:")
                        st.dataframe(df[col].value_counts().reset_index().rename(
                            columns={'index': 'Value', col: 'Count'}
                        ))
        else:
            st.warning("No categorical columns found in the dataset.")

    else:
        st.warning("Please upload an Excel/CSV file to proceed.")

# Bivariate Analysis Page
elif st.session_state.page == "Bivariate Analysis":
    st.title("Bivariate Analysis")

    # Initialize all session state variables
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'binned_columns' not in st.session_state:
        st.session_state.binned_columns = []
    if 'selected_cat_cols' not in st.session_state:
        st.session_state.selected_cat_cols = []
    if 'manual_cat_cols' not in st.session_state:
        st.session_state.manual_cat_cols = []
    if 'has_binning' not in st.session_state:
        st.session_state.has_binning = False

    # File uploader
    uploaded_file = st.file_uploader("Upload a dataset for Bivariate Analysis", 
                                   type=['csv','xlsx'], 
                                   accept_multiple_files=False, 
                                   key="bivariate_upload")

    if uploaded_file:
        # Only load the file if it hasn't been loaded already or a new file is uploaded
        if st.session_state.original_df is None:
            try:
                df = pd.read_excel(uploaded_file)
            except:
                df = pd.read_csv(uploaded_file)
            
            # Store original and working copies
            st.session_state.original_df = df.copy()
            st.session_state.df = df.copy()
        
        # Always use the session state dataframe
        df = st.session_state.df.copy()
        
        # Display dataset info
        st.write("Uploaded Dataset:")
        st.dataframe(df, hide_index=True)
        st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        
        # Find potential categorical columns
        potential_cat_cols = [col for col in df.columns 
                            if df[col].nunique() <= 10 
                            and (df[col].nunique()/df[col].count()) <= 0.1 
                            and df[col].dtype in ['int64', 'float64']]
        
        # Column conversion section
        st.header("Column Type Conversion")
        convert_tabs = st.tabs(["Auto-Detected Categorical", "Manual Conversion", "Binning"])
        
        with convert_tabs[0]:  # Auto-detected
            st.write("### Auto-Detected Potential Categorical Columns:")
            selected_cat_cols = []
            for col in potential_cat_cols:
                if st.checkbox(f"Convert '{col}' to categorical (nunique={df[col].nunique()})", 
                              key=f"auto_{col}"):
                    selected_cat_cols.append(col)
            
            st.session_state.selected_cat_cols = selected_cat_cols
        
        with convert_tabs[1]:  # Manual
            manual_cat_cols = st.multiselect(
                "Manually select columns to convert to categorical:",
                df.columns.tolist(),
                default=st.session_state.manual_cat_cols,
                key="manual_cat_selection"
            )
            
            st.session_state.manual_cat_cols = manual_cat_cols
            
            if manual_cat_cols:
                st.write("Current unique values:")
                for col in manual_cat_cols:
                    st.write(f"**{col}**: {df[col].nunique()} unique values")
        
        with convert_tabs[2]:  # Binning
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if numeric_cols:
                bin_column = st.selectbox("Select numeric column to bin:", numeric_cols, key="bin_column")
                
                # Add outlier handling options
                outlier_handling = st.radio(
                    "Outlier handling for binning:",
                    ["Keep all values", "Remove outliers before binning"],
                    key="outlier_handling"
                )
                
                # Add NaN handling options
                nan_handling = st.radio(
                    "NaN handling for binning:",
                    ["Include NaN values", "Exclude NaN values"],
                    key="nan_handling_binning"
                )
                
                # Create a temporary dataframe for preview and binning
                temp_df = df.copy()
                
                # Apply outlier handling if requested
                if outlier_handling == "Remove outliers before binning":
                    # Calculate IQR for outlier detection
                    q1 = temp_df[bin_column].quantile(0.25)
                    q3 = temp_df[bin_column].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Filter out outliers
                    outlier_count = len(temp_df[(temp_df[bin_column] < lower_bound) | (temp_df[bin_column] > upper_bound)])
                    temp_df = temp_df[(temp_df[bin_column] >= lower_bound) & (temp_df[bin_column] <= upper_bound)]
                    st.info(f"Removed {outlier_count} outliers for preview")
                
                # Apply NaN handling
                if nan_handling == "Exclude NaN values":
                    nan_count = temp_df[bin_column].isna().sum()
                    temp_df = temp_df.dropna(subset=[bin_column])
                    st.info(f"Excluded {nan_count} NaN values for preview")
                    
                bin_method = st.radio("Binning method:", ["Equal Width", "Equal Frequency", "Custom Edges"], key="bin_method")
                
                if bin_method in ["Equal Width", "Equal Frequency"]:
                    num_bins = st.slider("Number of bins:", 2, 20, 5, key="num_bins")
                    
                    # Create preview binning
                    try:
                        if bin_method == "Equal Width":
                            binned = pd.cut(temp_df[bin_column], bins=num_bins)
                        else:  # Equal Frequency
                            binned = pd.qcut(temp_df[bin_column], q=num_bins, duplicates='drop')
                        
                        # Format for display
                        binned_str = binned.astype(str)
                        binned_str = binned_str.str.replace(r'[\(\)\[\]]', '', regex=True)
                        binned_str = binned_str.str.replace(',', ' -')
                        
                        # Add to temp dataframe
                        temp_df['preview_binned'] = binned_str
                        
                        # Show distribution preview
                        st.subheader("Binning Preview")
                        fig = px.histogram(temp_df, x='preview_binned', title="Distribution of Binned Values")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show counts and statistics
                        bin_counts = temp_df['preview_binned'].value_counts().reset_index()
                        bin_counts.columns = ['Bin', 'Count']
                        bin_counts = bin_counts.sort_values('Bin')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Bin Counts:")
                            st.dataframe(bin_counts)
                        
                        with col2:
                            # Calculate statistics for each bin
                            bin_stats = temp_df.groupby('preview_binned')[bin_column].agg(['mean', 'min', 'max']).reset_index()
                            st.write("Bin Statistics:")
                            st.dataframe(bin_stats)
                    except Exception as e:
                        st.error(f"Error in preview binning: {str(e)}")
                
                else:  # Custom edges
                    st.write(f"Range: {df[bin_column].min():.2f} to {df[bin_column].max():.2f}")
                    custom_edges = st.text_input("Enter bin edges (comma-separated):", key="custom_edges")
                    
                    if custom_edges:
                        try:
                            edges = [float(x.strip()) for x in custom_edges.split(',')]
                            
                            # Create preview binning
                            binned = pd.cut(temp_df[bin_column], bins=edges)
                            
                            # Format for display
                            binned_str = binned.astype(str)
                            binned_str = binned_str.str.replace(r'[\(\)\[\]]', '', regex=True)
                            binned_str = binned_str.str.replace(',', ' -')
                            
                            # Add to temp dataframe
                            temp_df['preview_binned'] = binned_str
                            
                            # Show distribution preview
                            st.subheader("Binning Preview")
                            fig = px.histogram(temp_df, x='preview_binned', title="Distribution of Binned Values")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show counts and statistics
                            bin_counts = temp_df['preview_binned'].value_counts().reset_index()
                            bin_counts.columns = ['Bin', 'Count']
                            bin_counts = bin_counts.sort_values('Bin')
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Bin Counts:")
                                st.dataframe(bin_counts)
                            
                            with col2:
                                # Calculate statistics for each bin
                                bin_stats = temp_df.groupby('preview_binned')[bin_column].agg(['mean', 'min', 'max']).reset_index()
                                st.write("Bin Statistics:")
                                st.dataframe(bin_stats)
                        except Exception as e:
                            st.error(f"Error in preview binning: {str(e)}")
                
                bin_result_name = st.text_input("New column name:", f"{bin_column}_binned", key="bin_result_name")
                st.session_state.has_binning = bool(bin_column and bin_result_name)
                
                # Store binning settings in session state
                st.session_state.bin_options = {
                    'column': bin_column,
                    'method': bin_method,
                    'num_bins': num_bins if bin_method in ["Equal Width", "Equal Frequency"] else None,
                    'custom_edges': custom_edges if bin_method == "Custom Edges" else None,
                    'result_name': bin_result_name,
                    'outlier_handling': outlier_handling,
                    'nan_handling': nan_handling
                }
            else:
                st.warning("No numeric columns available for binning.")

        # Apply conversions button
        if st.button("✅ Apply All Conversions"):
            df = st.session_state.df.copy()
            changes_made = False
            
            # Apply categorical conversions
            all_cat_cols = st.session_state.selected_cat_cols + st.session_state.manual_cat_cols
            for col in all_cat_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')
                    changes_made = True
            
            # Apply binning
            if st.session_state.has_binning and hasattr(st.session_state, 'bin_options'):
                try:
                    opts = st.session_state.bin_options
                    bin_column = opts['column']
                    bin_result_name = opts['result_name']
                    
                    # Create a working copy for binning
                    temp_df = df.copy()
                    
                    # Apply outlier handling if requested
                    if opts['outlier_handling'] == "Remove outliers before binning":
                        # Calculate IQR for outlier detection
                        q1 = temp_df[bin_column].quantile(0.25)
                        q3 = temp_df[bin_column].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        # Filter out outliers
                        temp_df.loc[(temp_df[bin_column] < lower_bound) | (temp_df[bin_column] > upper_bound), bin_column] = None
                    
                    # Apply NaN handling
                    if opts['nan_handling'] == "Include NaN values":
                        # NaN values will be included in a separate category
                        pass  # Default behavior of pandas cut/qcut with NaN
                    
                    # Perform binning based on method
                    if opts['method'] == "Equal Width":
                        binned = pd.cut(temp_df[bin_column], bins=opts['num_bins'])
                    elif opts['method'] == "Equal Frequency":
                        binned = pd.qcut(temp_df[bin_column], q=opts['num_bins'], duplicates='drop')
                    else:  # Custom edges
                        edges = [float(x.strip()) for x in opts['custom_edges'].split(',')]
                        binned = pd.cut(temp_df[bin_column], bins=edges)
                    
                    # Clean and convert to categorical
                    df[bin_result_name] = binned.astype(str)
                    df[bin_result_name] = df[bin_result_name].str.replace(r'[\(\)\[\]]', '', regex=True)
                    df[bin_result_name] = df[bin_result_name].str.replace(',', ' -')
                    df[bin_result_name] = df[bin_result_name].astype('category')
                    
                    if bin_result_name not in st.session_state.binned_columns:
                        st.session_state.binned_columns.append(bin_result_name)
                    
                    changes_made = True
                except Exception as e:
                    st.error(f"Binning failed: {str(e)}")
            
            if changes_made:
                st.session_state.df = df.copy()
                st.success("Conversions applied successfully!")
                st.rerun()
            else:
                st.warning("No conversions were selected.")

        # Reset button
        if st.button("🔄 Reset All Changes"):
            if 'original_df' in st.session_state and st.session_state.original_df is not None:
                st.session_state.df = st.session_state.original_df.copy()
                st.session_state.binned_columns = []
                st.session_state.selected_cat_cols = []
                st.session_state.manual_cat_cols = []
                st.session_state.has_binning = False
                st.success("All changes have been reset to the original dataset!")
                st.rerun()
            else:
                st.warning("No original dataset found to reset to.")

        # Apply filters to the current dataframe
        filtered_df = apply_data_filters(df)
        
        # Get updated column lists with proper handling of converted columns
        # First, ensure proper types for our categorical columns
        for col in st.session_state.selected_cat_cols + st.session_state.manual_cat_cols:
            if col in filtered_df.columns:
                filtered_df[col] = filtered_df[col].astype('category')
        
        # Get updated column lists
        categorical_columns = list(set(
            filtered_df.select_dtypes(include=['object', 'category']).columns.tolist() +
            st.session_state.binned_columns
        ))
        
        numerical_columns = filtered_df.select_dtypes(include=['number']).columns.tolist()
        
        # Analysis section
        st.header("Data Analysis")
        
        if len(numerical_columns) >= 2:
            st.subheader("Correlation Matrix")
            corr_df = filtered_df[numerical_columns].corr()
            st.dataframe(corr_df.style.background_gradient(cmap='coolwarm'))

        analysis_type = st.radio("Analysis Type:", 
                               ["Continuous vs Continuous", "Categorical vs Continuous"],
                               key="analysis_type")

        if analysis_type == "Continuous vs Continuous":
            if len(numerical_columns) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis column", numerical_columns, key="x_cont")
                with col2:
                    y_col = st.selectbox("Y-axis column", numerical_columns, index=min(1, len(numerical_columns)-1), key="y_cont")
                
                # Outlier removal
                remove_x = st.checkbox(f"Remove outliers from {x_col}", key="remove_x")
                remove_y = st.checkbox(f"Remove outliers from {y_col}", key="remove_y")
                
                plot_df = filtered_df.copy()
                if remove_x:
                    plot_df = remove_outliers_from_column(plot_df, x_col)
                if remove_y:
                    plot_df = remove_outliers_from_column(plot_df, y_col)
                
                # Calculate correlation
                if len(plot_df) > 1:
                    # Drop NaN values from both columns together to ensure equal length
                    valid_data = plot_df[[x_col, y_col]].dropna()
                    
                    if len(valid_data) > 1:  # Check if we have enough data after dropping NaNs
                        corr, p_val = pearsonr(valid_data[x_col], valid_data[y_col])
                        st.write(f"**Pearson correlation:** {corr:.4f}")
                        st.write(f"**p-value:** {p_val:.4f}")
                    else:
                        st.warning("Not enough valid data points to calculate correlation after removing NaN values.")
                
                # Visualizations
                scatter_fig = px.scatter(plot_df, x=x_col, y=y_col, trendline="ols")
                st.plotly_chart(scatter_fig, use_container_width=True)
                st.download_button(
                    label="Download Scatter Plot",
                    data=download_plotly_fig(scatter_fig, "scatter_plot"),
                    file_name=f"Scatter_plot_{x_col}_vs_{y_col}.html",
                    mime="text/html"
                )
                
                heatmap_fig = px.density_heatmap(plot_df, x=x_col, y=y_col)
                st.plotly_chart(heatmap_fig, use_container_width=True)
                st.download_button(
                    label="Download Heatmap",
                    data=download_plotly_fig(heatmap_fig, "heatmap_plot"),
                    file_name=f"Heatmap_{x_col}_vs_{y_col}.html",
                    mime="text/html"
                )
            else:
                st.warning("Need at least 2 numerical columns for this analysis")
        
        else:  # Categorical vs Continuous
            if categorical_columns and numerical_columns:
                col1, col2 = st.columns(2)
                with col1:
                    cat_col = st.selectbox("Categorical column", categorical_columns, key="cat_col")
                with col2:
                    num_col = st.selectbox("Numerical column", numerical_columns, key="num_col")
                
                # Outlier removal and NaN handling options
                col1, col2 = st.columns(2)
                with col1:
                    remove_outliers = st.checkbox(f"Remove outliers from {num_col}", key="remove_outliers")
                with col2:
                    include_nan = st.checkbox(f"Include NaN values in {cat_col}", value=True, key="include_nan_cat")
                
                # Prepare data for plotting
                plot_df = filtered_df.copy()
                
                # Handle NaN values in categorical column
                if include_nan:
                    # Convert NaN values to a string representation for plotting
                    plot_df[cat_col] = plot_df[cat_col].fillna('NaN/Missing')
                    # If it was a category type, we need to add 'NaN/Missing' as a category
                    if plot_df[cat_col].dtype.name == 'category':
                        plot_df[cat_col] = plot_df[cat_col].cat.add_categories(['NaN/Missing'])
                else:
                    # Remove rows where categorical column is NaN
                    plot_df = plot_df.dropna(subset=[cat_col])
                
                # Apply outlier removal if selected
                if remove_outliers:
                    plot_df = remove_outliers_from_column(plot_df, num_col)
                
                # Show data info
                st.write(f"**Data after filtering:** {len(plot_df)} rows")
                nan_count = (filtered_df[cat_col].isna().sum())
                if nan_count > 0:
                    if include_nan:
                        st.write(f"**NaN values in {cat_col}:** {nan_count} (included as 'NaN/Missing')")
                    else:
                        st.write(f"**NaN values in {cat_col}:** {nan_count} (excluded from analysis)")
                
                # Analyze the number of categories
                cat_count = plot_df[cat_col].nunique()
                
                if cat_count > 30:
                    st.warning(f"The categorical column has {cat_count} unique values, which might make visualization cluttered.")
                
                # Show category distribution
                st.write(f"**Categories in {cat_col}:**")
                cat_counts = plot_df[cat_col].value_counts(dropna=False).reset_index()
                cat_counts.columns = ['Category', 'Count']
                st.dataframe(cat_counts)
                
                # Perform ANOVA test if there are at least 2 categories
                if cat_count >= 2:
                    # Group data for ANOVA - exclude 'NaN/Missing' category from statistical test
                    groups = []
                    for category in plot_df[cat_col].unique():
                        if category != 'NaN/Missing':  # Exclude NaN category from statistical test
                            group_data = plot_df[plot_df[cat_col] == category][num_col].dropna()
                            if len(group_data) > 0:
                                groups.append(group_data)
                    
                    if len(groups) >= 2 and all(len(group) > 0 for group in groups):
                        try:
                            f_stat, p_val = stats.f_oneway(*groups)
                            st.write(f"**ANOVA Test Results (excluding NaN/Missing category):**")
                            st.write(f"F-statistic: {f_stat:.4f}")
                            st.write(f"p-value: {p_val:.4f}")
                            
                            if p_val < 0.05:
                                st.write("**Result:** There are statistically significant differences between groups.")
                            else:
                                st.write("**Result:** No statistically significant differences detected between groups.")
                        except Exception as e:
                            st.warning(f"ANOVA test could not be performed: {str(e)}")
                
                # Visualizations
                try:
                    # Box plot with proper NaN handling
                    box_fig = px.box(plot_df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}")
                    
                    # Add mean markers - calculate means excluding NaN values in numerical column
                    means = plot_df.groupby(cat_col)[num_col].mean().reset_index()
                    box_fig.add_trace(go.Scatter(
                        x=means[cat_col],
                        y=means[num_col],
                        mode='markers',
                        marker=dict(color='red', symbol='x', size=10),
                        name='Mean'
                    ))
                    
                    # Rotate x-axis labels if there are many categories
                    if cat_count > 10:
                        box_fig.update_xaxes(tickangle=45)
                    
                    st.plotly_chart(box_fig, use_container_width=True)
                    st.download_button(
                        label="Download Box Plot",
                        data=download_plotly_fig(box_fig, "box_plot"),
                        file_name=f"Box_plot_{cat_col}_vs_{num_col}.html",
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"Error creating box plot: {str(e)}")
                
                try:
                    # Bar chart with std dev
                    agg_df = plot_df.groupby(cat_col).agg(
                        Mean=(num_col, 'mean'),
                        Std=(num_col, 'std'),
                        Count=(num_col, 'count')
                    ).reset_index()
                    
                    # Handle cases where std might be NaN (single value groups)
                    agg_df['Std'] = agg_df['Std'].fillna(0)
                    
                    # Add count info to hover text
                    bar_fig = px.bar(
                        agg_df, 
                        x=cat_col, 
                        y='Mean', 
                        error_y='Std',
                        title=f"Mean {num_col} by {cat_col} with Standard Deviation",
                        hover_data=['Count']
                    )

                    bar_fig.update_layout(yaxis_title=f"Mean - {num_col}")
                    
                    # Rotate x-axis labels if there are many categories
                    if cat_count > 10:
                        bar_fig.update_xaxes(tickangle=45)
                    
                    st.plotly_chart(bar_fig, use_container_width=True)
                    st.download_button(
                        label="Download Bar Chart",
                        data=download_plotly_fig(bar_fig, "bar_chart"),
                        file_name=f"Bar_chart_{cat_col}_vs_{num_col}.html",
                        mime="text/html"
                    )
                except Exception as e:
                    st.error(f"Error creating bar chart: {str(e)}")
                
                # Summary statistics table
                st.subheader(f"Summary Statistics for {num_col} by {cat_col}")
                try:
                    summary_stats = plot_df.groupby(cat_col)[num_col].agg([
                        'count', 'mean', 'std', 'min', 
                        lambda x: x.quantile(0.25), 
                        'median', 
                        lambda x: x.quantile(0.75), 
                        'max'
                    ]).reset_index()
                    summary_stats.columns = [cat_col, 'Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']
                    
                    # Round numerical columns for better display
                    numerical_cols = ['Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']
                    for col in numerical_cols:
                        summary_stats[col] = summary_stats[col].round(4)
                    
                    st.dataframe(summary_stats)
                except Exception as e:
                    st.error(f"Error creating summary statistics: {str(e)}")
                
            else:
                st.warning("Need both categorical and numerical columns for this analysis")
    else:
        st.warning("Please upload a dataset to begin analysis")