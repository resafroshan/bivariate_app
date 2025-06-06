import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import stats
from scipy.stats import pearsonr
import base64
import plotly.offline as pyo
from io import StringIO

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
    html = fig.to_html(
        full_html=True,           # Changed from False
        include_plotlyjs='cdn'    # Added this
    )
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
                # Numerical filter with both slider and manual input
                # Calculate range based on NaN handling
                if nan_handling == "Exclude rows with NaN values":
                    # Use range excluding NaN values
                    valid_data = df[column].dropna()
                    if len(valid_data) > 0:
                        min_val = float(valid_data.min())
                        max_val = float(valid_data.max())
                        range_note = " (excluding NaN)"
                    else:
                        min_val = 0.0
                        max_val = 1.0
                        range_note = " (no valid data)"
                elif nan_handling == "Only show rows with NaN values":
                    # For NaN-only mode, still show the full range for context but note it
                    min_val = float(df[column].min())
                    max_val = float(df[column].max())
                    range_note = " (NaN rows only - range for reference)"
                else:
                    # Include all values (default behavior)
                    min_val = float(df[column].min())
                    max_val = float(df[column].max())
                    range_note = ""
                
                step = (max_val - min_val) / 100 if (max_val - min_val) > 0 else 0.1
                
                st.write(f"**Filter '{column}' (Range: {min_val:.2f} to {max_val:.2f}{range_note}):**")
                
                # Manual input for range
                col1, col2 = st.columns(2)
                with col1:
                    min_input = st.number_input(
                        "Minimum value:",
                        value=min_val,
                        min_value=min_val,
                        max_value=max_val,
                        step=step,
                        key=f"min_input_{column}_{st.session_state.page}"
                    )
                with col2:
                    max_input = st.number_input(
                        "Maximum value:",
                        value=max_val,
                        min_value=min_val,
                        max_value=max_val,
                        step=step,
                        key=f"max_input_{column}_{st.session_state.page}"
                    )
                
                # Validate range
                if min_input > max_input:
                    st.error(f"⚠️ Minimum value ({min_input}) cannot be greater than maximum value ({max_input})")
                    selected_range = (min_val, max_val)  # Reset to default
                else:
                    selected_range = (min_input, max_input)
                
                filters[column] = selected_range
                
                # Add NaN checkbox for numerical columns
                include_nan_numeric = st.checkbox(
                    f"Include NaN/missing values for '{column}'",
                    value=nan_handling == "Include rows with NaN values" or nan_handling == "Only show rows with NaN values",
                    key=f"include_nan_{column}_{st.session_state.page}"
                )
                filters[f"{column}_include_nan"] = include_nan_numeric
                
                st.write("---")  # Add separator between numerical columns
        
        # Add reset filters button
        if st.button("Reset All Filters"):
            filters = {}
            filtered_df = df.copy()
    
    # Apply global NaN handling FIRST (regardless of column-specific filters)
    if nan_handling == "Exclude rows with NaN values":
        filtered_df = filtered_df.dropna()
    elif nan_handling == "Only show rows with NaN values":
        # Keep only rows with at least one NaN
        filtered_df = filtered_df[filtered_df.isna().any(axis=1)]
    
    # Apply column-specific filters
    if filters and columns_to_filter:
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

    # Show filter summary and filtered data
    original_rows = len(df)
    filtered_rows = len(filtered_df)

    # Check if any filtering was applied (global or column-specific)
    global_filter_applied = (nan_handling != "Include rows with NaN values")
    column_filters_applied = bool(filters and columns_to_filter)

    if global_filter_applied or column_filters_applied:
        # Build filter description
        filter_descriptions = []
        
        if global_filter_applied:
            if nan_handling == "Exclude rows with NaN values":
                filter_descriptions.append("Excluded rows with NaN values")
            elif nan_handling == "Only show rows with NaN values":
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

    # Initialize session state variables
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'page' not in st.session_state:
        st.session_state.page = 'univariate'

# File uploader
    uploaded_file = st.file_uploader("Upload a dataset for Univariate Analysis", 
                                    type=['csv','xlsx'], 
                                    accept_multiple_files=False, 
                                    key="univariate_upload")

    if uploaded_file:
        # Check if this is a new file by comparing file names
        current_file_name = uploaded_file.name
        if ('current_file_name' not in st.session_state or 
            st.session_state.current_file_name != current_file_name or
            st.session_state.original_df is None):
            
            try:
                df = pd.read_excel(uploaded_file)
            except:
                df = pd.read_csv(uploaded_file)
            
            # Store original and working copies
            st.session_state.original_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.current_file_name = current_file_name
        
        # Always use the session state dataframe
        df = st.session_state.df.copy()
        
        # Display dataset info
        st.write("Uploaded Dataset:")
        st.dataframe(df, hide_index=True)
        st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        
        # Apply filters to the current dataframe
        filtered_df = apply_data_filters(df)
        
        # Get column types
        numerical_columns = filtered_df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Analysis section
        st.header("Univariate Analysis")
        
        # Create tabs for different types of analysis
        if numerical_columns and categorical_columns:
            tab1, tab2 = st.tabs(["📊 Numerical Analysis", "📋 Categorical Analysis"])
        elif numerical_columns:
            tab1 = st.tabs(["📊 Numerical Analysis"])[0]
        elif categorical_columns:
            tab2 = st.tabs(["📋 Categorical Analysis"])[0]
        else:
            st.warning("No numerical or categorical columns found for analysis.")
            tab1, tab2 = None, None
        
        # Numerical Analysis Tab
        if numerical_columns and 'tab1' in locals():
            with tab1:
                if len(numerical_columns) > 0:
                    # Column selection
                    selected_columns = st.multiselect(
                        "Select numerical columns for analysis:",
                        numerical_columns,
                        default=numerical_columns[:3] if len(numerical_columns) >= 3 else numerical_columns,
                        key="numerical_columns_select"
                    )
                    
                    if selected_columns:
                        # Outlier removal option
                        remove_outliers = st.checkbox(
                            "Remove outliers for visualization and statistics",
                            value=False,
                            key="remove_outliers_numerical"
                        )
                        
                        # Prepare data for analysis
                        analysis_df = filtered_df[selected_columns].copy()
                        
                        # Apply outlier removal if selected
                        if remove_outliers:
                            original_count = len(analysis_df)
                            for col in selected_columns:
                                analysis_df = remove_outliers_from_column(analysis_df, col)
                            outliers_removed = original_count - len(analysis_df)
                            if outliers_removed > 0:
                                st.info(f"Removed {outliers_removed} rows containing outliers. Analyzing {len(analysis_df)} rows.")
                        
                        # Calculate and display statistics
                        st.subheader("📈 Descriptive Statistics")
                        if len(analysis_df) > 0:
                            stats_df = calculate_statistics(analysis_df[selected_columns])
                            st.dataframe(stats_df.style.format("{:.4f}"))
                            
                            # Download statistics
                            csv = stats_df.to_csv()
                            st.download_button(
                                label="📥 Download Statistics as CSV",
                                data=csv,
                                file_name="descriptive_statistics.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No data remaining after outlier removal.")
                        
                        # Visualizations
                        st.subheader("📊 Visualizations")
                        
                        # Box plots
                        if len(analysis_df) > 0:
                            st.write("**Box Plots:**")
                            box_fig = create_box_plot(analysis_df, selected_columns)
                            st.plotly_chart(box_fig, use_container_width=True)
                            
                            # Download box plot
                            st.download_button(
                                label="📥 Download Box Plot",
                                data=download_plotly_fig(box_fig, "box_plot"),
                                file_name="box_plots.html",
                                mime="text/html"
                            )
                        
                        # Individual histograms for each selected column
                        if len(analysis_df) > 0:
                            st.write("**Distribution Plots:**")
                            for col in selected_columns:
                                if analysis_df[col].dropna().nunique() > 1:  # Check if there's variation in the data
                                    hist_fig = create_histogram(analysis_df, col)
                                    st.plotly_chart(hist_fig, use_container_width=True)
                                    
                                    # Download histogram
                                    st.download_button(
                                        label=f"📥 Download {col} Distribution Plot",
                                        data=download_plotly_fig(hist_fig, f"histogram_{col}"),
                                        file_name=f"histogram_{col}.html",
                                        mime="text/html",
                                        key=f"download_hist_{col}"
                                    )
                                else:
                                    st.warning(f"Column '{col}' has insufficient variation for distribution plot.")
                    else:
                        st.info("Please select at least one numerical column for analysis.")
                else:
                    st.warning("No numerical columns available for analysis.")
        
        # Categorical Analysis Tab
        if categorical_columns and 'tab2' in locals():
            with tab2:
                if len(categorical_columns) > 0:
                    # Single column selection for categorical analysis
                    selected_cat_column = st.selectbox(
                        "Select a categorical column for analysis:",
                        categorical_columns,
                        key="categorical_column_select"
                    )
                    
                    if selected_cat_column:
                        # Prepare data for analysis
                        analysis_df = filtered_df.copy()
                        
                        # Handle NaN values option
                        nan_handling = st.radio(
                            f"How to handle NaN/missing values in '{selected_cat_column}':",
                            ["Include as 'Missing'", "Exclude from analysis"],
                            key="cat_nan_handling"
                        )
                        
                        if nan_handling == "Include as 'Missing'":
                            analysis_df[selected_cat_column] = analysis_df[selected_cat_column].fillna('Missing')
                        else:
                            analysis_df = analysis_df.dropna(subset=[selected_cat_column])
                        
                        # Check if there's data left for analysis
                        if len(analysis_df) > 0:
                            # Calculate and display statistics
                            st.subheader("📋 Categorical Statistics")
                            stats_df, bar_fig = create_categorical_visualizations(analysis_df, selected_cat_column)
                            
                            # Display statistics table
                            st.dataframe(stats_df)
                            
                            # Download statistics
                            csv = stats_df.to_csv()
                            st.download_button(
                                label="📥 Download Categorical Statistics as CSV",
                                data=csv,
                                file_name=f"categorical_stats_{selected_cat_column}.csv",
                                mime="text/csv",
                                key="download_cat_stats"
                            )
                            
                            # Visualizations
                            st.subheader("📊 Visualizations")
                            
                            # Bar chart
                            st.write("**Value Counts Bar Chart:**")
                            st.plotly_chart(bar_fig, use_container_width=True)
                            
                            # Download bar chart
                            st.download_button(
                                label="📥 Download Bar Chart",
                                data=download_plotly_fig(bar_fig, f"bar_chart_{selected_cat_column}"),
                                file_name=f"bar_chart_{selected_cat_column}.html",
                                mime="text/html",
                                key="download_bar_chart"
                            )
                            
                            # Pie chart (if not too many categories)
                            unique_values = analysis_df[selected_cat_column].nunique()
                            if unique_values <= 10:
                                st.write("**Pie Chart:**")
                                pie_fig = px.pie(
                                    stats_df.reset_index(),
                                    values='Count',
                                    names='index',
                                    title=f"Distribution of {selected_cat_column}"
                                )
                                pie_fig.update_traces(textposition='inside', textinfo='percent+label')
                                st.plotly_chart(pie_fig, use_container_width=True)
                                
                                # Download pie chart
                                st.download_button(
                                    label="📥 Download Pie Chart",
                                    data=download_plotly_fig(pie_fig, f"pie_chart_{selected_cat_column}"),
                                    file_name=f"pie_chart_{selected_cat_column}.html",
                                    mime="text/html",
                                    key="download_pie_chart"
                                )
                            else:
                                st.info(f"Pie chart not displayed due to too many categories ({unique_values}). Consider using filters to reduce categories.")
                            
                            # Additional insights
                            st.subheader("📝 Insights")
                            total_count = len(analysis_df)
                            most_common = stats_df.index[0]
                            most_common_pct = stats_df.loc[most_common, 'Percentage (%)']
                            
                            st.write(f"• **Total observations:** {total_count}")
                            st.write(f"• **Number of unique categories:** {unique_values}")
                            st.write(f"• **Most frequent category:** '{most_common}' ({most_common_pct}%)")
                            
                            if unique_values > 1:
                                least_common = stats_df.index[-1]
                                least_common_pct = stats_df.loc[least_common, 'Percentage (%)']
                                st.write(f"• **Least frequent category:** '{least_common}' ({least_common_pct}%)")
                            
                            # Check for highly imbalanced data
                            if most_common_pct > 90:
                                st.warning(f"⚠️ Data is highly imbalanced: '{most_common}' represents {most_common_pct}% of all observations.")
                            elif most_common_pct > 70:
                                st.info(f"ℹ️ Data shows some imbalance: '{most_common}' represents {most_common_pct}% of all observations.")
                        
                        else:
                            st.warning("No data available for analysis after handling missing values.")
                    else:
                        st.info("Please select a categorical column for analysis.")
                else:
                    st.warning("No categorical columns available for analysis.")

    else:
        st.warning("Please upload a dataset to begin analysis")

# Bivariate Analysis Page
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
        # Check if this is a new file by comparing file names
        current_file_name = uploaded_file.name
        if ('current_file_name_biv' not in st.session_state or 
            st.session_state.current_file_name_biv != current_file_name or
            st.session_state.original_df is None):
            
            try:
                df = pd.read_excel(uploaded_file)
            except:
                df = pd.read_csv(uploaded_file)
            
            # Store original and working copies
            st.session_state.original_df = df.copy()
            st.session_state.df = df.copy()
            st.session_state.current_file_name_biv = current_file_name
            
            # Reset conversion-related session state when new file is loaded
            st.session_state.binned_columns = []
            st.session_state.selected_cat_cols = []
            st.session_state.manual_cat_cols = []
            st.session_state.has_binning = False
        
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
                df.columns.to_list(),
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
                        
                        # Create ordered categories for preview
                        unique_categories = binned_str[binned_str != 'nan'].unique()
                        
                        def extract_lower_bound(interval_str):
                            try:
                                if interval_str == 'nan':
                                    return float('inf')
                                lower_bound = float(interval_str.split(' - ')[0])
                                return lower_bound
                            except:
                                return float('inf')
                        
                        sorted_categories = sorted(unique_categories, key=extract_lower_bound)
                        if 'nan' in binned_str.values:
                            sorted_categories.append('nan')
                        
                        # Add to temp dataframe with ordered categories
                        temp_df['preview_binned'] = pd.Categorical(binned_str, categories=sorted_categories, ordered=True)
                        
                        # Show distribution preview
                        st.subheader("Binning Preview")
                        fig = px.histogram(temp_df, x='preview_binned', title="Distribution of Binned Values", 
                                          category_orders={'preview_binned': sorted_categories})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show counts and statistics
                        bin_counts = temp_df['preview_binned'].value_counts().reset_index()
                        bin_counts.columns = ['Bin', 'Count']
                        # Sort by category order, not alphabetically
                        bin_counts['Bin'] = pd.Categorical(bin_counts['Bin'], categories=sorted_categories, ordered=True)
                        bin_counts = bin_counts.sort_values('Bin')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Bin Counts:")
                            st.dataframe(bin_counts)
                        
                        with col2:
                            # Calculate statistics for each bin
                            bin_stats = temp_df.groupby('preview_binned', observed=True)[bin_column].agg(['mean', 'min', 'max']).reset_index()
                            st.write("Bin Statistics:")
                            st.dataframe(bin_stats)
                    except Exception as e:
                        st.error(f"Error in preview binning: {str(e)}")
                
                else:  # Custom edges
                    st.write(f"Range: {temp_df[bin_column].min():.2f} to {temp_df[bin_column].max():.2f}")
                    custom_edges = st.text_input("Enter bin edges (comma-separated):", key="custom_edges")
                    
                    if custom_edges:
                        try:
                            edges = [float(x.strip()) for x in custom_edges.split(',')]
                            
                            # Create preview binning
                            binned = pd.cut(temp_df[bin_column], bins=edges, include_lowest=True)
                            
                            # Format for display
                            binned_str = binned.astype(str)
                            binned_str = binned_str.str.replace(r'[\(\)\[\]]', '', regex=True)
                            binned_str = binned_str.str.replace(',', ' -')
                            
                            # Create ordered categories for preview
                            unique_categories = binned_str[binned_str != 'nan'].unique()
                            
                            def extract_lower_bound(interval_str):
                                try:
                                    if interval_str == 'nan':
                                        return float('inf')
                                    lower_bound = float(interval_str.split(' - ')[0])
                                    return lower_bound
                                except:
                                    return float('inf')
                            
                            sorted_categories = sorted(unique_categories, key=extract_lower_bound)
                            if 'nan' in binned_str.values:
                                sorted_categories.append('nan')
                            
                            # Add to temp dataframe with ordered categories
                            temp_df['preview_binned'] = pd.Categorical(binned_str, categories=sorted_categories, ordered=True)
                            
                            # Show distribution preview
                            st.subheader("Binning Preview")
                            fig = px.histogram(temp_df, x='preview_binned', title="Distribution of Binned Values",
                                              category_orders={'preview_binned': sorted_categories})
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show counts and statistics
                            bin_counts = temp_df['preview_binned'].value_counts().reset_index()
                            bin_counts.columns = ['Bin', 'Count']
                            # Sort by category order, not alphabetically
                            bin_counts['Bin'] = pd.Categorical(bin_counts['Bin'], categories=sorted_categories, ordered=True)
                            bin_counts = bin_counts.sort_values('Bin')
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Bin Counts:")
                                st.dataframe(bin_counts)
                            
                            with col2:
                                # Calculate statistics for each bin
                                bin_stats = temp_df.groupby('preview_binned', observed=True)[bin_column].agg(['mean', 'min', 'max']).reset_index()
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
            
            # Apply binning with ordered categories
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
                        binned = pd.cut(temp_df[bin_column], bins=edges, include_lowest=True)
                    
                    # Clean and convert to string first
                    binned_str = binned.astype(str)
                    binned_str = binned_str.str.replace(r'[\(\)\[\]]', '', regex=True)
                    binned_str = binned_str.str.replace(',', ' -')
                    
                    # Create ordered categories by sorting the unique interval strings
                    # Get unique non-null categories
                    unique_categories = binned_str[binned_str != 'nan'].unique()
                    
                    # Sort categories by their lower bound
                    def extract_lower_bound(interval_str):
                        try:
                            if interval_str == 'nan':
                                return float('inf')  # Put NaN at the end
                            # Extract the first number (lower bound) from strings like "1.0 - 2.0"
                            lower_bound = float(interval_str.split(' - ')[0])
                            return lower_bound
                        except:
                            return float('inf')  # If parsing fails, put at the end
                    
                    # Sort categories by lower bound
                    sorted_categories = sorted(unique_categories, key=extract_lower_bound)
                    
                    # Add 'nan' category at the end if it exists
                    if 'nan' in binned_str.values:
                        sorted_categories.append('nan')
                    
                    # Convert to ordered categorical
                    df[bin_result_name] = pd.Categorical(binned_str, categories=sorted_categories, ordered=True)
                    
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
                if x_col != y_col:
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
                        label="📥 Download Scatter Plot",
                        data=download_plotly_fig(scatter_fig, "scatter_plot"),
                        file_name=f"Scatter_plot_{x_col}_vs_{y_col}.html",
                        mime="text/html"
                    )
                    
                    heatmap_fig = px.density_heatmap(plot_df, x=x_col, y=y_col)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                    st.download_button(
                        label="📥 Download Heatmap",
                        data=download_plotly_fig(heatmap_fig, "heatmap_plot"),
                        file_name=f"Heatmap_{x_col}_vs_{y_col}.html",
                        mime="text/html"
                    )
                else:
                    st.warning("Please select two different columns!")

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
                    # Check if there are any existing NaN-related categories
                    existing_categories = plot_df[cat_col].cat.categories if plot_df[cat_col].dtype.name == 'category' else []
                    nan_categories = [cat for cat in existing_categories if str(cat).lower() in ['nan', 'missing', 'null', 'none']]
                    
                    if plot_df[cat_col].dtype.name == 'category':
                        if len(nan_categories) == 0:  # No existing NaN categories
                            # Add 'NaN/Missing' as a category if it doesn't exist
                            if 'NaN/Missing' not in plot_df[cat_col].cat.categories:
                                plot_df[cat_col] = plot_df[cat_col].cat.add_categories(['NaN/Missing'])
                            # Fill NaN values
                            plot_df[cat_col] = plot_df[cat_col].fillna('NaN/Missing')
                        else:
                            st.warning("NaN already exists as a category")
                        # If NaN categories already exist, actual NaN values will show up as NaN in the plot
                    else:
                        # For non-categorical columns, direct fillna works
                        plot_df[cat_col] = plot_df[cat_col].fillna('NaN/Missing')
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
                    # Box plot with count annotations near each box
                    # col1, col2 = st.columns([4, 1])

                    # with col2:
                    #     reset_clicked = st.button("Reset", key='reset_box_title')

                    # with col1:
                    #     if reset_clicked:
                    #         box_title = None
                    #     box_title = st.text_input("Enter a custom title for Box Plot (optional):")

                    if 'box_title' not in st.session_state:
                        st.session_state.box_title = ''

                    box_title = st.text_input("Enter a custom title for Box Plot (optional):", value=st.session_state.box_title)
                    st.warning("Clear the field and press enter to reset")
                    st.session_state.box_title = box_title  
                        
                    # Calculate counts for each category
                    count_data = plot_df.groupby(cat_col)[num_col].count().reset_index()
                    count_data.columns = [cat_col, 'Count']
                    
                    # Create category order for proper display of binned columns
                    category_order = None
                    if cat_col in st.session_state.binned_columns:
                        # For binned columns, maintain the order
                        if plot_df[cat_col].dtype.name == 'category' and hasattr(plot_df[cat_col].dtype, 'ordered') and plot_df[cat_col].dtype.ordered:
                            category_order = plot_df[cat_col].cat.categories.tolist()

                    # Create box plot
                    if box_title:
                        box_fig = px.box(plot_df, x=cat_col, y=num_col, title=box_title,
                                        category_orders={cat_col: category_order} if category_order else None)
                    else:
                        box_fig = px.box(plot_df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}",
                                        category_orders={cat_col: category_order} if category_order else None)

                    # Add mean markers
                    means = plot_df.groupby(cat_col)[num_col].mean().reset_index()
                    box_fig.add_trace(go.Scatter(
                        x=means[cat_col],
                        y=means[num_col],
                        mode='markers',
                        marker=dict(color='red', symbol='x', size=10),
                        name='Mean'
                    ))
                    
                    # Add count annotations near each box
                    # Calculate the y-position for annotations (slightly above the maximum value of each box)
                    for i, (category, count) in enumerate(zip(count_data[cat_col], count_data['Count'])):
                        # Get the maximum value for this category to position annotation above it
                        category_data = plot_df[plot_df[cat_col] == category][num_col].dropna()
                        if len(category_data) > 0:
                            max_val = category_data.max()
                            # Add some padding above the max value
                            y_position = max_val + (plot_df[num_col].max() - plot_df[num_col].min()) * 0.02
                            
                            box_fig.add_annotation(
                                x=category,
                                y=y_position,
                                text=f"n={count}",
                                showarrow=False,
                                font=dict(size=10, color="black"),
                                bgcolor="rgba(255,255,255,0.8)",
                                bordercolor="gray",
                                borderwidth=1
                            )
                    
                    # Rotate x-axis labels if there are many categories
                    if cat_count > 10:
                        box_fig.update_xaxes(tickangle=45)
                    
                    st.plotly_chart(box_fig, use_container_width=True)
                    st.download_button(
                        label="📥 Download Box Plot",
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
                    if 'bar_title' not in st.session_state:
                        st.session_state.bar_title = ''

                    # Layout: text input and reset button in one row
                    # col1, col2 = st.columns([4, 1])
                    # with col1:
                    #     bar_title = st.text_input("Enter a custom title for Bar graph (optional):", value=st.session_state.bar_title)
                    #     st.warning("Clear the field and press enter to reset")
                    #     st.session_state.bar_title = bar_title
                    # with col2:
                    #     if st.button('Reset', key='reset_bar_title'):
                    #         st.session_state.bar_title = ''

                    bar_title = st.text_input("Enter a custom title for Bar graph (optional):", value=st.session_state.bar_title)
                    st.warning("Clear the field and press enter to reset")
                    st.session_state.bar_title = bar_title                  

                    # Create bar chart
                    if bar_title:
                        bar_fig = px.bar(
                            agg_df, 
                            x=cat_col, 
                            y='Mean', 
                            error_y='Std',
                            title=bar_title,
                            hover_data=['Count']
                        )
                    else:
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
                        label="📥 Download Bar Chart",
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
