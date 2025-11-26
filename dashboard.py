import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


try:
    from config import CSV_FILE_PATH, TOXICITY_THRESHOLD_SECONDS
    DEFAULT_FILE_PATH = CSV_FILE_PATH
except ImportError:
    DEFAULT_FILE_PATH = "AugSept.csv"
    TOXICITY_THRESHOLD_SECONDS = 60

# Page config
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data(file_path):
    """Load and process trading data"""
    trades = pd.read_csv(file_path)
    
    # Process datetime
    trades['Transfer Date'] = pd.to_datetime(trades['Transfer Date'], format='%m/%d/%y %H:%M')
    trades['Trading Day'] = trades['Transfer Date'].dt.date
    trades['Trading Time'] = trades['Transfer Date'].dt.time
    trades['Trading Hour'] = trades['Transfer Date'].dt.hour
    trades['Day of Week'] = trades['Transfer Date'].dt.day_name()
    trades['Trading Week'] = trades['Transfer Date'].dt.isocalendar().week
    trades['Trading Month'] = trades['Transfer Date'].dt.month
    trades['Calendar Month'] = trades['Transfer Date'].dt.month_name()
    
    # Clean amounts and P/L
    trades['Trade Amount'] = trades['Trade Amount'].str.replace(',', '').astype(float)
    trades['Settled PL'] = trades['Settled PL'].replace('—', np.nan)
    trades['Settled PL'] = trades['Settled PL'].astype(str).str.replace(',','')
    trades['Settled PL'] = pd.to_numeric(trades['Settled PL'], errors='coerce')
    
    # Calculate direction and signed amounts
    trades['Trade Direction'] = trades['Buy/Sell'].map({'Buy': 1, 'Sell': -1})
    trades['Signed Trade Amount'] = trades['Trade Amount'] * trades['Trade Direction']
    trades['Is_settled'] = trades['Settled PL'].notna()
    
    # Sort by date
    trades = trades.sort_values(by='Transfer Date').reset_index(drop=True)
    
    return trades

def calculate_cumulative_position(df, instrument):
    """Calculate cumulative position for an instrument"""
    df_inst = df[df['Instrument'] == instrument].copy()
    df_inst['Cumulative Position'] = df_inst['Signed Trade Amount'].cumsum()
    return df_inst

def fifo_matching(df_inst):
    """FIFO matching for trades"""
    open_buys = deque()
    open_sells = deque()
    matched_pairs = []
    
    for idx, row in df_inst.iterrows():
        trade_type = row['Buy/Sell']
        amount = row['Trade Amount']
        price = row['Trade Price']
        timestamp = row['Transfer Date']
        order_id = row['Order ID']
        settled_pl = row['Settled PL']
        
        if trade_type == 'Buy':
            if len(open_sells) > 0:
                sell_trade = open_sells.popleft()
                entry_time = sell_trade['timestamp']
                exit_time = timestamp
                holding_time = (exit_time - entry_time).total_seconds()
                pl = (sell_trade['price'] - price) * amount
                
                matched_pairs.append({
                    'Entry_order_ID': sell_trade['order_id'],
                    'Exit_order_ID': order_id,
                    'Position_Type': 'Short',
                    'Entry_Time': entry_time,
                    'Exit_Time': exit_time,
                    'Holding_Time_Seconds': holding_time,
                    'Entry_Price': sell_trade['price'],
                    'Exit_Price': price,
                    'Amount': amount,
                    'Settled_PL': settled_pl if pd.notna(settled_pl) else pl
                })
            else:
                open_buys.append({
                    'order_id': order_id,
                    'price': price,
                    'amount': amount,
                    'timestamp': timestamp
                })
        else:  # Sell
            if len(open_buys) > 0:
                buy_trade = open_buys.popleft()
                entry_time = buy_trade['timestamp']
                exit_time = timestamp
                holding_time = (exit_time - entry_time).total_seconds()
                pl = (price - buy_trade['price']) * amount
                
                matched_pairs.append({
                    'Entry_order_ID': buy_trade['order_id'],
                    'Exit_order_ID': order_id,
                    'Position_Type': 'Long',
                    'Entry_Time': entry_time,
                    'Exit_Time': exit_time,
                    'Holding_Time_Seconds': holding_time,
                    'Entry_Price': buy_trade['price'],
                    'Exit_Price': price,
                    'Amount': amount,
                    'Settled_PL': settled_pl if pd.notna(settled_pl) else pl
                })
            else:
                open_sells.append({
                    'order_id': order_id,
                    'price': price,
                    'amount': amount,
                    'timestamp': timestamp
                })
    
    if len(matched_pairs) > 0:
        matched_df = pd.DataFrame(matched_pairs)
        matched_df['Holding_Time_Minutes'] = matched_df['Holding_Time_Seconds'] / 60
        matched_df['Holding_Time_Hours'] = matched_df['Holding_Time_Seconds'] / 3600
        matched_df['Holding_Time_Days'] = matched_df['Holding_Time_Seconds'] / 86400
        matched_df['Is_Toxic'] = matched_df['Holding_Time_Seconds'] <= TOXICITY_THRESHOLD_SECONDS
        return matched_df, len(open_buys), len(open_sells)
    
    return pd.DataFrame(), len(open_buys), len(open_sells)

def main():
    st.markdown('<div class="main-header">Data Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
  
    
    # Default file path from config file
    file_path = DEFAULT_FILE_PATH
    # file_path = st.sidebar.text_input(
    #     "CSV File Path",
    #     value=DEFAULT_FILE_PATH,
    #     help="Enter the path to your trading data CSV file"
    # )

    
    # Add reload button
    reload_data = st.sidebar.button("Reload Data", help="Click to reload data from the file")
    
    # Check if file exists
    import os
    if not os.path.exists(file_path):
        st.error(f"File not found: `{file_path}`")
        st.info("Please enter a valid file path in the sidebar")
        st.markdown("""
        
        ### Example file paths:
        - Windows: `C:/Users/YourName/Documents/trading.csv`
        - Mac/Linux: `/home/user/data/trading.csv`
        - Relative: `data/trading.csv` or just `trading.csv`
        - Current directory: Just the filename like `AugSept copy.csv`
        """)
        return
    
    # Load data
    try:
        # Clear cache if reload button is clicked
        if reload_data:
            st.cache_data.clear()
            # st.sidebar.success("Cache cleared, reloading...")
        
        trades = load_and_process_data(file_path)
        
        # Show file info
        file_size = os.path.getsize(file_path) / 1024  
        file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        
    except Exception as e:
        st.error(f" Error loading file: {str(e)}")
        st.info("Please check that your CSV file format matches the expected structure")
        
        # Show detailed error for debugging
        with st.expander("Show detailed error"):
            st.code(str(e))
        return
    
    # Get instruments
    instruments = trades['Instrument'].unique()
    
    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Overview")
    st.sidebar.metric("Total Trades", f"{len(trades):,}")
    st.sidebar.metric("Date Range", f"{trades['Trading Day'].min()} to {trades['Trading Day'].max()}")
    st.sidebar.metric("Trading Days", f"{trades['Trading Day'].nunique()}")
    
    selected_instruments = st.sidebar.multiselect(
        "Select Instruments",
        options=list(instruments),
        default=list(instruments[:2])  # Default to first 2
    )
    
    if not selected_instruments:
        st.warning("Please select at least one instrument")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", 
        "Position & P/L", 
        "Toxicity Analysis", 
        "Holding Time", 
        "Detailed Metrics"
    ])
    
    # Process data for selected instruments
    filtered_trades = trades[trades['Instrument'].isin(selected_instruments)]
    
    # TAB 1: OVERVIEW
    with tab1:
        st.header("Trading Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_pl = filtered_trades['Settled PL'].sum()
        settled_count = filtered_trades['Is_settled'].sum()
        total_volume = filtered_trades['Trade Amount'].sum()
        
        with col1:
            st.metric("Total P/L", f"${total_pl:,.2f}", 
                     delta=None,
                     delta_color="normal" if total_pl >= 0 else "inverse")
        with col2:
            st.metric("Settled Trades", f"{settled_count:,}")
        with col3:
            st.metric("Total Volume", f"{total_volume:,.0f}")
        with col4:
            win_rate = (filtered_trades[filtered_trades['Settled PL'] > 0].shape[0] / 
                       settled_count * 100) if settled_count > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        st.markdown("---")
        
        # Trade distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Trades by instrument
            inst_counts = filtered_trades.groupby('Instrument').size().reset_index(name='Count')
            fig = px.pie(inst_counts, values='Count', names='Instrument',
                        title='Trade Distribution by Instrument',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Buy/Sell distribution
            buysell_counts = filtered_trades.groupby('Buy/Sell').size().reset_index(name='Count')
            fig = px.bar(buysell_counts, x='Buy/Sell', y='Count',
                        title='Buy vs Sell Trades',
                        color='Buy/Sell',
                        color_discrete_map={'Buy': '#28a745', 'Sell': '#dc3545'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Trading activity over time
        st.subheader("Trading Activity Over Time")
        daily_trades = filtered_trades.groupby(['Trading Day', 'Instrument']).size().reset_index(name='Trades')
        fig = px.line(daily_trades, x='Trading Day', y='Trades', color='Instrument',
                     title='Daily Trading Volume',
                     markers=True)
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: POSITION & P/L
    with tab2:
        st.header("Position & P/L Analysis")
        
        for instrument in selected_instruments:
            st.subheader(f"{instrument}")
            
            df_inst = calculate_cumulative_position(filtered_trades, instrument)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Cumulative Position Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_inst['Transfer Date'],
                    y=df_inst['Cumulative Position'],
                    mode='lines',
                    name='Position',
                    fill='tozeroy',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig.update_layout(
                    title=f'{instrument} - Cumulative Position',
                    xaxis_title='Date',
                    yaxis_title='Position',
                    hovermode='x unified'
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
                
                # Position metrics
                final_pos = df_inst['Cumulative Position'].iloc[-1]
                max_long = df_inst['Cumulative Position'].max()
                max_short = df_inst['Cumulative Position'].min()
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Final Position", f"{final_pos:,.0f}")
                col_b.metric("Max Long", f"{max_long:,.0f}")
                col_c.metric("Max Short", f"{max_short:,.0f}")
            
            with col2:
                # Cumulative P/L Chart
                df_inst['Cumulative_PL'] = df_inst['Settled PL'].fillna(0).cumsum()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_inst['Transfer Date'],
                    y=df_inst['Cumulative_PL'],
                    mode='lines',
                    name='P/L',
                    line=dict(color='#28a745' if df_inst['Cumulative_PL'].iloc[-1] > 0 else '#dc3545', width=2)
                ))
                fig.update_layout(
                    title=f'{instrument} - Cumulative P/L',
                    xaxis_title='Date',
                    yaxis_title='P/L ($)',
                    hovermode='x unified'
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
                
                # P/L metrics
                total_pl_inst = df_inst['Settled PL'].sum()
                avg_pl = df_inst['Settled PL'].mean()
                settled_inst = df_inst['Is_settled'].sum()
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total P/L", f"${total_pl_inst:,.2f}")
                col_b.metric("Avg P/L", f"${avg_pl:,.2f}")
                col_c.metric("Settled", f"{settled_inst:,}")
            
            st.markdown("---")
        
        # Weekly P/L comparison
        st.subheader("Weekly P/L Comparison")
        weekly_pl = filtered_trades.groupby(['Trading Week', 'Instrument'])['Settled PL'].sum().reset_index()
        weekly_pl = weekly_pl.pivot(index='Trading Week', columns='Instrument', values='Settled PL').fillna(0)
        
        fig = go.Figure()
        for col in weekly_pl.columns:
            fig.add_trace(go.Bar(
                x=[f"Week {i}" for i in weekly_pl.index],
                y=weekly_pl[col],
                name=col
            ))
        
        fig.update_layout(
            title='Weekly P/L by Instrument',
            xaxis_title='Week',
            yaxis_title='P/L ($)',
            barmode='group',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: TOXICITY ANALYSIS
    with tab3:
        st.header("Toxicity Analysis")
        st.markdown(f"""
        **Toxicity Definition**: Trades held for ≤{TOXICITY_THRESHOLD_SECONDS} seconds ({TOXICITY_THRESHOLD_SECONDS/60:.1f} minute{'s' if TOXICITY_THRESHOLD_SECONDS != 60 else ''}) are considered "toxic" 
        as they indicate very quick position flips that may signal adverse selection or latency arbitrage.
        """)
        
        toxicity_results = {}
        
        for instrument in selected_instruments:
            df_inst = filtered_trades[filtered_trades['Instrument'] == instrument].copy()
            matched_df, unmatched_buys, unmatched_sells = fifo_matching(df_inst)
            
            if len(matched_df) > 0:
                toxicity_results[instrument] = matched_df
        
        if toxicity_results:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            total_matched = sum(len(df) for df in toxicity_results.values())
            total_toxic = sum((df['Is_Toxic']).sum() for df in toxicity_results.values())
            toxic_pct = (total_toxic / total_matched * 100) if total_matched > 0 else 0
            
            with col1:
                st.metric("Total Matched Trades", f"{total_matched:,}")
            with col2:
                st.metric(f"Toxic Trades (≤{TOXICITY_THRESHOLD_SECONDS}s)", f"{total_toxic:,}", 
                         delta=f"{toxic_pct:.2f}%")
            with col3:
                toxic_pl = sum(df[df['Is_Toxic']]['Settled_PL'].sum() for df in toxicity_results.values())
                st.metric("Toxic P/L", f"${toxic_pl:,.2f}")
            
            st.markdown("---")
            
            # Toxicity by instrument
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie charts
                for instrument, matched_df in toxicity_results.items():
                    toxic_count = matched_df['Is_Toxic'].sum()
                    normal_count = (~matched_df['Is_Toxic']).sum()
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=[f'Toxic (≤{TOXICITY_THRESHOLD_SECONDS}s)', f'Normal (>{TOXICITY_THRESHOLD_SECONDS}s)'],
                        values=[toxic_count, normal_count],
                        marker_colors=['#dc3545', '#28a745'],
                        hole=0.4
                    )])
                    fig.update_layout(title=f'{instrument} Toxicity')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Scatter plot: Holding time vs Volume
                fig = go.Figure()
                
                for instrument, matched_df in toxicity_results.items():
                    fig.add_trace(go.Scatter(
                        x=matched_df['Holding_Time_Seconds'],
                        y=matched_df['Amount'],
                        mode='markers',
                        name=instrument,
                        marker=dict(
                            size=8,
                            color=matched_df['Is_Toxic'].map({True: 'red', False: 'blue'}),
                            opacity=0.6
                        ),
                        text=[f"P/L: ${pl:.2f}" for pl in matched_df['Settled_PL']],
                        hovertemplate='%{text}<br>Hold: %{x:.0f}s<br>Volume: %{y:,.0f}'
                    ))
                
                fig.add_vline(x=TOXICITY_THRESHOLD_SECONDS, line_dash="dash", line_color="red",
                             annotation_text=f"{TOXICITY_THRESHOLD_SECONDS}s Threshold")
                
                fig.update_layout(
                    title='Holding Time vs Trade Volume (Red = Toxic)',
                    xaxis_title='Holding Time (seconds)',
                    yaxis_title='Trade Volume',
                    xaxis_type='log',
                    hovermode='closest'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed toxicity table
            st.subheader("Detailed Toxicity Breakdown")
            
            toxicity_data = []
            for instrument, matched_df in toxicity_results.items():
                toxic = matched_df[matched_df['Is_Toxic']]
                normal = matched_df[~matched_df['Is_Toxic']]
                
                toxicity_data.append({
                    'Instrument': instrument,
                    'Total Matched': len(matched_df),
                    'Toxic Count': len(toxic),
                    'Toxic %': f"{len(toxic)/len(matched_df)*100:.2f}%",
                    'Toxic Volume': f"{toxic['Amount'].sum():,.0f}",
                    'Toxic P/L': f"${toxic['Settled_PL'].sum():,.2f}",
                    'Normal P/L': f"${normal['Settled_PL'].sum():,.2f}",
                    'Avg Toxic P/L': f"${toxic['Settled_PL'].mean():.2f}" if len(toxic) > 0 else "$0.00"
                })
            
            toxicity_df = pd.DataFrame(toxicity_data)
            st.dataframe(toxicity_df, use_container_width=True)
        else:
            st.warning("No matched trades available for toxicity analysis")
    
    # TAB 4: HOLDING TIME
    with tab4:
        st.header("Holding Time Analysis")
        
        for instrument in selected_instruments:
            df_inst = filtered_trades[filtered_trades['Instrument'] == instrument].copy()
            matched_df, _, _ = fifo_matching(df_inst)
            
            if len(matched_df) > 0:
                st.subheader(f"{instrument}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Statistics
                    st.markdown("**Holding Time Statistics**")
                    stats_df = pd.DataFrame({
                        'Metric': [
                            'Minimum',
                            'Maximum',
                            'Average',
                            'Median',
                            'Std Deviation',
                            '25th Percentile',
                            '75th Percentile'
                        ],
                        'Seconds': [
                            f"{matched_df['Holding_Time_Seconds'].min():.0f}",
                            f"{matched_df['Holding_Time_Seconds'].max():.0f}",
                            f"{matched_df['Holding_Time_Seconds'].mean():.0f}",
                            f"{matched_df['Holding_Time_Seconds'].median():.0f}",
                            f"{matched_df['Holding_Time_Seconds'].std():.0f}",
                            f"{matched_df['Holding_Time_Seconds'].quantile(0.25):.0f}",
                            f"{matched_df['Holding_Time_Seconds'].quantile(0.75):.0f}"
                        ],
                        'Hours': [
                            f"{matched_df['Holding_Time_Hours'].min():.2f}",
                            f"{matched_df['Holding_Time_Hours'].max():.2f}",
                            f"{matched_df['Holding_Time_Hours'].mean():.2f}",
                            f"{matched_df['Holding_Time_Hours'].median():.2f}",
                            f"{matched_df['Holding_Time_Hours'].std():.2f}",
                            f"{matched_df['Holding_Time_Hours'].quantile(0.25):.2f}",
                            f"{matched_df['Holding_Time_Hours'].quantile(0.75):.2f}"
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig = go.Figure()
                    fig.add_trace(go.Box(
                        y=matched_df['Holding_Time_Hours'],
                        name=instrument,
                        boxmean='sd',
                        marker_color='#1f77b4'
                    ))
                    fig.update_layout(
                        title=f'{instrument} - Holding Time Distribution',
                        yaxis_title='Holding Time (hours)',
                        yaxis_type='log'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Holding time categories
                st.markdown("**Holding Time Categories**")
                
                # Define buckets
                matched_df['Time_Category'] = pd.cut(
                    matched_df['Holding_Time_Seconds'],
                    bins=[0, TOXICITY_THRESHOLD_SECONDS, 300, 900, 3600, 86400, float('inf')],
                    labels=[f'≤{TOXICITY_THRESHOLD_SECONDS}s (Toxic)', '1-5 min', '5-15 min', '15-60 min', '1-24 hrs', '>24 hrs']
                )
                
                category_counts = matched_df['Time_Category'].value_counts().sort_index()
                category_pct = (category_counts / len(matched_df) * 100).round(2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    fig = px.bar(
                        x=category_counts.index,
                        y=category_counts.values,
                        labels={'x': 'Time Category', 'y': 'Trade Count'},
                        title=f'{instrument} - Trades by Holding Time',
                        color=category_counts.values,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Category table
                    category_df = pd.DataFrame({
                        'Category': category_counts.index,
                        'Count': category_counts.values,
                        'Percentage': [f"{p:.2f}%" for p in category_pct]
                    })
                    st.dataframe(category_df, use_container_width=True)
                
                # Holding time vs P/L
                st.markdown("**Holding Time vs P/L Correlation**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=matched_df['Holding_Time_Hours'],
                    y=matched_df['Settled_PL'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=matched_df['Settled_PL'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="P/L ($)")
                    ),
                    text=[f"Entry: {et}<br>Exit: {xt}<br>P/L: ${pl:.2f}" 
                          for et, xt, pl in zip(matched_df['Entry_Time'], 
                                               matched_df['Exit_Time'],
                                               matched_df['Settled_PL'])],
                    hovertemplate='%{text}<br>Hold: %{x:.2f} hrs'
                ))
                fig.update_layout(
                    title=f'{instrument} - Holding Time vs P/L',
                    xaxis_title='Holding Time (hours)',
                    yaxis_title='P/L ($)',
                    xaxis_type='log',
                    hovermode='closest'
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
    
    # TAB 5: DETAILED METRICS
    with tab5:
        st.header("Detailed Trading Metrics")
        
        # Overall performance summary
        st.subheader("Performance Summary")
        
        summary_data = []
        for instrument in selected_instruments:
            df_inst = filtered_trades[filtered_trades['Instrument'] == instrument].copy()
            settled = df_inst[df_inst['Is_settled']]
            
            profit_trades = settled[settled['Settled PL'] > 0]
            loss_trades = settled[settled['Settled PL'] < 0]
            
            summary_data.append({
                'Instrument': instrument,
                'Total Trades': len(df_inst),
                'Settled Trades': len(settled),
                'Win Rate': f"{len(profit_trades)/len(settled)*100:.2f}%" if len(settled) > 0 else "N/A",
                'Total P/L': f"${settled['Settled PL'].sum():,.2f}",
                'Avg P/L': f"${settled['Settled PL'].mean():.2f}" if len(settled) > 0 else "N/A",
                'Avg Win': f"${profit_trades['Settled PL'].mean():.2f}" if len(profit_trades) > 0 else "N/A",
                'Avg Loss': f"${loss_trades['Settled PL'].mean():.2f}" if len(loss_trades) > 0 else "N/A",
                'Largest Win': f"${profit_trades['Settled PL'].max():.2f}" if len(profit_trades) > 0 else "N/A",
                'Largest Loss': f"${loss_trades['Settled PL'].min():.2f}" if len(loss_trades) > 0 else "N/A"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        st.markdown("---")
        
        # Monthly breakdown
        st.subheader("Monthly Performance")
        monthly_data = filtered_trades.groupby(['Calendar Month', 'Instrument']).agg({
            'Settled PL': 'sum',
            'Order ID': 'count',
            'Trade Amount': 'sum'
        }).reset_index()
        monthly_data.columns = ['Month', 'Instrument', 'Total P/L', 'Trade Count', 'Total Volume']
        
        fig = px.bar(monthly_data, x='Month', y='Total P/L', color='Instrument',
                    title='Monthly P/L by Instrument',
                    barmode='group')
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Raw data viewer
        st.subheader("Raw Data Viewer")
        if st.checkbox("Show raw trading data"):
            st.dataframe(filtered_trades, use_container_width=True)
            
            # Download button
            csv = filtered_trades.to_csv(index=False)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name=f"trading_data_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()