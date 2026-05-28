import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import plotly.graph_objects as go
from prophet import Prophet
from scipy.stats import norm
import scipy.stats as stats
from plotly.subplots import make_subplots
import plotly.colors as pc

# --- Session State Initialization ---
if 'next_clicked' not in st.session_state:
    st.session_state.next_clicked = False
if 'seed_counter' not in st.session_state:
    st.session_state.seed_counter = 42

st.set_page_config(page_title="Supply Chain Analytics Platform", layout="wide")

st.title("🚀 Supply Chain Analytics Platform")

tab1, tab2, tab3 = st.tabs(["Average Demand", "📊 Demand Histogram", "Periodic Review Inventory Audit"])

with tab1:
    st.header("The Basic Thumb Rule Used For Inventory Planning")
    # st.markdown("""
    # **The Concept:** Demonstrating how static, average-based demand strategies systematically introduce internal sabotage. 
    # While an average looks clean over a 300-day window, daily variability will trigger stockouts during finite replenishment cycles.
    # """)
    
    # --- Step 1: Baseline Strategy Input Section ---
    col1, col2 = st.columns(2)
    
    with col1:
        annual_sales = st.number_input("Annual Sales (Units)", value=12000, step=500)
        working_days = st.number_input("Working Days per Year", value=300)
        
    with col2:
        # Calculate Average Daily Sales (ADS) baseline
        avg_daily_sales = annual_sales / working_days
        st.metric("Avg. Daily Sales (ADS)", f"{avg_daily_sales:.2f}")
        
        suggested_baseline = avg_daily_sales * 10
        requisite_inventory = st.number_input(
            "Enter Requisite Inventory Strategy Limit", 
            value=int(suggested_baseline),
            help="This is the target inventory volume you have allocated to cover your business lead time window."
        )

    # Trigger persistent UI view state
    if st.button("Next"):
        st.session_state.next_clicked = True

    # --- Step 2: Persisted Stress-Testing Environment ---
    if st.session_state.next_clicked:
        st.divider()
        st.subheader("🎯 Stress Test Parameters & Reality Simulator")
        
        # User Parameter Input Boxes
        c1, c2, c3 = st.columns(3)
        with c1:
            std_dev = st.number_input("Demand Standard Deviation (Volatility)", value=10, min_value=0)
        with c2:
            sim_days = st.number_input("Number of Simulation Days", value=100, min_value=1)
        with c3:
            rolling_window = st.number_input("Look-Forward Window (Days)", value=10, min_value=1, max_value=int(sim_days))

        # Action Buttons Layout: Regenerate Button
        btn_col1, btn_col2 = st.columns([1, 5])
        with btn_col1:
            if st.button("🔄 Regenerate Demand"):
                st.session_state.seed_counter += 1  # Shifts the seed to force a new layout run

        # Generate Volatile Demand Data Array
        np.random.seed(st.session_state.seed_counter)
        daily_demand = np.random.normal(avg_daily_sales, std_dev, sim_days)
        daily_demand = np.clip(daily_demand, 0, None).round(0)  # Prevents impossible negative demand days
        
        days = [f"Day {i+1}" for i in range(sim_days)]
        
        # --- Visual Asset 1: Daily Demand Timeline ---
        st.write("### 📈 Daily Demand Volatility")
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=days, y=daily_demand, mode='lines+markers', name='Daily Demand Actual',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_daily.add_hline(y=avg_daily_sales, line_dash="dash", line_color="gray", annotation_text="Calculated Static Average")
        fig_daily.update_layout(template="plotly_white", height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_daily, use_container_width=True)

        # Pre-calculating Data for Tables & Charts
        df_summary = pd.DataFrame({
            "Lead Time Day": days,
            "Daily Demand (Units)": daily_demand.astype(int)
        })
        
        # Look-Forward Core Mathematical Optimization Matrix
        forward_sums = df_summary["Daily Demand (Units)"].iloc[::-1].rolling(window=rolling_window).sum().iloc[::-1]
        df_summary[f"Demand Next {rolling_window} Days"] = forward_sums
        df_summary["Inventory Level Provided"] = int(requisite_inventory)
        
        # Metric Scorecard Data Compilation
        valid_forward_days = forward_sums.dropna()
        total_valid_days = len(valid_forward_days)
        deficits_series = valid_forward_days > requisite_inventory
        total_deficits = deficits_series.sum()
        pct_deficits = (total_deficits / total_valid_days * 100) if total_valid_days > 0 else 0.0
        
        # Calculate Maximum Forward Window Value
        max_window_demand = valid_forward_days.max() if total_valid_days > 0 else 0.0

        # --- Visual Asset 2: Collapsible Diagnostic Data Table & Scorecard ---
        with st.expander("📋 Generated Demand Data Table", expanded=False):
            st.markdown("### 📊 Window Analysis Summary")
            
            # Expanded layout matrix (changed to 4 columns to fit the new metric)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Days with Valid Window", f"{total_valid_days} Days")
            with m2:
                # Calculate absolute peak gap to show if the strategy safely absorbed it
                peak_gap = int(max_window_demand - requisite_inventory)
                st.metric(
                    "Max Window Demand Peak", 
                    f"{int(max_window_demand)} Units",
                    delta=f"+{peak_gap} Over Limit" if peak_gap > 0 else f"{peak_gap} Under Limit",
                    delta_color="inverse" if peak_gap > 0 else "normal"
                )
            with m3:
                st.metric("Total Deficit Occurrences", f"{total_deficits} Days", 
                          delta=f"-{total_deficits} Stockouts" if total_deficits > 0 else None, 
                          delta_color="inverse" if total_deficits > 0 else "normal")
            with m4:
                st.metric("Deficit Risk Rate (%)", f"{pct_deficits:.1f}%",
                          delta="CRITICAL RISK" if pct_deficits > 30 else "STABLE BUFFER",
                          delta_color="inverse" if pct_deficits > 30 else "normal")
                
            st.divider()

            # Row-by-row functional mapper for color injection logic
            def calculate_status(row):
                forward_demand = row[f"Demand Next {rolling_window} Days"]
                if pd.isna(forward_demand):
                    return ""
                
                net_value = int(row["Inventory Level Provided"] - forward_demand)
                if net_value >= 0:
                    return f'<span style="color: #2e7d32; font-weight: bold;">🟢 Surplus (+{net_value})</span>'
                else:
                    return f'<span style="color: #d32f2f; font-weight: bold;">🔴 Deficit ({net_value})</span>'

            # Build and finalize table display dataframe
            df_table = df_summary.copy()
            df_table["Net Status"] = df_table.apply(calculate_status, axis=1)
            df_table[f"Demand Next {rolling_window} Days"] = df_table[f"Demand Next {rolling_window} Days"].apply(
                lambda x: f"{int(x)}" if not pd.isna(x) else ""
            )
            
            st.write(df_table.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.write("<br>", unsafe_allow_html=True)

        # --- Visual Asset 3: Collapsible Charts for Forward Window Analytics ---
        with st.expander("📊 View Forward Window Trend & Distribution Analysis", expanded=False):
            df_clean_charts = df_summary.dropna().copy()
            
            graph_col1, graph_col2 = st.columns(2)
            
            with graph_col1:
                st.markdown(f"#### 📉 Forward Window Demand Trend")
                fig_trend = go.Figure()
                
                fig_trend.add_trace(go.Scatter(
                    x=df_clean_charts["Lead Time Day"], 
                    y=df_clean_charts[f"Demand Next {rolling_window} Days"],
                    mode='lines',
                    name=f'{rolling_window}-Day Demand',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig_trend.add_hline(
                    y=requisite_inventory, 
                    line_dash="dash", 
                    line_color="#d62728", 
                    annotation_text="Your Stock Limit",
                    annotation_position="top left"
                )
                fig_trend.update_layout(
                    template="plotly_white", 
                    xaxis_title="Simulation Day Index",
                    yaxis_title="Total Window Units",
                    height=350,
                    margin=dict(t=30, b=10)
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                
            with graph_col2:
                st.markdown(f"#### 📊 Look-Forward Window Distribution")
                
                fig_hist = px.histogram(
                    df_clean_charts, 
                    x=f"Demand Next {rolling_window} Days",
                    nbins=20,
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.add_vline(
                    x=requisite_inventory, 
                    line_dash="dash", 
                    line_color="#d62728", 
                    annotation_text="Stock Ceiling",
                    annotation_position="top right"
                )
                fig_hist.update_layout(
                    template="plotly_white",
                    xaxis_title=f"Aggregated Demand in {rolling_window}-Day Windows",
                    yaxis_title="Frequency Occurrence Count",
                    height=350,
                    margin=dict(t=30, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        # Final Summary Executive Alerts
        if total_deficits > 0:
            st.error(f"❌ **Internal Sabotage Confirmed:** Volatility breached your static 'Average' allocation baseline strategy on **{total_deficits} separate window cycles** ({pct_deficits:.1f}% risk rate).")
        else:
            st.success(f"✅ **Strategic Parameter Verified.** Under these isolated settings, the current allocation buffer safely absorbed the simulated variance across all window blocks.")



with tab2:
    st.header("Demand Histogram Analyzer")
    
    # --- 1. Data Configuration ---
    st.subheader("1. Data Configuration")
    data_source = st.radio("Select Data Source:", ("Generate Synthetic Data", "Upload Your Own Data"), horizontal=True, key="ds_p1")

    df = None

    if data_source == "Generate Synthetic Data":
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            dist_type = st.selectbox("Distribution Type", ("Normal", "Poisson", "Uniform"), key="dist_p1")
        with col_b:
            avg_demand = st.number_input("Average Demand", min_value=1.0, value=100.0, key="avg_p1")
        with col_c:
            num_periods = st.number_input("Number of Periods", min_value=10, value=10000, key="periods_p1")
        with col_d:
            if dist_type == "Normal":
                variation = st.number_input("Std Dev (Variation)", min_value=0.1, value=15.0, key="v_norm")
            elif dist_type == "Uniform":
                variation = st.number_input("Range (+/-)", min_value=1.0, value=30.0, key="v_uni")
            else:
                st.markdown("<p style='padding-top:25px; color:gray;'>Poisson variation fixed by Mean.</p>", unsafe_allow_html=True)

        np.random.seed(42)
        if dist_type == "Normal":
            generated = np.random.normal(avg_demand, variation, num_periods)
        elif dist_type == "Poisson":
            generated = np.random.poisson(avg_demand, num_periods)
        else:
            generated = np.random.uniform(avg_demand - variation, avg_demand + variation, num_periods)
        
        df = pd.DataFrame({'Demand': np.floor(np.clip(generated, 0, None))})

    elif data_source == "Upload Your Own Data":
        up_col1, up_col2 = st.columns([2, 1])
        
        with up_col1:
            uploaded_file = st.file_uploader("Upload your historical demand file (.xlsx or .csv):", type=["xlsx", "csv"])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_upload = pd.read_csv(uploaded_file)
                    else:
                        df_upload = pd.read_excel(uploaded_file)
                    
                    if 'Demand' in df_upload.columns:
                        df = df_upload[['Demand']].dropna().copy()
                        df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce')
                        df = df.dropna()
                        st.success("✅ File successfully uploaded and parsed!")
                    else:
                        st.error("❌ Invalid Format: Your file must contain a column named exactly **'Demand'**.")
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
                    
        with up_col2:
            st.markdown("#### 📋 Download Template")
            st.caption("Please match your data format to this template. The sheet must include a column header named **Demand**.")
            
            template_df = pd.DataFrame({'Demand': [120, 95, 110, 135, 80, 105, 115]})
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                template_df.to_excel(writer, index=False, sheet_name='Template')
            
            st.download_button(
                label="📥 Download Excel Template",
                data=buffer.getvalue(),
                file_name="demand_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    # --- Collapsible Raw Data Table ---
    if df is not None:
        with st.expander("🔢 View / Download Raw Data Table", expanded=False):
            raw_display_df = df.copy()
            raw_display_df.index.name = "Period"
            
            exp_col1, exp_col2 = st.columns([3, 1])
            with exp_col1:
                st.dataframe(raw_display_df, use_container_width=True, height=250)
            with exp_col2:
                st.markdown("#### Export Current Data")
                st.caption("Download this active dataset as a CSV file for offline use.")
                csv_data = raw_display_df.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name="demand_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # --- 2. Advanced Analysis (Thresholds & Percentiles) ---
    if df is not None:
        st.divider()
        st.subheader("2. Probability & Coverage Analysis")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.markdown("#### Threshold Lookup (Points Below X)")
            threshold = st.number_input("Enter Demand Value:", value=40.0, step=1.0)
            count_below = len(df[df['Demand'] < threshold])
            percent_below = (count_below / len(df)) * 100
            st.metric(f"Chances of Demand < {threshold}", f"{percent_below:.1f}%")
            st.caption(f"There are {count_below} periods where demand was less than {threshold}.")

        with analysis_col2:
            st.markdown("#### Percentile Lookup (Coverage Level)")
            target_perc = st.number_input("Enter Service Level % (e.g. 95):", min_value=0.0, max_value=100.0, value=95.0, step=1.0)
            demand_at_perc = np.percentile(df['Demand'], target_perc)
            st.metric(f"Demand at {target_perc}% Service Level", f"{int(demand_at_perc)}")
            st.caption(f"To cover {target_perc}% of all periods, you need to satisfy a demand of {int(demand_at_perc)}.")

        # --- 3. Visual Distribution & Tables Below ---
        st.divider()
        st.subheader("3. Visual Distribution")
        
        num_bins = st.slider("Select Number of Bins:", 5, 50, 15)
        
        counts, bin_edges = np.histogram(df['Demand'], bins=num_bins)
        bin_size = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1

        fig = px.histogram(df, x="Demand", template="plotly_white", color_discrete_sequence=['#4F8BF9'])
        
        fig.update_traces(
            xbins=dict(
                start=bin_edges[0],
                end=bin_edges[-1],
                size=bin_size
            )
        )
        
        fig.add_vline(
            x=threshold, 
            line_dash="dot", 
            line_color="#EF553B", 
            line_width=2.5,
            annotation_text=f"Threshold ({threshold})", 
            annotation_position="top left"
        )
        
        fig.add_vline(
            x=demand_at_perc, 
            line_dash="dot", 
            line_color="#00CC96", 
            line_width=2.5,
            annotation_text=f"{target_perc}% Service Level ({int(demand_at_perc)})", 
            annotation_position="top right"
        )
        
        fig.update_layout(bargap=0.1, xaxis_title="Demand Quantity", yaxis_title="Count of Periods")
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        table_col1, table_col2 = st.columns([1, 1])

        with table_col1:
            st.markdown("#### 📋 Statistical Summary")
            summary_stats = df['Demand'].describe().to_frame().T
            st.dataframe(summary_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']], use_container_width=True)

        with table_col2:
            st.markdown("#### Bin Frequency Table")
            pct_total = counts / len(df) * 100
            
            bin_df = pd.DataFrame({
                "Bin Range": [f"{int(bin_edges[i])} - {int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)],
                "Frequency (Count)": counts,
                "% of Total": pct_total.round(1),
                "Cum. Count": counts.cumsum(),
                "Cum. %": pct_total.cumsum().round(1)
            })
            st.dataframe(bin_df, use_container_width=True, hide_index=True)

        # --- 4. Coefficient of Variation (CoV) Analysis ---
        st.divider()
        st.subheader("📊 Demand Volatility Analysis (CoV)")
        
        cov_col1, cov_col2 = st.columns([1, 2])
        
        with cov_col1:
            st.markdown("#### Formula")
            st.latex(r"CoV = \frac{\sigma}{\mu}")
            st.caption(r"Where $\sigma$ = Standard Deviation and $\mu$ = Mean")
            
        with cov_col2:
            # Extract statistics directly from data stream
            mean_val = float(df['Demand'].mean())
            std_val = float(df['Demand'].std())
            
            # Defensive check for edge case where mean is zero
            cov_val = (std_val / mean_val) if mean_val > 0 else 0.0
            
            # Determine demand volatility profile category
            if cov_val <= 0.10:
                status_text = "🟢 Ultra-Stable / Constant"
                explanation = "Highly repetitive and predictable demand. Use automated just-in-time (JIT) scheduling or lean kanbans. Minimize safety stock to release working capital."
                alert_type = "success"
            elif cov_val <= 0.25:
                status_text = "🟢 Stable / Predictable"
                explanation = "Normal variation patterns present. Standard statistical forecasting and fixed reorder points will yield high accuracy with minimal safety stock buffers."
                alert_type = "success"
            elif cov_val <= 0.50:
                status_text = "🟡 Moderate Volatility"
                explanation = "Demand exhibits noticeable fluctuations. Requires proactive demand sensing and traditional statistical safety stocks to counter stockout risks."
                alert_type = "warning"
            elif cov_val <= 1.00:
                status_text = "🟠 High Volatility"
                explanation = "Highly variable demand spikes. Avoid automated ordering systems without collaborative forecasting inputs. Expect to maintain higher, dynamic safety stock thresholds."
                alert_type = "warning"
            else:
                status_text = "🔴 Erratic / Lumpy / Sporadic"
                explanation = "Highly unpredictable or intermittent demand. Traditional safety stock formulas do not work well here. Consider move-to-order (MTO) execution or project-based buffers."
                alert_type = "error"
                
            # Render key calculation metrics inside columns
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Mean ($\mu$)", f"{mean_val:.2f}")
            with m_col2:
                st.metric("Std Dev ($\sigma$)", f"{std_val:.2f}")
            with m_col3:
                st.metric("Calculated CoV", f"{cov_val:.3f}")
                
            # Render descriptive behavioral classification banner
            st.markdown(f"### Profile: {status_text}")
            st.info(explanation)


with tab3:
    st.header("🔄 Periodic Review Analysis (Target-Level System)")
    
    st.markdown("""
    In a periodic review system, inventory is checked at fixed intervals. The strategy must account for the mechanical reality of the **Protection Interval**—the time from when an order is placed until the *next* order can be placed and received.
    """)

    # --- 1. System Parameters Input ---
    st.subheader("1. Supply Chain Parameters")
    
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        pr_avg_demand = st.number_input("Average Daily Demand ($\mu$)", value=100.0, step=10.0)
        pr_std_dev = st.number_input("Demand Std Dev ($\sigma$)", value=15.0, step=5.0)
    with p_col2:
        review_period = st.number_input("Review Period ($T$ days)", value=14, min_value=1, step=1)
        lead_time = st.number_input("Lead Time ($L$ days)", value=7, min_value=1, step=1)
    with p_col3:
        target_service_level = st.slider("Target Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.1)
        z_score = norm.ppf(target_service_level / 100.0)
    with p_col4:
        ordering_cost = st.number_input("Ordering Cost ($ per order)", value=250.0, step=50.0)
        holding_cost_annual = st.number_input("Annual Holding Cost ($ per unit)", value=15.0, step=1.0)
        holding_cost_daily = holding_cost_annual / 365.0

    # --- 2. Mathematical Target Calculation ---
    st.divider()
    st.subheader("2. Recommended Target Inventory Level")
    
    protection_interval = review_period + lead_time
    expected_demand_pi = pr_avg_demand * protection_interval
    std_dev_pi = pr_std_dev * np.sqrt(protection_interval)
    safety_stock = z_score * std_dev_pi
    recommended_target = expected_demand_pi + safety_stock

    c_math1, c_math2 = st.columns([1.5, 1])
    with c_math1:
        st.markdown("**Target Level (Order-Up-To) Formula:**")
        st.latex(r"OUL = \mu(T+L) + Z\sigma\sqrt{T+L}")
        st.caption("Where $T$ = Review Period, $L$ = Lead Time, $Z$ = Service Level Factor.")
        
    with c_math2:
        st.metric("Calculated Recommended Target", f"{int(recommended_target)} Units")
        st.caption(f"Includes {int(expected_demand_pi)} expected demand + {int(safety_stock)} safety stock buffer.")

    # --- 3. User Override & Comparative Simulation ---
    st.divider()
    st.subheader("3. Comparative Strategy Simulation")
    
    st.markdown("Test the recommended mathematical target against your own defined target level across a 365-day operational simulation.")
    
    user_target = st.number_input("Enter User-Defined Target Level:", value=int(recommended_target * 0.9), step=50)
    
    # Generate static demand array for fair comparison
    np.random.seed(st.session_state.seed_counter)
    sim_days_pr = 365
    daily_demand_pr = np.clip(np.random.normal(pr_avg_demand, pr_std_dev, sim_days_pr), 0, None).round(0)

    # Simulation Function
    def simulate_periodic_system(demand_array, T, L, target, order_c, hold_c_daily):
        inventory = target
        pipeline = []
        inv_history = []
        
        total_demand = 0
        units_fulfilled = 0
        orders_placed = 0
        holding_units_total = 0
        
        for day in range(len(demand_array)):
            current_demand = demand_array[day]
            
            # Receive pipeline inventory
            for order in pipeline[:]:
                if order['arrival'] == day:
                    inventory += order['qty']
                    pipeline.remove(order)
            
            # Fulfill demand
            fulfilled = min(inventory, current_demand)
            inventory -= fulfilled
            
            total_demand += current_demand
            units_fulfilled += fulfilled
            holding_units_total += inventory
            
            # Review and Order Logic
            if day % T == 0:
                on_order = sum(o['qty'] for o in pipeline)
                inv_position = inventory + on_order
                
                if inv_position < target:
                    order_qty = target - inv_position
                    pipeline.append({'arrival': day + L, 'qty': order_qty})
                    orders_placed += 1
                    
            inv_history.append(inventory)
            
        total_order_cost = orders_placed * order_c
        total_holding_cost = holding_units_total * hold_c_daily
        
        return {
            'history': inv_history,
            'total_demand': total_demand,
            'units_fulfilled': units_fulfilled,
            'orders_placed': orders_placed,
            'avg_inventory': holding_units_total / len(demand_array),
            'total_order_cost': total_order_cost,
            'total_holding_cost': total_holding_cost,
            'total_cost': total_order_cost + total_holding_cost
        }

    # Run simulations
    res_recommended = simulate_periodic_system(daily_demand_pr, review_period, lead_time, recommended_target, ordering_cost, holding_cost_daily)
    res_user = simulate_periodic_system(daily_demand_pr, review_period, lead_time, user_target, ordering_cost, holding_cost_daily)

    # --- KPI Scorecards (Displaying Absolute Physical Values) ---
    kpi_col1, kpi_col2 = st.columns(2)
    
    with kpi_col1:
        st.markdown("#### 🔵 Recommended Strategy")
        st.metric("Total System Cost", f"${res_recommended['total_cost']:,.2f}")
        st.markdown(f"**Total Physical Demand:** {int(res_recommended['total_demand'])} units")
        st.markdown(f"**Total Physical Fulfilled:** {int(res_recommended['units_fulfilled'])} units")
        st.markdown(f"**Average Inventory Carried:** {int(res_recommended['avg_inventory'])} units")
        st.markdown(f"**Total Orders Placed:** {res_recommended['orders_placed']}")
        
    with kpi_col2:
        st.markdown("#### ⚪ User-Defined Strategy")
        cost_diff = res_user['total_cost'] - res_recommended['total_cost']
        st.metric(
            "Total System Cost", 
            f"${res_user['total_cost']:,.2f}", 
            delta=f"{'+' if cost_diff > 0 else ''}${cost_diff:,.2f} vs Recommended",
            delta_color="inverse"
        )
        st.markdown(f"**Total Physical Demand:** {int(res_user['total_demand'])} units")
        st.markdown(f"**Total Physical Fulfilled:** {int(res_user['units_fulfilled'])} units")
        st.markdown(f"**Average Inventory Carried:** {int(res_user['avg_inventory'])} units")
        st.markdown(f"**Total Orders Placed:** {res_user['orders_placed']}")

    # --- Visual Asset: Clean Blue Outline Comparison Chart ---
    st.write("<br>", unsafe_allow_html=True)
    st.markdown("#### 📉 Inventory Level Trajectory Comparison")
    
    fig_comp = go.Figure()
    
    # Recommended Trace (Solid Blue)
    fig_comp.add_trace(go.Scatter(
        x=list(range(sim_days_pr)), 
        y=res_recommended['history'], 
        mode='lines', 
        name='Recommended Target',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # User Trace (Light Blue / Dashed)
    fig_comp.add_trace(go.Scatter(
        x=list(range(sim_days_pr)), 
        y=res_user['history'], 
        mode='lines', 
        name='User-Defined Target',
        line=dict(color='#85c1e9', width=2, dash='dot')
    ))
    
    # Zero line to spot stockouts easily
    fig_comp.add_hline(y=0, line_dash="solid", line_color="#333333", line_width=1)
    
    fig_comp.update_layout(
        template="plotly_white",
        xaxis_title="Simulation Day",
        yaxis_title="Physical Units On Hand",
        height=400,
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)
