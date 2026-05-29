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

tab1, tab2, tab3, tab4 = st.tabs(["Average Demand", "📊 Demand Histogram", "🔄 Periodic Review", "Inventory Audit"])

# ==========================================
# TAB 1: AVERAGE DEMAND ANALYZER
# ==========================================
with tab1:
    st.header("The Basic Thumb Rule Used For Inventory Planning")
    
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
            if st.button("🔄 Regenerate Demand", key="regen_tab1"):
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
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Days with Valid Window", f"{total_valid_days} Days")
            with m2:
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

            def calculate_status(row):
                forward_demand = row[f"Demand Next {rolling_window} Days"]
                if pd.isna(forward_demand):
                    return ""
                
                net_value = int(row["Inventory Level Provided"] - forward_demand)
                if net_value >= 0:
                    return f'<span style="color: #2e7d32; font-weight: bold;">🟢 Surplus (+{net_value})</span>'
                else:
                    return f'<span style="color: #d32f2f; font-weight: bold;">🔴 Deficit ({net_value})</span>'

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

        if total_deficits > 0:
            st.error(f"❌ **Internal Sabotage Confirmed:** Volatility breached your static 'Average' allocation baseline strategy on **{total_deficits} separate window cycles** ({pct_deficits:.1f}% risk rate).")
        else:
            st.success(f"✅ **Strategic Parameter Verified.** Under these isolated settings, the current allocation buffer safely absorbed the simulated variance across all window blocks.")


# ==========================================
# TAB 2: DEMAND HISTOGRAM ANALYZER
# ==========================================
with tab2:
    st.header("Demand Histogram Analyzer")
    
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

        st.divider()
        st.subheader("📊 Demand Volatility Analysis (CoV)")
        
        cov_col1, cov_col2 = st.columns([1, 2])
        
        with cov_col1:
            st.markdown("#### Formula")
            st.latex(r"CoV = \frac{\sigma}{\mu}")
            st.caption(r"Where $\sigma$ = Standard Deviation and $\mu$ = Mean")
            
        with cov_col2:
            mean_val = float(df['Demand'].mean())
            std_val = float(df['Demand'].std())
            cov_val = (std_val / mean_val) if mean_val > 0 else 0.0
            
            if cov_val <= 0.10:
                status_text = "🟢 Ultra-Stable / Constant"
                explanation = "Highly repetitive and predictable demand. Use automated just-in-time (JIT) scheduling or lean kanbans. Minimize safety stock to release working capital."
            elif cov_val <= 0.25:
                status_text = "🟢 Stable / Predictable"
                explanation = "Normal variation patterns present. Standard statistical forecasting and fixed reorder points will yield high accuracy with minimal safety stock buffers."
            elif cov_val <= 0.50:
                status_text = "🟡 Moderate Volatility"
                explanation = "Demand exhibits noticeable fluctuations. Requires proactive demand sensing and traditional statistical safety stocks to counter stockout risks."
            elif cov_val <= 1.00:
                status_text = "🟠 High Volatility"
                explanation = "Highly variable demand spikes. Avoid automated ordering systems without collaborative forecasting inputs. Expect to maintain higher, dynamic safety stock thresholds."
            else:
                status_text = "🔴 Erratic / Lumpy / Sporadic"
                explanation = "Highly unpredictable or intermittent demand. Traditional safety stock formulas do not work well here. Consider move-to-order (MTO) execution or project-based buffers."
                
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Mean ($\mu$)", f"{mean_val:.2f}")
            with m_col2:
                st.metric("Std Dev ($\sigma$)", f"{std_val:.2f}")
            with m_col3:
                st.metric("Calculated CoV", f"{cov_val:.3f}")
                
            st.markdown(f"### Profile: {status_text}")
            st.info(explanation)

# ==========================================
# TAB 3: PERIODIC REVIEW (VECTORIZED LOGIC)
# ==========================================
with tab3:
    st.header("🔄 Periodic Review Analysis (Target-Level System)")
    
    st.markdown("""
    In a periodic review system, inventory is checked at fixed intervals. The strategy must account for the mechanical reality of the **Protection Interval**—the time from when an order is placed until the *next* order can be placed and received.
    """)

    # --- Action: Regenerate Demand Button ---
    btn_col1, btn_col2 = st.columns([1, 5])
    with btn_col1:
        if st.button("🔄 Generate New Demand", key="regen_demand_pr"):
            st.session_state.seed_counter += 1

    # --- 1. Baseline System Parameters Input ---
    st.subheader("1. Supply Chain Parameters & Recommended Baseline")
    
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    with p_col1:
        pr_avg_demand = st.number_input("Avg Daily Demand", value=100.0, step=10.0)
        pr_std_dev = st.number_input("Demand Std Dev", value=15.0, step=5.0)
    with p_col2:
        review_period = st.number_input("Recommended Review (Days)", value=14, min_value=1, step=1)
        lead_time = st.number_input("Lead Time (Days)", value=7, min_value=1, step=1)
    with p_col3:
        target_service_level = st.slider("Target Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.1)
        z_score = norm.ppf(target_service_level / 100.0)
    with p_col4:
        unit_cost = st.number_input("Unit Cost ($)", value=50.0, step=5.0)
        ordering_cost = st.number_input("Ordering Cost ($/order)", value=250.0, step=50.0, key="baseline_oc")
    with p_col5:
        holding_cost_pct = st.number_input("Annual Holding Cost (%)", value=20.0, step=1.0)
        holding_cost_annual = unit_cost * (holding_cost_pct / 100.0)
        holding_cost_daily = holding_cost_annual / 365.0

    # Calculate Recommended Baseline Target
    protection_interval = review_period + lead_time
    expected_demand_pi = pr_avg_demand * protection_interval
    std_dev_pi = pr_std_dev * np.sqrt(protection_interval)
    safety_stock = z_score * std_dev_pi
    recommended_target = expected_demand_pi + safety_stock

    st.info(f"**Calculated Baseline Target:** {int(recommended_target)} Units (Accommodating a {review_period}-day review cycle and {lead_time}-day lead time).")

    # --- 2. Multi-Scenario Customization Setup ---
    st.divider()
    
    def sync_ordering_costs():
        for i in range(5):  # Max slider value is 5
            st.session_state[f"oc_key_{i}"] = st.session_state.baseline_oc

    head_col1, head_col2 = st.columns([2, 1])
    with head_col1:
        st.subheader("2. Multi-Scenario Strategy Comparison")
    with head_col2:
        st.write("") # Spacing
        st.button("📋 Sync Baseline Cost to All Scenarios", on_click=sync_ordering_costs)
        
    st.markdown("Test the recommended baseline against custom strategies. Modify the review period to see the mathematically optimum target update instantly.")
    
    num_scenarios = st.slider("Select Number of Custom Scenarios to Compare:", min_value=1, max_value=5, value=2)
    
    scenarios_data = []
    s_cols = st.columns(num_scenarios)
    
    for i, col in enumerate(s_cols):
        with col:
            st.markdown(f"##### Scenario {i+1}")
            
            default_t = int(review_period + ((i+1) * 7)) 
            t_val = st.number_input(f"Review Period (Days)", value=default_t, min_value=1, step=1, key=f"t_{i}")
            
            u_pi = t_val + lead_time
            opt_target = (pr_avg_demand * u_pi) + (z_score * (pr_std_dev * np.sqrt(u_pi)))
            st.caption(f"✨ **Optimum Target:** {int(opt_target)} Units")
            target_val = st.number_input(f"Target Level (Units)", value=int(opt_target), step=50, key=f"target_{i}")
            
            if f"oc_key_{i}" not in st.session_state:
                st.session_state[f"oc_key_{i}"] = ordering_cost
                
            oc_val = st.number_input(f"Ordering Cost ($)", step=10.0, key=f"oc_key_{i}")
            
            scenarios_data.append({
                'name': f"Scenario {i+1}", 
                'T': t_val, 
                'Target': target_val, 
                'OrderCost': oc_val
            })

    # --- NumPy Optimized Simulation Engine ---
    np.random.seed(st.session_state.seed_counter)
    sim_days_pr = 365
    daily_demand_pr = np.clip(np.random.normal(pr_avg_demand, pr_std_dev, sim_days_pr), 0, None).round(0)

    def simulate_periodic_system_vectorized(demand_array, T, L, target, order_c, hold_c_daily):
        sim_days = len(demand_array)
        inventory_history = np.zeros(sim_days)
        receipts = np.zeros(sim_days + L + 1) 
        
        current_inv = target
        order_sizes = []
        units_fulfilled = 0
        
        for day in range(sim_days):
            current_inv += receipts[day]
            current_demand = demand_array[day]
            
            fulfilled = min(max(current_inv, 0), current_demand)
            current_inv -= fulfilled
            units_fulfilled += fulfilled
            
            inventory_history[day] = current_inv
            
            if day % T == 0:
                on_order = np.sum(receipts[day+1:day+L+1])
                inv_position = current_inv + on_order
                
                if inv_position < target:
                    order_qty = target - inv_position
                    receipts[day + L] += order_qty
                    order_sizes.append(order_qty)
                    
        holding_units_total = np.sum(np.maximum(inventory_history, 0))
        orders_placed = len(order_sizes)
        total_order_cost = orders_placed * order_c
        total_holding_cost = holding_units_total * hold_c_daily
        total_demand_sim = np.sum(demand_array)
        
        return {
            'history': inventory_history,
            'total_demand': total_demand_sim,
            'units_fulfilled': units_fulfilled,
            'lost_sales': total_demand_sim - units_fulfilled,
            'fill_rate': (units_fulfilled / total_demand_sim) * 100 if total_demand_sim > 0 else 0,
            'orders_placed': orders_placed,
            'min_order_size': np.min(order_sizes) if order_sizes else 0,
            'max_order_size': np.max(order_sizes) if order_sizes else 0,
            'avg_order_size': np.mean(order_sizes) if order_sizes else 0,
            'avg_inventory': holding_units_total / sim_days,
            'max_inventory': np.max(np.maximum(inventory_history, 0)),
            'min_inventory': np.min(inventory_history),
            'total_order_cost': total_order_cost,
            'total_holding_cost': total_holding_cost,
            'total_cost': total_order_cost + total_holding_cost
        }

    # Execute simulations 
    res_baseline = simulate_periodic_system_vectorized(daily_demand_pr, review_period, lead_time, recommended_target, ordering_cost, holding_cost_daily)
    
    scenario_results = []
    for s in scenarios_data:
        res = simulate_periodic_system_vectorized(daily_demand_pr, s['T'], lead_time, s['Target'], s['OrderCost'], holding_cost_daily)
        scenario_results.append(res)

    # --- 3. Logically Bifurcated Summary Tables ---
    st.divider()
    st.markdown("### 📊 Policy Comparison & KPI Summary")
    
    def fmt_usd(val): return f"${val:,.2f}"
    
    # 3A. Operational Health Matrix
    st.markdown("#### A. Operational & Capital Health Matrix")
    ops_data = {
        "Metric": [
            "Review Interval", 
            "Target Inventory Level", 
            "Fill Rate (%)", 
            "Lost Sales (Units)", 
            "Min Inventory Level (Depth)",
            "Avg Working Capital", 
            "Max Working Capital"
        ]
    }
    ops_data["Recommended Baseline"] = [
        f"{review_period} Days", f"{int(recommended_target)}", f"{res_baseline['fill_rate']:.2f}%", 
        f"{int(res_baseline['lost_sales'])}", f"{int(res_baseline['min_inventory'])}",
        fmt_usd(res_baseline['avg_inventory'] * unit_cost), fmt_usd(res_baseline['max_inventory'] * unit_cost)
    ]
    for idx, s in enumerate(scenarios_data):
        res = scenario_results[idx]
        ops_data[s['name']] = [
            f"{s['T']} Days", f"{int(s['Target'])}", f"{res['fill_rate']:.2f}%", 
            f"{int(res['lost_sales'])}", f"{int(res['min_inventory'])}",
            fmt_usd(res['avg_inventory'] * unit_cost), fmt_usd(res['max_inventory'] * unit_cost)
        ]
    st.dataframe(pd.DataFrame(ops_data), use_container_width=True, hide_index=True)

    # 3B. Order Dynamics Matrix
    st.markdown("#### B. Order Dynamics Matrix")
    order_data = {
        "Metric": ["Total No. of Orders", "Average Order Size", "Minimum Order Size", "Maximum Order Size"]
    }
    order_data["Recommended Baseline"] = [
        f"{res_baseline['orders_placed']}", f"{int(res_baseline['avg_order_size'])} Units", 
        f"{int(res_baseline['min_order_size'])} Units", f"{int(res_baseline['max_order_size'])} Units"
    ]
    for idx, s in enumerate(scenarios_data):
        res = scenario_results[idx]
        order_data[s['name']] = [
            f"{res['orders_placed']}", f"{int(res['avg_order_size'])} Units", 
            f"{int(res['min_order_size'])} Units", f"{int(res['max_order_size'])} Units"
        ]
    st.dataframe(pd.DataFrame(order_data), use_container_width=True, hide_index=True)

    # 3C. Financial Matrix
    st.markdown("#### C. Financial Projections Matrix")
    fin_data = {
        "Metric": ["Applied Ordering Cost ($/order)", "Total Ordering Cost", "Total Holding Cost", "Total System Cost"]
    }
    fin_data["Recommended Baseline"] = [
        fmt_usd(ordering_cost), fmt_usd(res_baseline['total_order_cost']), 
        fmt_usd(res_baseline['total_holding_cost']), fmt_usd(res_baseline['total_cost'])
    ]
    for idx, s in enumerate(scenarios_data):
        res = scenario_results[idx]
        fin_data[s['name']] = [
            fmt_usd(s['OrderCost']), fmt_usd(res['total_order_cost']), 
            fmt_usd(res['total_holding_cost']), fmt_usd(res['total_cost'])
        ]
    st.dataframe(pd.DataFrame(fin_data), use_container_width=True, hide_index=True)

    # --- 4. Visual Bifurcation & Trajectory ---
    chart_col1, chart_col2 = st.columns([1, 1])
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    with chart_col1:
        st.markdown("#### Cost Bifurcation Analysis")
        names = ["Baseline"] + [s['name'] for s in scenarios_data]
        order_costs = [res_baseline['total_order_cost']] + [r['total_order_cost'] for r in scenario_results]
        hold_costs = [res_baseline['total_holding_cost']] + [r['total_holding_cost'] for r in scenario_results]
        
        fig_cost = go.Figure(data=[
            go.Bar(name='Ordering Cost', x=names, y=order_costs, marker_color='#2ca02c'),
            go.Bar(name='Holding Cost', x=names, y=hold_costs, marker_color='#1f77b4')
        ])
        fig_cost.update_layout(
            barmode='stack', template="plotly_white", yaxis_title="Total Cost ($)",
            height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with chart_col2:
        st.markdown("#### Physical Inventory Trajectory")
        fig_comp = go.Figure()
        
        fig_comp.add_trace(go.Scatter(
            x=list(range(sim_days_pr)), y=res_baseline['history'], mode='lines', 
            name='Baseline', line=dict(color='#1f77b4', width=3)
        ))
        
        for idx, s in enumerate(scenarios_data):
            fig_comp.add_trace(go.Scatter(
                x=list(range(sim_days_pr)), y=scenario_results[idx]['history'], mode='lines', 
                name=s['name'], line=dict(color=colors[idx], width=1.5, dash='dot')
            ))
        
        fig_comp.add_hline(y=0, line_dash="solid", line_color="#333333", line_width=1)
        fig_comp.update_layout(
            template="plotly_white", xaxis_title="Simulation Day", yaxis_title="Units On Hand",
            height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # --- 5. Blocked Working Capital Chart ---
    st.write("<br>", unsafe_allow_html=True)
    st.markdown("#### 💰 Blocked Working Capital Trajectory")
    st.caption("Visualizes the daily capital tied up on the warehouse floor (ignores backorders).")
    
    fig_wc = go.Figure()
    
    # Baseline WC
    baseline_wc = np.maximum(res_baseline['history'], 0) * unit_cost
    fig_wc.add_trace(go.Scatter(
        x=list(range(sim_days_pr)), y=baseline_wc, mode='lines', 
        name='Baseline', line=dict(color='#1f77b4', width=3)
    ))
    
    # Scenarios WC
    for idx, s in enumerate(scenarios_data):
        scenario_wc = np.maximum(scenario_results[idx]['history'], 0) * unit_cost
        fig_wc.add_trace(go.Scatter(
            x=list(range(sim_days_pr)), y=scenario_wc, mode='lines', 
            name=s['name'], line=dict(color=colors[idx], width=1.5, dash='dot')
        ))
        
    fig_wc.update_layout(
        template="plotly_white", xaxis_title="Simulation Day", yaxis_title="Capital Blocked ($)",
        height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_wc, use_container_width=True)

    # --- 6. Collapsible Raw Data Logs ---
    with st.expander("📋 View Daily Simulation Log Tables"):
        st.markdown("Raw 365-day tracking for Physical Inventory levels side-by-side.")
        
        log_data = {
            "Day": range(1, sim_days_pr + 1),
            "Daily Demand": daily_demand_pr.astype(int),
            "Baseline Inv": res_baseline['history'].astype(int)
        }
        
        for idx, s in enumerate(scenarios_data):
            log_data[f"{s['name']} Inv"] = scenario_results[idx]['history'].astype(int)
            
        log_df = pd.DataFrame(log_data)
        
        def highlight_stockouts(val):
            color = '#ffcccc' if isinstance(val, (int, float)) and val < 0 else ''
            return f'background-color: {color}'
            
        st.dataframe(
            log_df.style.map(highlight_stockouts, subset=[c for c in log_df.columns if 'Inv' in c]), 
            use_container_width=True, hide_index=True
        )



with tab4:
    st.header("🔙 Historical Policy Backtesting")
    st.markdown("""
    Upload your actual historical demand data to simulate how different inventory policies would have performed. 
    Compare **Continuous Review** (ordering a fixed quantity when stock hits a minimum) against **Periodic Review** (ordering up to a target level at fixed intervals).
    """)

    # --- 1. Data Ingestion & Basic EDA ---
    st.subheader("1. Data Upload & Demand Profiling")
    
    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        uploaded_file = st.file_uploader("Upload Historical Demand (.csv or .xlsx).", type=["csv", "xlsx"], key="backtest_upload")
    
    with col_up2:
        st.markdown("#### 📋 Template")
        st.caption("Ensure your file has a header named **Demand**, **Store Sale**, or select it manually upon upload. Optionally include **Closing Balance**.")
        template_df = pd.DataFrame({
            'Day': [1, 2, 3, 4, 5], 
            'Demand': [120, 95, 110, 135, 80],
            'Closing Balance': [100, 75, 50, 25, 0]
        })
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            template_df.to_excel(writer, index=False, sheet_name='Template')
        st.download_button(label="📥 Download Template", data=buffer.getvalue(), file_name="backtest_template.xlsx", mime="application/vnd.ms-excel", use_container_width=True)

    if uploaded_file is not None:
        # Read Data
        try:
            if uploaded_file.name.endswith('.csv'):
                df_bt = pd.read_csv(uploaded_file)
            else:
                df_bt = pd.read_excel(uploaded_file)
                
            # --- Column Cleanup (Handles "\n" in headers like "Closing\nBalance") ---
            df_bt.columns = [str(c).replace('\n', ' ').strip() for c in df_bt.columns]
                
            # --- Dynamic Column Mapping ---
            if 'Store Sale' in df_bt.columns:
                df_bt.rename(columns={'Store Sale': 'Demand'}, inplace=True)
            elif 'Bhiwandi Sales' in df_bt.columns:
                df_bt.rename(columns={'Bhiwandi Sales': 'Demand'}, inplace=True)
            elif 'Sales' in df_bt.columns:
                df_bt.rename(columns={'Sales': 'Demand'}, inplace=True)
                
            if 'Demand' not in df_bt.columns:
                st.warning("⚠️ Could not automatically detect a 'Demand' column.")
                selected_col = st.selectbox("Please select the column that represents your daily demand:", df_bt.columns)
                if selected_col:
                    df_bt.rename(columns={selected_col: 'Demand'}, inplace=True)
                else:
                    st.stop()
                
            df_bt['Demand'] = pd.to_numeric(df_bt['Demand'], errors='coerce').fillna(0)
            
            # --- 3. Demand Mean, Std Dev and CoV ---
            mean_dem = df_bt['Demand'].mean()
            std_dem = df_bt['Demand'].std()
            cov_dem = (std_dem / mean_dem) if mean_dem > 0 else 0
            
            st.markdown("#### 📊 Statistical Profile")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Periods (Days)", len(df_bt))
            m2.metric("Mean Demand ($\mu$)", f"{mean_dem:.1f}")
            m3.metric("Std Dev ($\sigma$)", f"{std_dem:.1f}")
            m4.metric("CoV", f"{cov_dem:.3f}", delta="High Volatility" if cov_dem > 0.5 else "Stable", delta_color="inverse" if cov_dem > 0.5 else "normal")
            
            # --- Graphical Section 1: Demand Profiles ---
            st.write("<br>", unsafe_allow_html=True)
            chart_c1, chart_c2 = st.columns(2)
            
            # Determine generic X-Axis for all historical plots
            x_axis_hist = df_bt.index
            if 'Date' in df_bt.columns:
                x_axis_hist = df_bt['Date']
            elif 'Day' in df_bt.columns:
                x_axis_hist = df_bt['Day']
            
            with chart_c1:
                st.markdown("**📉 Historical Demand Curve**")
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=x_axis_hist, y=df_bt['Demand'], mode='lines', name='Demand', line=dict(color='#1f77b4', width=2)
                ))
                fig_line.update_layout(template="plotly_white", height=300, xaxis_title="Period", yaxis_title="Units", margin=dict(t=10, b=10))
                st.plotly_chart(fig_line, use_container_width=True)
                
            with chart_c2:
                st.markdown("**📊 Demand Distribution (Histogram)**")
                fig_hist = px.histogram(df_bt, x='Demand', nbins=20, template="plotly_white", color_discrete_sequence=['#85c1e9'])
                fig_hist.update_layout(height=300, xaxis_title="Demand Quantity", yaxis_title="Frequency", margin=dict(t=10, b=10))
                st.plotly_chart(fig_hist, use_container_width=True)

            # --- Graphical Section 2: Historical Closing Balance ---
            if 'Closing Balance' in df_bt.columns:
                df_bt['Closing Balance'] = pd.to_numeric(df_bt['Closing Balance'], errors='coerce').fillna(0)
                st.write("<br>", unsafe_allow_html=True)
                st.markdown("#### 📉 Actual Historical Closing Stock (From Uploaded Data)")
                st.caption("Visualizes the raw inventory levels recorded historically before any simulated policy changes.")
                
                fig_hist_close = go.Figure()
                fig_hist_close.add_trace(go.Scatter(
                    x=x_axis_hist, 
                    y=df_bt['Closing Balance'], 
                    mode='lines', 
                    name='Historical Closing Balance',
                    line=dict(color='#9467bd', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(148, 103, 189, 0.15)'
                ))
                fig_hist_close.add_hline(y=0, line_dash="solid", line_color="#333333", line_width=1)
                fig_hist_close.update_layout(
                    template="plotly_white", 
                    xaxis_title="Historical Period", 
                    yaxis_title="Units On Hand",
                    height=300,
                    margin=dict(t=10, b=10)
                )
                st.plotly_chart(fig_hist_close, use_container_width=True)

            # --- 2. Cost & Policy Configuration ---
            st.divider()
            st.subheader("2. Financial Costs & Policy Configuration")
            
            # User Input: Costs and Service Level
            cost_c1, cost_c2, cost_c3, cost_c4, cost_c5 = st.columns(5)
            with cost_c1:
                bt_unit_cost = st.number_input("Unit Cost ($)", value=50.0, step=5.0, key="bt_uc")
            with cost_c2:
                bt_order_cost = st.number_input("Ordering Cost ($/order)", value=250.0, step=50.0, key="bt_oc")
            with cost_c3:
                bt_hold_pct = st.number_input("Annual Holding Cost (%)", value=20.0, step=1.0, key="bt_hp")
            with cost_c4:
                bt_lead_time = st.number_input("Lead Time (Days)", value=7, min_value=1, step=1, key="bt_lt")
            with cost_c5:
                bt_service_lvl = st.slider("Target Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.1, key="bt_sl")
                
            bt_hold_daily = (bt_unit_cost * (bt_hold_pct / 100.0)) / 365.0
            bt_z_score = norm.ppf(bt_service_lvl / 100.0)

            # Policy Selection Toggle
            st.write("<br>", unsafe_allow_html=True)
            policy_type = st.radio(
                "Select Inventory Control Policy to Backtest:", 
                ["Continuous Review (Reorder Point & Fixed Qty)", "Periodic Review (Fixed Interval & Target Level)"],
                horizontal=True
            )
            
            pol_c1, pol_c2, pol_c3 = st.columns(3)
            
            # Dynamic UI based on policy selection
            if "Continuous" in policy_type:
                with pol_c1:
                    calc_rop = (mean_dem * bt_lead_time) + (bt_z_score * std_dem * np.sqrt(bt_lead_time))
                    rop = st.number_input("Reorder Point (ROP)", value=int(calc_rop), step=10, help="Order is triggered when inventory drops below this.")
                with pol_c2:
                    eoq_est = np.sqrt((2 * (mean_dem*365) * bt_order_cost) / (bt_unit_cost * (bt_hold_pct/100))) if bt_hold_pct > 0 else 100
                    order_q = st.number_input("Order Quantity (Q)", value=int(eoq_est) if eoq_est > 0 else 100, step=10, help="Fixed amount ordered each time.")
                with pol_c3:
                    start_inv = st.number_input("Starting Inventory", value=int(rop + order_q/2), step=10)
            else:
                with pol_c1:
                    review_t = st.number_input("Review Period (Days)", value=14, min_value=1, step=1)
                with pol_c2:
                    pi = review_t + bt_lead_time
                    calc_targ = (mean_dem * pi) + (bt_z_score * std_dem * np.sqrt(pi))
                    target_lvl = st.number_input("Target Level (Order-Up-To)", value=int(calc_targ), step=10)
                with pol_c3:
                    start_inv = st.number_input("Starting Inventory", value=int(target_lvl), step=10)

            # --- 3. Simulation Engine ---
            demand_arr = df_bt['Demand'].values
            sim_len = len(demand_arr)
            
            inv_history = np.zeros(sim_len)
            receipts = np.zeros(sim_len + bt_lead_time + 1)
            fulfilled_arr = np.zeros(sim_len)
            
            current_inv = float(start_inv)
            orders_placed = 0
            units_fulfilled = 0
            
            for day in range(sim_len):
                current_inv += receipts[day]
                
                # Fulfill
                fulfilled = min(max(current_inv, 0), demand_arr[day])
                current_inv -= fulfilled
                units_fulfilled += fulfilled
                fulfilled_arr[day] = fulfilled
                
                inv_history[day] = current_inv
                
                # Apply specific policy rules
                if "Continuous" in policy_type:
                    inv_pos = current_inv + np.sum(receipts[day+1:day+bt_lead_time+1])
                    if inv_pos <= rop:
                        receipts[day + bt_lead_time] += order_q
                        orders_placed += 1
                else:
                    if day % review_t == 0:
                        inv_pos = current_inv + np.sum(receipts[day+1:day+bt_lead_time+1])
                        if inv_pos < target_lvl:
                            order_qty = target_lvl - inv_pos
                            receipts[day + bt_lead_time] += order_qty
                            orders_placed += 1

            # KPI Calculations
            total_dem = np.sum(demand_arr)
            lost_sales = total_dem - units_fulfilled
            fill_rate = (units_fulfilled / total_dem) * 100 if total_dem > 0 else 0
            
            physical_held = np.sum(np.maximum(inv_history, 0))
            total_hold_cost = physical_held * bt_hold_daily
            total_ord_cost = orders_placed * bt_order_cost
            total_sys_cost = total_hold_cost + total_ord_cost

            # --- 4. Results & Combined Closing Stock Curve ---
            st.divider()
            st.subheader("3. Backtest Results & Financial Impact")
            
            def f_usd(v): return f"${v:,.2f}"
            
            res_c1, res_c2, res_c3, res_c4 = st.columns(4)
            res_c1.metric("Total System Cost", f_usd(total_sys_cost))
            res_c2.metric("Fill Rate (%)", f"{fill_rate:.2f}%", delta=f"{int(lost_sales)} Units Lost", delta_color="inverse" if lost_sales > 0 else "normal")
            res_c3.metric("Holding Cost (Capital Blocked)", f_usd(total_hold_cost))
            res_c4.metric("Ordering Cost (Logistics)", f_usd(total_ord_cost), delta=f"{orders_placed} Total Orders", delta_color="off")

            # 4. Closing Stock & Demand Overlay Graph
            st.markdown("#### 📉 Simulated Demand vs. Inventory Trajectory Overlay")
            st.caption("Visualizes your actual daily demand (bars) against the NEW resulting inventory levels (line) based on the policy above.")
            
            fig_close = go.Figure()
            
            # 1. Overlay Demand Trace (Bars)
            fig_close.add_trace(go.Bar(
                x=x_axis_hist,
                y=df_bt['Demand'],
                name='Actual Demand (Units)',
                marker_color='rgba(156, 163, 175, 0.5)' # Neutral gray
            ))

            # 2. Add Closing Stock Trace (Line)
            fig_close.add_trace(go.Scatter(
                x=x_axis_hist, 
                y=inv_history, 
                mode='lines', 
                name='Simulated Closing Stock',
                line=dict(color='#2ca02c', width=2),
                fill='tozeroy',
                fillcolor='rgba(44, 160, 44, 0.15)'
            ))
            
            fig_close.add_hline(y=0, line_dash="solid", line_color="#d62728", line_width=1.5)
            
            # ROP / Target line depending on policy
            if "Continuous" in policy_type:
                fig_close.add_hline(y=rop, line_dash="dash", line_color="#1f77b4", annotation_text="Reorder Point (ROP)", annotation_position="top left")
            else:
                fig_close.add_hline(y=target_lvl, line_dash="dash", line_color="#1f77b4", annotation_text="Target Level", annotation_position="top left")

            fig_close.update_layout(
                template="plotly_white", 
                xaxis_title="Historical Period", 
                yaxis_title="Quantity (Units)",
                height=450,
                margin=dict(t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_close, use_container_width=True)
            
            # --- 5. Data Table of the Simulated Policy ---
            with st.expander("📋 View Simulated Daily Log Table"):
                st.markdown("Raw day-by-day tracking for Demand, Fulfillment, and Simulated Inventory.")
                
                sim_log_df = pd.DataFrame({
                    "Period": x_axis_hist,
                    "Demand": demand_arr.astype(int),
                    "Units Fulfilled": fulfilled_arr.astype(int),
                    "Inventory Receipts": receipts[:sim_len].astype(int),
                    "Simulated Closing Stock": inv_history.astype(int)
