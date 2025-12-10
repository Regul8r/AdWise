# ============================================================================
# MARKETING BUDGET OPTIMIZER - COMPLETE DASHBOARD
# ============================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np




# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="AdWise - Smart Marketing Budgets",
    page_icon="📊",
    layout="wide"
)


st.markdown("""
    <style>
    .stChatMessage {
        line-height: 1.8;
        font-size: 16px;
    }
    .stChatMessage p {
        margin-bottom: 16px;
    }
    .stChatMessage ul, .stChatMessage ol {
        margin: 16px 0;
        padding-left: 24px;
    }
    .stChatMessage li {
        margin-bottom: 10px;
    }
    .stChatMessage h3, .stChatMessage h4 {
        margin-top: 24px;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# Welcome screen for first-time users
if "onboarded" not in st.session_state:
    st.session_state.onboarded = False

if not st.session_state.onboarded:
    st.markdown("""
        <h1 style='text-align: center;'>📊 Welcome to AdWise!</h1>
        <p style='text-align: center; font-size: 18px; color: gray;'>
            Your AI-powered marketing budget advisor
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            ### What AdWise Does:
            
            - 🎯 **Analyzes** your marketing spend
            - 🔮 **Predicts** which budget mix works best  
            - 💬 **Advises** you with AI-powered recommendations
            
            ---
            
            ### How to Use:
            
            1. Enter your product info in the sidebar
            2. Upload your marketing data (or use our sample)
            3. Explore the tabs to optimize your budget
            
            ---
            
            **No marketing experience needed.** AdWise explains everything in plain English.
        """)
        
        st.markdown("")
        
        if st.button("🚀 Get Started", use_container_width=True):
            st.session_state.onboarded = True
            st.rerun()
    
    st.stop()  # Don't show the rest of the app until they click "Get Started"


# ============================================================================
# TITLE & INSTRUCTIONS
# ============================================================================
st.title("📊 Marketing Budget Optimizer")
st.markdown("### AI-Powered Budget Allocation Tool")

# Add user instructions
with st.expander("ℹ️ How to Use This Tool"):
    st.markdown("""
    **3 Ways to Get Started:**
    
    1. **Use Sample Data** (Easiest)
       - Check "Or use sample data" in the sidebar
       - Explore the tool with 300 real campaigns
    
    2. **Download Template** (Recommended for your data)
       - Click "📄 Download Sample CSV" in sidebar
       - Fill in your marketing budget data
       - Upload it back
    
    3. **Upload Your Own CSV** (Most flexible)
       - Your CSV just needs marketing spend columns and sales data
       - The tool will help you map your column names!
    
    **What You'll Get:**
    - Your current marketing strategy classification
    - Product-specific campaign recommendations
    - Budget optimization suggestions
    - Expected sales predictions
    """)

# ============================================================================
# SIDEBAR - PRODUCT INFORMATION
# ============================================================================
st.sidebar.header("📦 Product Information")

# Product Category
product_category = st.sidebar.selectbox(
    "Product Category",
    ["Sports Apparel", "Electronics", "Beauty Products", 
     "Home Goods", "Food & Beverage", "Software/SaaS", "Other"]
)

# Target Audience
target_audience = st.sidebar.multiselect(
    "Target Audience",
    ["Teens (13-17)", "Young Adults (18-34)", "Adults (35-54)", 
     "Seniors (55+)", "Fitness Enthusiasts", "Professionals", 
     "Parents", "Students"],
    default=["Young Adults (18-34)"]
)

# Price Point
price_point = st.sidebar.radio(
    "Price Point",
    ["Budget ($0-50)", "Mid-Range ($50-150)", "Premium ($150+)"],
    index=1
)

# Average Order Value
aov = st.sidebar.number_input("Average Order Value ($)", 0, 10000, 85)

# Sales Channel
sales_channel = st.sidebar.multiselect(
    "Sales Channel",
    ["E-commerce Website", "Amazon/Marketplace", 
     "Brick & Mortar", "Both Online & Offline"],
    default=["E-commerce Website"]
)

# Monthly Budget
monthly_budget = st.sidebar.number_input(
    "Monthly Marketing Budget ($)", 
    0, 1000000, 50000, 5000
)

# Store in session state
st.session_state['product_info'] = {
    'category': product_category,
    'audience': target_audience,
    'price': price_point,
    'aov': aov,
    'channels': sales_channel,
    'budget': monthly_budget
}

# ============================================================================
# SIDEBAR - DATA UPLOAD
# ============================================================================
st.sidebar.markdown("---")
with st.sidebar:
    st.markdown("### ⚙️ View Mode")
    view_mode = st.radio(
        "Choose your experience:",
        ["Simple", "Advanced"],
        index=0,
        horizontal=True,
        help="Simple mode hides technical details"
    )
st.sidebar.header("📁 Data Input")

# Provide sample CSV download FIRST
st.sidebar.subheader("📥 Need a Template?")

sample_csv = """TV,Billboards,Google_Ads,Social_Media,Influencer_Marketing,Affiliate_Marketing,Product_Sold
25000,15000,20000,18000,12000,10000,7500
30000,10000,25000,20000,15000,12000,8200
20000,18000,15000,22000,10000,15000,7800
35000,12000,18000,16000,14000,11000,8500
22000,20000,22000,25000,8000,13000,7900
"""

st.sidebar.download_button(
    label="📄 Download Sample CSV Template",
    data=sample_csv,
    file_name="marketing_budget_template.csv",
    mime="text/csv",
    help="Download this template and fill in your data"
)

st.sidebar.markdown("---")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload your marketing data (CSV)", 
    type=['csv'],
    help="Upload a CSV with marketing channel spends and sales data"
)

# Option to use sample data
use_sample = st.sidebar.checkbox("Or use sample data", value=True)

# ============================================================================
# SMART DATA LOADING WITH COLUMN MAPPING
# ============================================================================

# Function to detect columns intelligently
def detect_column_mapping(df_columns):
    """
    Try to automatically detect which columns are which
    """
    column_patterns = {
        'TV': ['tv', 'television', 'tv advertising', 'tv ads', 'tv spend', 'tv_spend'],
        'Billboards': ['billboard', 'billboards', 'outdoor', 'outdoor advertising', 'outdoor_advertising'],
        'Google_Ads': ['google', 'google ads', 'google_ads', 'googleads', 'search ads', 'ppc', 'paid search', 'search'],
        'Social_Media': ['social', 'social media', 'social_media', 'socialmedia', 'social ads', 'social_ads'],
        'Influencer_Marketing': ['influencer', 'influencer marketing', 'influencer_marketing', 'influencers', 'influencer_ads'],
        'Affiliate_Marketing': ['affiliate', 'affiliate marketing', 'affiliate_marketing', 'affiliates', 'affiliate_ads'],
        'Product_Sold': ['product_sold', 'sales', 'units sold', 'revenue', 'product sold', 'units', 'sold', 'products_sold']
    }
    
    detected = {}
    used_columns = set()
    
    # Try to match each column
    for standard_name, patterns in column_patterns.items():
        for col in df_columns:
            if col in used_columns:
                continue
            col_lower = col.lower().strip().replace(' ', '_')
            if col_lower in patterns or any(pattern in col_lower for pattern in patterns):
                detected[standard_name] = col
                used_columns.add(col)
                break
    
    return detected

# Load data
if use_sample or uploaded_file is None:
    try:
        df = pd.read_csv('product_advertising.csv')
        st.sidebar.success("✅ Using sample data (300 campaigns)")
        
        # For sample data, we know the columns are correct
        channels = ['TV', 'Billboards', 'Google_Ads', 'Social_Media', 
                   'Influencer_Marketing', 'Affiliate_Marketing']
        sales_column = 'Product_Sold'
        
        # Flag that columns are ready
        st.session_state['columns_ready'] = True
        st.session_state['channels'] = channels
        st.session_state['sales_column'] = sales_column
        st.session_state['df'] = df
        
    except FileNotFoundError:
        st.error("❌ Error: 'product_advertising.csv' not found in the same folder as dashboard.py")
        st.info("💡 Please make sure 'product_advertising.csv' is in the same directory, or upload your own CSV file.")
        st.stop()
else:
    # User uploaded their own CSV
    df = pd.read_csv(uploaded_file)
    st.sidebar.info(f"📊 Loaded {len(df)} rows from your file")
    
    # Try to auto-detect columns
    detected_mapping = detect_column_mapping(df.columns)
    
    # Check if we detected all required columns
    required_channels = ['TV', 'Billboards', 'Google_Ads', 'Social_Media', 
                        'Influencer_Marketing', 'Affiliate_Marketing']
    
    all_detected = all(ch in detected_mapping for ch in required_channels) and 'Product_Sold' in detected_mapping
    
    if all_detected:
        # Success! All columns auto-detected
        st.sidebar.success("✅ Columns auto-detected!")
        
        # Create clean dataframe with standard column names
        df_clean = pd.DataFrame()
        for standard_name, original_name in detected_mapping.items():
            df_clean[standard_name] = pd.to_numeric(df[original_name], errors='coerce')
        
        # Remove rows with NaN values
        df_clean = df_clean.dropna()
        
        if len(df_clean) == 0:
            st.error("❌ No valid numeric data found. Please check your CSV.")
            st.stop()
        
        df = df_clean
        channels = required_channels
        sales_column = 'Product_Sold'
        
        st.session_state['columns_ready'] = True
        st.session_state['channels'] = channels
        st.session_state['sales_column'] = sales_column
        st.session_state['df'] = df
        
        st.sidebar.success(f"✅ {len(df)} valid rows ready!")
        
    else:
        # Need manual mapping
        st.sidebar.warning("⚠️ Please map your columns below")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📋 Column Mapping")
        st.sidebar.write("Match your CSV columns to the required format:")
        
        # Show what was detected
        if detected_mapping:
            st.sidebar.info(f"Auto-detected: {len(detected_mapping)}/7 columns")
        
        # Manual column selection
        available_columns = ['(Select a column)'] + list(df.columns)
        mapping = {}
        
        def get_default_index(standard_name):
            if standard_name in detected_mapping:
                return available_columns.index(detected_mapping[standard_name])
            return 0
        
        mapping['TV'] = st.sidebar.selectbox(
            "TV Advertising:",
            available_columns,
            index=get_default_index('TV'),
            key='map_tv'
        )
        
        mapping['Billboards'] = st.sidebar.selectbox(
            "Billboards:",
            available_columns,
            index=get_default_index('Billboards'),
            key='map_bb'
        )
        
        mapping['Google_Ads'] = st.sidebar.selectbox(
            "Google Ads:",
            available_columns,
            index=get_default_index('Google_Ads'),
            key='map_google'
        )
        
        mapping['Social_Media'] = st.sidebar.selectbox(
            "Social Media:",
            available_columns,
            index=get_default_index('Social_Media'),
            key='map_social'
        )
        
        mapping['Influencer_Marketing'] = st.sidebar.selectbox(
            "Influencer Marketing:",
            available_columns,
            index=get_default_index('Influencer_Marketing'),
            key='map_influencer'
        )
        
        mapping['Affiliate_Marketing'] = st.sidebar.selectbox(
            "Affiliate Marketing:",
            available_columns,
            index=get_default_index('Affiliate_Marketing'),
            key='map_affiliate'
        )
        
        mapping['Product_Sold'] = st.sidebar.selectbox(
            "Sales/Revenue:",
            available_columns,
            index=get_default_index('Product_Sold'),
            key='map_sales'
        )
        
        # Check if all mapped
        if all(v != '(Select a column)' for v in mapping.values()):
            st.sidebar.success("✅ All columns mapped!")
            
            # Create clean dataframe
            df_clean = pd.DataFrame()
            for standard_name, original_name in mapping.items():
                df_clean[standard_name] = pd.to_numeric(df[original_name], errors='coerce')
            
            # Remove rows with NaN values
            df_clean = df_clean.dropna()
            
            if len(df_clean) == 0:
                st.sidebar.error("❌ No valid numeric data after cleaning")
                st.session_state['columns_ready'] = False
                st.stop()
            
            df = df_clean
            channels = required_channels
            sales_column = 'Product_Sold'
            
            st.session_state['columns_ready'] = True
            st.session_state['channels'] = channels
            st.session_state['sales_column'] = sales_column
            st.session_state['df'] = df
            
            st.sidebar.success(f"✅ {len(df)} valid rows ready!")
        else:
            st.sidebar.error("❌ Please select all columns")
            st.session_state['columns_ready'] = False

# Only proceed if columns are ready
if not st.session_state.get('columns_ready', False):
    st.warning("⚠️ Please upload data or map columns in the sidebar to continue")
    st.info("👈 Use the sidebar to either check 'Use sample data' or upload your own CSV file")
    st.stop()

# Get data from session state
df = st.session_state['df']
channels = st.session_state['channels']
sales_column = st.session_state['sales_column']

# Show data preview
with st.expander("📋 View Data Preview"):
    st.dataframe(df.head(10))
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write(f"**Channels:** {', '.join(channels)}")
    st.write(f"**Sales Column:** {sales_column}")

# ============================================================================
# RUN CLUSTERING
# ============================================================================

@st.cache_data
def perform_clustering(data, channel_list, sales_col):
    X = data[channel_list].copy()
    X_pct = X.div(X.sum(axis=1), axis=0) * 100
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pct)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters, X_pct, scaler, kmeans

df['Cluster'], X_pct, scaler, kmeans = perform_clustering(df, channels, sales_column)

# Cluster names
cluster_names = {
    0: "Digital-First Strategy",
    1: "Balanced Mix Strategy", 
    2: "Traditional Media Strategy",
    3: "Social-Heavy Strategy",
    4: "Influencer-Focused Strategy"
}

# Cluster performance
cluster_stats = df.groupby('Cluster')[sales_column].agg(['mean', 'std', 'count'])

# ============================================================================
# PRODUCT-SPECIFIC RECOMMENDATIONS
# ============================================================================

def get_channel_recommendations(product_info):
    """
    Generate product-specific channel recommendations
    """
    category = product_info['category']
    budget = product_info['budget']
    audience = product_info['audience']
    
    # Define product-specific strategies
    strategies = {
        'Sports Apparel': {
            'primary_channels': ['Influencer_Marketing', 'Social_Media', 'Affiliate_Marketing'],
            'rationale': 'Visual product, trust-based purchases, fitness influencer impact',
            'allocation': {
                'TV': 0.04,
                'Billboards': 0.06,
                'Google_Ads': 0.16,
                'Social_Media': 0.24,
                'Influencer_Marketing': 0.30,
                'Affiliate_Marketing': 0.20
            },
            'messaging': ['Authenticity', 'Performance proof', 'User-generated content'],
            'platforms': ['Instagram', 'TikTok', 'YouTube'],
            'expected_roas': 14.5,
            'description': 'Sports apparel buyers trust fitness influencers and respond to visual proof. Focus on authentic user-generated content and performance-based channels.'
        },
        
        'Electronics': {
            'primary_channels': ['Google_Ads', 'Affiliate_Marketing', 'Social_Media'],
            'rationale': 'High research intent, comparison shopping, review-driven',
            'allocation': {
                'TV': 0.15,
                'Billboards': 0.05,
                'Google_Ads': 0.35,
                'Social_Media': 0.15,
                'Influencer_Marketing': 0.05,
                'Affiliate_Marketing': 0.25
            },
            'messaging': ['Technical specs', 'Comparisons', 'Value proposition'],
            'platforms': ['Google Shopping', 'YouTube', 'Review sites'],
            'expected_roas': 10.2,
            'description': 'Electronics buyers do extensive research. Capture high-intent searches and leverage review sites for credibility.'
        },
        
        'Beauty Products': {
            'primary_channels': ['Influencer_Marketing', 'Social_Media', 'TV'],
            'rationale': 'Visual transformation, influencer credibility, aspirational',
            'allocation': {
                'TV': 0.20,
                'Billboards': 0.02,
                'Google_Ads': 0.08,
                'Social_Media': 0.25,
                'Influencer_Marketing': 0.35,
                'Affiliate_Marketing': 0.10
            },
            'messaging': ['Before/after', 'Beauty experts', 'Self-care'],
            'platforms': ['Instagram', 'TikTok', 'YouTube Beauty gurus'],
            'expected_roas': 12.8,
            'description': 'Beauty products sell through visual transformation stories and influencer endorsements. Prioritize Instagram and beauty-focused influencers.'
        },
        
        'Home Goods': {
            'primary_channels': ['TV', 'Social_Media', 'Google_Ads'],
            'rationale': 'Broad appeal, lifestyle-oriented, seasonal patterns',
            'allocation': {
                'TV': 0.25,
                'Billboards': 0.10,
                'Google_Ads': 0.20,
                'Social_Media': 0.20,
                'Influencer_Marketing': 0.10,
                'Affiliate_Marketing': 0.15
            },
            'messaging': ['Home transformation', 'Comfort', 'Value'],
            'platforms': ['Facebook', 'Pinterest', 'Home & Garden shows'],
            'expected_roas': 9.5,
            'description': 'Home goods appeal to broad demographics. Use TV for reach and social media for lifestyle inspiration and seasonal campaigns.'
        },
        
        'Food & Beverage': {
            'primary_channels': ['TV', 'Social_Media', 'Billboards'],
            'rationale': 'Mass appeal, impulse-driven, brand awareness critical',
            'allocation': {
                'TV': 0.30,
                'Billboards': 0.15,
                'Google_Ads': 0.15,
                'Social_Media': 0.25,
                'Influencer_Marketing': 0.10,
                'Affiliate_Marketing': 0.05
            },
            'messaging': ['Taste', 'Convenience', 'Experience'],
            'platforms': ['TV commercials', 'Instagram', 'Food bloggers'],
            'expected_roas': 11.0,
            'description': 'Food & beverage requires broad awareness and impulse triggers. Combine TV reach with social media appetite appeal.'
        },
        
        'Software/SaaS': {
            'primary_channels': ['Google_Ads', 'Affiliate_Marketing', 'Social_Media'],
            'rationale': 'B2B/B2C mix, trial-focused, education-heavy',
            'allocation': {
                'TV': 0.05,
                'Billboards': 0.02,
                'Google_Ads': 0.40,
                'Social_Media': 0.20,
                'Influencer_Marketing': 0.08,
                'Affiliate_Marketing': 0.25
            },
            'messaging': ['Problem-solution', 'ROI', 'Free trial'],
            'platforms': ['Google Search', 'LinkedIn', 'Tech review sites'],
            'expected_roas': 8.5,
            'description': 'SaaS requires targeted, intent-based marketing. Focus on capturing searches and leveraging affiliates/partners for trial signups.'
        },
        
        'Other': {
            'primary_channels': ['Google_Ads', 'Social_Media', 'Affiliate_Marketing'],
            'rationale': 'Balanced digital approach',
            'allocation': {
                'TV': 0.15,
                'Billboards': 0.10,
                'Google_Ads': 0.25,
                'Social_Media': 0.20,
                'Influencer_Marketing': 0.15,
                'Affiliate_Marketing': 0.15
            },
            'messaging': ['Value proposition', 'Differentiation', 'Customer benefits'],
            'platforms': ['Google', 'Facebook', 'Instagram'],
            'expected_roas': 10.0,
            'description': 'A balanced digital-first approach suitable for most products.'
        }
    }
    
    # Get strategy for this product
    strategy = strategies.get(category, strategies['Other']).copy()
    strategy['allocation'] = strategy['allocation'].copy()
    
    # Adjust based on budget size
    if budget < 25000:
        strategy['allocation']['TV'] = max(0.02, strategy['allocation']['TV'] - 0.10)
        strategy['allocation']['Influencer_Marketing'] += 0.05
        strategy['allocation']['Affiliate_Marketing'] += 0.05
    elif budget > 100000:
        strategy['allocation']['TV'] = min(0.40, strategy['allocation']['TV'] + 0.10)
    
    # Adjust based on audience
    if 'Young Adults (18-34)' in audience or 'Teens (13-17)' in audience:
        strategy['allocation']['Social_Media'] += 0.05
        strategy['allocation']['TV'] = max(0.02, strategy['allocation']['TV'] - 0.05)
    
    if 'Seniors (55+)' in audience:
        strategy['allocation']['TV'] += 0.08
        strategy['allocation']['Social_Media'] = max(0.05, strategy['allocation']['Social_Media'] - 0.05)
    
    # Normalize to ensure it sums to 1.0
    total = sum(strategy['allocation'].values())
    strategy['allocation'] = {k: v/total for k, v in strategy['allocation'].items()}
    
    return strategy

# ============================================================================
# CREATE TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Your Strategy", 
    "🎯 Campaign Builder", 
    "💰 Budget Optimizer", 
    "📈 Performance", 
    "💬 Ask AI"
])

# ============================================================================
# TAB 1: YOUR STRATEGY
# ============================================================================

with tab1:
    st.header("🎯 Your Marketing Strategy Profile")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Your Budget")
        
        tv_input = st.number_input("TV ($)", 0, 100000, 25000, 1000, key='input_tv')
        billboard_input = st.number_input("Billboards ($)", 0, 100000, 15000, 1000, key='input_bb')
        google_input = st.number_input("Google Ads ($)", 0, 100000, 20000, 1000, key='input_google')
        social_input = st.number_input("Social Media ($)", 0, 100000, 18000, 1000, key='input_social')
        influencer_input = st.number_input("Influencer Marketing ($)", 0, 100000, 12000, 1000, key='input_influencer')
        affiliate_input = st.number_input("Affiliate Marketing ($)", 0, 100000, 10000, 1000, key='input_affiliate')
        
        total_budget = tv_input + billboard_input + google_input + social_input + influencer_input + affiliate_input
        
        st.metric("Total Budget", f"${total_budget:,.0f}")
    
    with col2:
        st.subheader("Your Strategy Analysis")
        
        if total_budget == 0:
            st.warning("Please enter your budget amounts on the left")
        else:
            # Calculate user's cluster
            user_budget = np.array([[tv_input, billboard_input, google_input, 
                                    social_input, influencer_input, affiliate_input]])
            user_pct = (user_budget / user_budget.sum()) * 100
            
            # Scale and predict
            user_scaled = scaler.transform(user_pct)
            user_cluster = kmeans.predict(user_scaled)[0]
            
            # Display results
            st.success(f"**Your Strategy Type:** {cluster_names[user_cluster]}")
            
            avg_sales = cluster_stats.loc[user_cluster, 'mean']
            std_sales = cluster_stats.loc[user_cluster, 'std']
            n_campaigns = cluster_stats.loc[user_cluster, 'count']
            
            st.metric("Expected Sales", f"{avg_sales:.0f} units", 
                     help=f"Based on {n_campaigns:.0f} similar campaigns")
            st.metric("Typical Range", f"±{std_sales:.0f} units",
                     help="Standard deviation shows result consistency")
            
            # Compare to best
            best_cluster = cluster_stats['mean'].idxmax()
            best_sales = cluster_stats.loc[best_cluster, 'mean']
            
            if user_cluster != best_cluster:
                improvement = ((best_sales - avg_sales) / avg_sales) * 100
                st.warning(f"💡 **Opportunity:** {cluster_names[best_cluster]} averages {best_sales:.0f} units ({improvement:.1f}% better)")
                st.info("👉 Check the **Campaign Builder** tab for product-specific recommendations!")
            else:
                st.success("🎉 You're already using the top-performing strategy!")

# ============================================================================
# TAB 2: CAMPAIGN BUILDER
# ============================================================================

with tab2:
    st.header("🎯 AI Campaign Builder")
    
    if 'product_info' in st.session_state:
        product_info = st.session_state['product_info']
        
        # Show product summary
        st.subheader(f"📦 Campaign for: {product_info['category']}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Target Audience", f"{len(product_info['audience'])} segments" if product_info['audience'] else "Not specified")
        col2.metric("Price Point", product_info['price'])
        col3.metric("Average Order Value", f"${product_info['aov']}")
        
        st.markdown("---")
        
        # Get recommendations
        strategy = get_channel_recommendations(product_info)
        
        # Show strategy overview
        st.subheader("🎯 Recommended Strategy")
        st.info(f"**{strategy['rationale']}**")
        st.write(strategy['description'])
        
        # Show allocation
        st.subheader("💰 Recommended Budget Allocation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            allocation_df = pd.DataFrame({
                'Channel': channels,
                'Allocation (%)': [strategy['allocation'][ch] * 100 for ch in channels],
                'Budget ($)': [strategy['allocation'][ch] * product_info['budget'] for ch in channels]
            })
            allocation_df = allocation_df.sort_values('Budget ($)', ascending=False)
            
            st.dataframe(allocation_df.style.format({
                'Allocation (%)': '{:.1f}%',
                'Budget ($)': '${:,.0f}'
            }), use_container_width=True)
            
            # Show primary channels
            st.markdown("**Primary Focus Channels:**")
            for ch in strategy['primary_channels']:
                pct = strategy['allocation'][ch] * 100
                st.write(f"✅ **{ch}**: {pct:.1f}% (${strategy['allocation'][ch] * product_info['budget']:,.0f})")
        
        with col2:
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4', '#DDA15E']
            wedges, texts, autotexts = ax.pie(
                allocation_df['Allocation (%)'],
                labels=allocation_df['Channel'],
                autopct='%1.1f%%',
                startangle=90,
                colors=colors
            )
            ax.set_title(f"Budget Mix for {product_info['category']}", fontsize=14, fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            st.pyplot(fig)
            plt.close()
        
        # Expected performance
        st.subheader("📈 Expected Performance")
        
        col1, col2, col3 = st.columns(3)
        
        expected_sales = strategy['expected_roas'] * product_info['budget'] / product_info['aov']
        expected_revenue = expected_sales * product_info['aov']
        
        col1.metric("Expected Sales", f"{expected_sales:,.0f} units")
        col2.metric("Expected Revenue", f"${expected_revenue:,.0f}")
        col3.metric("Expected ROAS", f"{strategy['expected_roas']:.1f}x")
        
        # Messaging strategy
        st.subheader("💬 Messaging & Creative Strategy")
        st.write("**Recommended Message Angles:**")
        for msg in strategy['messaging']:
            st.write(f"• {msg}")
        
        st.write("**Recommended Platforms:**")
        for platform in strategy['platforms']:
            st.write(f"• {platform}")
        
        # Comparison to generic approach
        st.markdown("---")
        st.subheader("📊 Why Product-Specific Matters")
        
        generic_sales = 6500
        
        comparison_df = pd.DataFrame({
            'Approach': ['Generic (One-Size-Fits-All)', f'Optimized for {product_info["category"]}'],
            'Top Channel Focus': ['Equal across all', f'{strategy["primary_channels"][0]}'],
            'Expected Sales': [generic_sales, expected_sales],
            'Strategy': ['Spreads budget evenly', strategy['rationale']]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        improvement_vs_generic = ((expected_sales - generic_sales) / generic_sales) * 100
        if improvement_vs_generic > 0:
            st.success(f"🎉 **Product-specific approach shows {improvement_vs_generic:.1f}% improvement over generic strategy!**")
    else:
        st.info("👈 Please fill out product information in the sidebar first!")

# ============================================================================
# TAB 3: BUDGET OPTIMIZER
# ============================================================================

with tab3:
    st.header("🔮 AI Budget Optimizer")
    st.info("💡 Adjust the sliders to see how different budget splits could impact your results.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Adjust Budget Allocation")
        tv_opt = st.slider("TV Budget ($)", 0, 100000, tv_input, 1000, key='tv_opt')
        billboard_opt = st.slider("Billboards ($)", 0, 100000, billboard_input, 1000, key='bb_opt')
        google_opt = st.slider("Google Ads ($)", 0, 100000, google_input, 1000, key='g_opt')
        social_opt = st.slider("Social Media ($)", 0, 100000, social_input, 1000, key='s_opt')
        influencer_opt = st.slider("Influencer ($)", 0, 100000, influencer_input, 1000, key='i_opt')
        affiliate_opt = st.slider("Affiliate ($)", 0, 100000, affiliate_input, 1000, key='a_opt')
        total_opt = tv_opt + billboard_opt + google_opt + social_opt + influencer_opt + affiliate_opt
        st.metric("New Total Budget", f"${total_opt:,.0f}")
    
    with col2:
        st.subheader("Predicted Outcome")
        if total_opt == 0:
            st.warning("Please adjust budget sliders on the left")
        else:
            # Calculate new cluster
            new_budget = np.array([[tv_opt, billboard_opt, google_opt, 
                                    social_opt, influencer_opt, affiliate_opt]])
            new_pct = (new_budget / new_budget.sum()) * 100
            new_scaled = scaler.transform(new_pct)
            new_cluster = kmeans.predict(new_scaled)[0]
            new_avg_sales = cluster_stats.loc[new_cluster, 'mean']
            
            st.success(f"**New Strategy:** {cluster_names[new_cluster]}")
            sales_change = new_avg_sales - avg_sales
            pct_change = (sales_change / avg_sales) * 100 if avg_sales != 0 else 0
            st.metric("Predicted Sales", 
                      f"{new_avg_sales:.0f} units",
                      f"{sales_change:+.0f} ({pct_change:+.1f}%)")
            
            # ADVANCED MODE ONLY - Show detailed breakdown
            if view_mode == "Advanced":
                st.subheader("Budget Changes")
                budget_df = pd.DataFrame({
                    'Channel': channels,
                    'Current ($)': [tv_input, billboard_input, google_input, 
                                   social_input, influencer_input, affiliate_input],
                    'Optimized ($)': [tv_opt, billboard_opt, google_opt,
                                     social_opt, influencer_opt, affiliate_opt],
                    'Change ($)': [tv_opt - tv_input, billboard_opt - billboard_input,
                                  google_opt - google_input, social_opt - social_input,
                                  influencer_opt - influencer_input, affiliate_opt - affiliate_input]
                })
                st.dataframe(budget_df.style.background_gradient(subset=['Change ($)'], cmap='RdYlGn'))
# ============================================================================
# TAB 4: PERFORMANCE ANALYSIS
# ============================================================================

with tab4:
    st.header("📈 Detailed Performance Analysis")
    st.info("💡 This shows how your current strategy compares to other successful approaches.")
    st.subheader("Cluster Performance Summary")
    
    # Create summary table - EVERYONE SEES THIS
    summary_df = pd.DataFrame({
        'Strategy': [cluster_names[i] for i in range(4)],
        'Campaigns': cluster_stats['count'].values.astype(int),
        'Avg Sales': cluster_stats['mean'].values.astype(int),
        'Std Dev': cluster_stats['std'].values.astype(int),
    })
    st.dataframe(summary_df, use_container_width=True)
    
    # ADVANCED MODE ONLY - Charts and detailed breakdowns
    if view_mode == "Advanced":
        # Visualize cluster performance
        st.subheader("📊 Strategy Performance Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of average sales by cluster
            fig, ax = plt.subplots(figsize=(10, 6))
            colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
            bars = ax.bar(range(4), cluster_stats['mean'].values, 
                         color=colors_list, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Cluster', fontsize=12)
            ax.set_ylabel('Average Sales (units)', fontsize=12)
            ax.set_title('Average Sales by Strategy Type', fontsize=14, fontweight='bold')
            ax.set_xticks(range(4))
            ax.set_xticklabels([f'C{i}' for i in range(4)])
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Pie chart of cluster distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            cluster_counts = df['Cluster'].value_counts().sort_index()
            ax.pie(cluster_counts.values, labels=[f'Cluster {i}' for i in range(4)],
                   autopct='%1.1f%%', startangle=90, colors=colors_list)
            ax.set_title('Campaign Distribution by Strategy', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Show detailed cluster breakdown
        st.subheader("🔍 Detailed Cluster Breakdown")
        for i in range(4):
            with st.expander(f"📊 {cluster_names[i]} - Cluster {i}"):
                cluster_data = df[df['Cluster'] == i]
                col1, col2, col3 = st.columns(3)
                col1.metric("Campaigns", f"{len(cluster_data)}")
                col2.metric("Avg Sales", f"{cluster_data[sales_column].mean():.0f} units")
                col3.metric("Sales Range", f"{cluster_data[sales_column].min():.0f} - {cluster_data[sales_column].max():.0f}")
                st.write("**Average Channel Spend:**")
                avg_spend = cluster_data[channels].mean()
                spend_df = pd.DataFrame({
                    'Channel': channels,
                    'Avg Spend ($)': avg_spend.values,
                    '% of Total': (avg_spend / avg_spend.sum() * 100).values
                })
                st.dataframe(spend_df.style.format({
                    'Avg Spend ($)': '${:,.0f}',
                    '% of Total': '{:.1f}%'
                }), use_container_width=True)
        
        # Show existing visualizations if available
        st.markdown("---")
        st.subheader("📈 Additional Analysis Visualizations")
        viz_files = {
            "Spend Distributions": "viz1_spend_distributions.png",
            "Correlation Analysis": "viz2_correlation_and_trends.png",
            "Cluster Results": "viz4_kmeans_results.png",
            "PCA Analysis": "viz5_pca_analysis.png"
        }
        viz_option = st.selectbox("Select Visualization", list(viz_files.keys()))
        try:
            st.image(viz_files[viz_option], use_container_width=True)
        except:
            st.info(f"💡 Visualization '{viz_files[viz_option]}' not found. Run your analysis script to generate visualizations.")
    
    else:
        # SIMPLE MODE - Just show a tip
        st.info("💡 Switch to **Advanced** mode in the sidebar to see detailed charts and breakdowns.")

# ============================================================================
# TAB 5: AI ADVISOR (LLM Integration)
# ============================================================================

with tab5:
    st.header("💬 Ask Your AI Marketing Advisor")
    
    st.write("Get personalized explanations and recommendations based on your strategy and the data.")
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("❓ Why this strategy?"):
            st.session_state['suggested_question'] = "Why is the recommended strategy best for my product?"
    
    with col2:
        if st.button("📊 Compare strategies"):
            st.session_state['suggested_question'] = "Compare the different strategy archetypes and their performance"
    
    with col3:
        if st.button("📝 Generate brief"):
            st.session_state['suggested_question'] = "Create a campaign brief I can present to my team"
    
    st.markdown("---")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle suggested questions
    default_prompt = ""
    if 'suggested_question' in st.session_state:
        default_prompt = st.session_state['suggested_question']
        del st.session_state['suggested_question']
    
    # User input
    prompt = st.chat_input("Ask me anything about your marketing strategy...", key="chat_input")
    
    if prompt or default_prompt:
        user_question = prompt if prompt else default_prompt
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Build context from current state
        product_info = st.session_state.get('product_info', {})
        
        # Check if user has entered budget
        has_budget = 'user_cluster' in locals() and total_budget > 0
        
        context = f"""You are an expert marketing advisor helping a user optimize their advertising budget.

USER'S CONTEXT:
- Product Category: {product_info.get('category', 'Not specified')}
- Target Audience: {', '.join(product_info.get('audience', [])) if product_info.get('audience') else 'Not specified'}
- Price Point: {product_info.get('price', 'Not specified')}
- Average Order Value: ${product_info.get('aov', 0)}
- Monthly Budget: ${product_info.get('budget', 0):,}

DATA ANALYSIS FINDINGS:
- Analyzed 300 real marketing campaigns
- 4 strategy archetypes identified:
  * Cluster 0: Influencer-Heavy (6,870 units avg)
  * Cluster 1: Billboard-Focused (6,956 units avg)
  * Cluster 2: Google Ads-Dominant (6,582 units avg - WORST)
  * Cluster 3: Affiliate-Focused (7,774 units avg - BEST, 18% better than worst)
- Key finding: TV and Affiliate Marketing are the primary strategic differentiators
- Product-specific strategies vary significantly by category

USER QUESTION: {user_question}

Provide a helpful, conversational response in 2-3 paragraphs. Be specific, reference the data findings when relevant, 
and give actionable advice. Use plain language that a marketer without ML background would understand."""

# AI Advisor Tab
st.header("💬 Ask Your AI Marketing Advisor")
st.write("Get personalized explanations and recommendations based on your strategy and the data.")

# Clear button at the top
if st.button("🗑️ Clear conversation"):
    st.session_state.chat_history = []
    st.rerun()



# Initialize conversation history (put this near the top of your file with other session state)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Call Claude API
try:
    import anthropic
    client = anthropic.Anthropic(
        api_key=st.secrets["ANTHROPIC_API_KEY"]
    )
    
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": context})
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Loading spinner while waiting for first token
        with st.spinner("🧠 Analyzing your marketing data..."):
            with client.messages.stream(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                system=(
                    "You are a friendly marketing advisor for beginners. "
                    "IMPORTANT RULES: "
                    "1. Never use technical terms like 'cluster', 'PCA', 'K-means', or variable names like 'tv_input'. "
                    "2. Use simple language a small business owner would understand. "
                    "3. Instead of 'Cluster 2', say the strategy name like 'Balanced Strategy'. "
                    "4. Instead of variable names, say 'your TV budget' or 'your social media spend'. "
                    "5. Focus on actionable advice, not data analysis. "
                    "6. Use headings with ### and bullet points. "
                    "7. Keep paragraphs short (2-3 sentences max). "
                    "8. Be encouraging and supportive."
                ),
                messages=st.session_state.chat_history
            ) as stream:
                buffer = ""
                buffer_count = 0
                for text in stream.text_stream:
                    buffer += text
                    buffer_count += 1
                    if buffer_count >= 5:
                        full_response += buffer
                        message_placeholder.markdown(full_response + "▌")
                        buffer = ""
                        buffer_count = 0
                full_response += buffer
        
        message_placeholder.markdown(full_response)
        
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

except KeyError:
    st.error("❌ API key not found!")
    st.info("💡 Create a `.streamlit/secrets.toml` file with:")
    st.code('ANTHROPIC_API_KEY = "sk-ant-..."')

except Exception as e:
    st.error(f"Error: {e}")
    st.info("💡 Check that your API key is valid at https://console.anthropic.com/")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>📊 Marketing Budget Optimizer | Built with Streamlit & scikit-learn</p>
        <p><small>Based on analysis of 300 marketing campaigns using K-Means clustering and product-specific recommendations</small></p>
    </div>
""", unsafe_allow_html=True)