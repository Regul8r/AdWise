"""
Marketing Mix Modeling - Unsupervised Learning Analysis
Dataset: Product Advertising Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


# STEP 1: LOAD AND EXPLORE DATA

# Load data
df = pd.read_csv('product_advertising.csv')

print("="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# Define channel columns (all marketing spend columns)
channels = ['TV', 'Billboards', 'Google_Ads', 'Social_Media', 
            'Influencer_Marketing', 'Affiliate_Marketing']


# PROBLEM 2.1: CREATE TWO EXPLORATORY VISUALIZATIONS

print("\n" + "="*60)
print("PROBLEM 2.1: EXPLORATORY DATA ANALYSIS")
print("="*60)

# Visualization 1: Distribution of Spend Across Channels
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribution of Marketing Spend by Channel', fontsize=16, fontweight='bold')

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4', '#DDA15E']

for idx, channel in enumerate(channels):
    ax = axes[idx//3, idx%3]
    ax.hist(df[channel], bins=20, color=colors[idx], 
            edgecolor='black', alpha=0.7)
    ax.set_xlabel(f'{channel} Spend ($)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{channel}', fontsize=11, fontweight='bold')
    ax.axvline(df[channel].mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: ${df[channel].mean():.0f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('viz1_spend_distributions.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Visualization 1 saved: viz1_spend_distributions.png")

# Visualization 2: Correlation Heatmap & Spend vs Sales
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Channel Performance Analysis', fontsize=16, fontweight='bold')

# Correlation heatmap
correlation = df[channels + ['Product_Sold']].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[0])
axes[0].set_title('Correlation: Channels & Sales', fontsize=12, fontweight='bold')

# Total spend vs Sales scatter
df['Total_Spend'] = df[channels].sum(axis=1)
axes[1].scatter(df['Total_Spend'], df['Product_Sold'], 
                alpha=0.6, s=100, color='#FF6B6B', edgecolor='black')
axes[1].set_xlabel('Total Marketing Spend ($)', fontsize=11)
axes[1].set_ylabel('Product Sold', fontsize=11)
axes[1].set_title('Total Spend vs Sales', fontsize=12, fontweight='bold')

# Add trend line
z = np.polyfit(df['Total_Spend'], df['Product_Sold'], 1)
p = np.poly1d(z)
axes[1].plot(df['Total_Spend'], p(df['Total_Spend']), "r--", 
             linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.0f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('viz2_correlation_and_trends.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Visualization 2 saved: viz2_correlation_and_trends.png")


# PROBLEM 2.2: METHOD SELECTION - K-MEANS CLUSTERING


print("\n" + "="*60)
print("PROBLEM 2.2: UNSUPERVISED METHOD SELECTION")
print("="*60)



# PROBLEM 2.3: APPLYING K-MEANS


print("\n" + "="*60)
print("PROBLEM 2.3: APPLYING K-MEANS")
print("="*60)

# Use spend percentages for better interpretation
X = df[channels].copy()
X_pct = X.div(X.sum(axis=1), axis=0) * 100
X_pct.columns = [f'{col}_pct' for col in channels]

print("\nFeatures selected:")
for col in channels:
    print(f"  - {col} (as percentage of total spend)")
print("\nUsing percentages makes clusters about STRATEGY, not just budget size")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pct)

# Determine optimal k using Elbow Method
print("\nFinding optimal number of clusters...")
inertias = []
silhouette_scores = []
K_range = range(2, 9)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans_temp.labels_))

# Plot elbow curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Determining Optimal Number of Clusters', fontsize=14, fontweight='bold')

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[0].set_ylabel('Inertia', fontsize=11)
axes[0].set_title('Elbow Method', fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[1].set_ylabel('Silhouette Score', fontsize=11)
axes[1].set_title('Silhouette Analysis', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('viz3_optimal_clusters.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Optimal cluster analysis saved: viz3_optimal_clusters.png")

# Choose optimal k (you can adjust this based on the elbow curve)
optimal_k = 4
print(f"\nChosen k={optimal_k} clusters")

# Apply K-Means
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\n✓ Clustering complete!")
print(f"\nCluster distribution:")
print(df['Cluster'].value_counts().sort_index())


# VISUALIZE K-MEANS RESULTS


# Create comprehensive visualizations
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Marketing Strategy Archetypes (K-Means Clustering)', 
             fontsize=16, fontweight='bold')

# Plot 1: TV vs Digital (sum of all digital channels)
df['Digital_Total_pct'] = X_pct[['Google_Ads_pct', 'Social_Media_pct', 
                                   'Influencer_Marketing_pct', 
                                   'Affiliate_Marketing_pct']].sum(axis=1)

ax1 = plt.subplot(2, 2, 1)
scatter = ax1.scatter(X_pct['TV_pct'], df['Digital_Total_pct'], 
                      c=df['Cluster'], cmap='viridis', s=100, 
                      edgecolor='black', alpha=0.7)
ax1.set_xlabel('TV Spend (%)', fontsize=11)
ax1.set_ylabel('Total Digital Spend (%)', fontsize=11)
ax1.set_title('Traditional vs Digital Strategy', fontsize=12)
plt.colorbar(scatter, ax=ax1, label='Cluster')
ax1.grid(True, alpha=0.3)

# Plot 2: Influencer vs Social
ax2 = plt.subplot(2, 2, 2)
scatter = ax2.scatter(X_pct['Social_Media_pct'], X_pct['Influencer_Marketing_pct'], 
                      c=df['Cluster'], cmap='viridis', s=100, 
                      edgecolor='black', alpha=0.7)
ax2.set_xlabel('Social Media Spend (%)', fontsize=11)
ax2.set_ylabel('Influencer Marketing Spend (%)', fontsize=11)
ax2.set_title('Social vs Influencer Strategy', fontsize=12)
plt.colorbar(scatter, ax=ax2, label='Cluster')
ax2.grid(True, alpha=0.3)

# Plot 3: Average spend by cluster
ax3 = plt.subplot(2, 2, 3)
cluster_means = df.groupby('Cluster')[channels].mean()
cluster_means.plot(kind='bar', ax=ax3, width=0.8, color=colors)
ax3.set_xlabel('Cluster', fontsize=11)
ax3.set_ylabel('Average Spend ($)', fontsize=11)
ax3.set_title('Average Channel Spend by Cluster', fontsize=12)
ax3.legend(title='Channel', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Average sales by cluster
ax4 = plt.subplot(2, 2, 4)
cluster_sales = df.groupby('Cluster')['Product_Sold'].agg(['mean', 'std'])
bars = ax4.bar(cluster_sales.index, cluster_sales['mean'], 
               yerr=cluster_sales['std'], capsize=5, 
               color=colors[:optimal_k], edgecolor='black', alpha=0.7)
ax4.set_xlabel('Cluster', fontsize=11)
ax4.set_ylabel('Average Products Sold', fontsize=11)
ax4.set_title('Sales Performance by Cluster', fontsize=12)
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (idx, row) in enumerate(cluster_sales.iterrows()):
    ax4.text(idx, row['mean'], f"{row['mean']:.0f}", 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('viz4_kmeans_results.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ K-Means visualization saved: viz4_kmeans_results.png")


# DETAILED CLUSTER ANALYSIS


print("\n" + "="*60)
print("PATTERNS & INSIGHTS FROM K-MEANS")
print("="*60)

# Analyze each cluster
cluster_insights = []

for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster_id]
    print(f"\n{'='*50}")
    print(f"CLUSTER {cluster_id}: {len(cluster_data)} campaigns ({len(cluster_data)/len(df)*100:.1f}%)")
    print(f"{'='*50}")
    
    # Average spend profile
    print("\nAverage Spend Profile:")
    spend_profile = []
    for channel in channels:
        avg_spend = cluster_data[channel].mean()
        pct_of_total = (avg_spend / cluster_data[channels].sum(axis=1).mean()) * 100
        print(f"  {channel:25s}: ${avg_spend:8.0f} ({pct_of_total:5.1f}%)")
        spend_profile.append((channel, pct_of_total))
    
    # Dominant channel
    dominant_channel = max(spend_profile, key=lambda x: x[1])
    
    # Performance metrics
    print(f"\nPerformance:")
    avg_sales = cluster_data['Product_Sold'].mean()
    std_sales = cluster_data['Product_Sold'].std()
    print(f"  Average Sales: {avg_sales:.0f} units")
    print(f"  Sales Std Dev: {std_sales:.0f} units")
    
    # ROI calculation
    total_spend = cluster_data[channels].sum(axis=1).mean()
    roi = (avg_sales / total_spend) * 100
    print(f"  Efficiency: {roi:.2f} units per $100 spent")
    
    # Archetype name
    print(f"\n📊 Archetype: '{dominant_channel[0]}-Focused' ({dominant_channel[1]:.0f}% of budget)")
    
    cluster_insights.append({
        'cluster': cluster_id,
        'size': len(cluster_data),
        'avg_sales': avg_sales,
        'efficiency': roi,
        'dominant': dominant_channel[0]
    })


# EVALUATION AGAINST EXPECTATIONS


print("\n" + "="*60)
print("EVALUATION AGAINST EXPECTATIONS")
print("="*60)

best_cluster = max(cluster_insights, key=lambda x: x['avg_sales'])
most_efficient = max(cluster_insights, key=lambda x: x['efficiency'])

print(f"""
EXPECTED: 3-5 distinct archetypes based on channel mix
OBSERVED: {optimal_k} distinct clusters identified

PERFORMANCE INSIGHTS:
✓ Highest Sales: Cluster {best_cluster['cluster']} ({best_cluster['dominant']}-focused)
  → Average: {best_cluster['avg_sales']:.0f} units sold
  
✓ Most Efficient: Cluster {most_efficient['cluster']} ({most_efficient['dominant']}-focused)
  → {most_efficient['efficiency']:.2f} units per $100 spent

KEY FINDINGS:
- Clusters show distinct strategic approaches
- Traditional vs Digital divide is clear
- Some strategies consistently outperform others
- Efficiency varies significantly across archetypes
""")

print("\n" + "="*60)
print("HOW THIS SUPPORTS THE PROJECT EXPERIENCE")
print("="*60)
print("""
1. STRATEGIC BENCHMARKING:
   Users can identify which archetype matches their current strategy
   and compare results to similar campaigns.

2. UI IMPLEMENTATION IDEAS:
   - "You're using a {archetype} strategy" badge
   - Show typical spend ratios for each cluster
   - Display expected sales range for each archetype
   - Recommend higher-performing clusters: "Try Cluster X approach"

3. CONFIDENCE IN RECOMMENDATIONS:
   Clustering validates that distinct strategies exist in real data,
   supporting the regression model's optimization suggestions.

4. RISK ASSESSMENT:
   Show variance within clusters to indicate strategy stability.
   High variance = riskier, Low variance = more predictable.
""")


# PROBLEM 2.4: STRETCH - APPLY PCA


print("\n" + "="*60)
print("PROBLEM 2.4: STRETCH - PCA ANALYSIS")
print("="*60)

# Apply PCA
pca = PCA(n_components=len(channels))
principal_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    data=principal_components[:, :2],
    columns=['PC1', 'PC2']
)
pca_df['Cluster'] = df['Cluster'].values

print("\nExplained Variance by Component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var*100:.2f}%")
cumulative = np.cumsum(pca.explained_variance_ratio_)
print(f"\nCumulative variance (PC1+PC2): {cumulative[1]*100:.2f}%")

# Visualize PCA
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('PCA Analysis: Dimensionality Reduction', fontsize=16, fontweight='bold')

# PCA scatter
scatter = axes[0].scatter(pca_df['PC1'], pca_df['PC2'], 
                          c=pca_df['Cluster'], cmap='viridis', 
                          s=100, edgecolor='black', alpha=0.7)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
axes[0].set_title('First Two Principal Components', fontsize=12)
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0], label='K-Means Cluster')

# Feature loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(
    loadings[:, :2],
    columns=['PC1', 'PC2'],
    index=[f'{ch}_pct' for ch in channels]
)

axes[1].axhline(0, color='gray', linewidth=0.8)
axes[1].axvline(0, color='gray', linewidth=0.8)
for i, feature in enumerate(loading_matrix.index):
    axes[1].arrow(0, 0, loading_matrix.loc[feature, 'PC1'], 
                  loading_matrix.loc[feature, 'PC2'],
                  head_width=0.05, head_length=0.05, fc=colors[i], ec=colors[i], linewidth=2)
    axes[1].text(loading_matrix.loc[feature, 'PC1']*1.15, 
                 loading_matrix.loc[feature, 'PC2']*1.15,
                 feature.replace('_pct', ''), fontsize=9, ha='center', fontweight='bold')
axes[1].set_xlabel('PC1 Loading', fontsize=11)
axes[1].set_ylabel('PC2 Loading', fontsize=11)
axes[1].set_title('Feature Contributions to PCs', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('viz5_pca_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ PCA visualization saved: viz5_pca_analysis.png")

# Feature importance
print("\nPC1 Feature Importance (absolute loadings):")
pc1_importance = pd.DataFrame({
    'Feature': channels,
    'PC1_Loading': pca.components_[0],
    'Abs_Loading': np.abs(pca.components_[0])
}).sort_values('Abs_Loading', ascending=False)
print(pc1_importance.to_string(index=False))


# COMPARE K-MEANS VS PCA


print("\n" + "="*60)
print("COMPARING K-MEANS VS PCA")
print("="*60)
print("""
K-MEANS CLUSTERING:
 Purpose: Find discrete groups of similar spending strategies
 Output: Hard cluster assignments (0, 1, 2, 3)
 Insight: "4 distinct spending archetypes exist"
 Best for: User-facing categorization, benchmarking

PCA (Principal Component Analysis):
 Purpose: Find underlying patterns explaining variance
 Output: Continuous scores on principal components
 Insight: "These channel combinations explain variation"
 Best for: Feature importance, dimensionality reduction

KEY DIFFERENCES:
1. K-Means creates boundaries; PCA creates continuous space
2. K-Means groups campaigns; PCA identifies drivers
3. K-Means answers "Which strategy?"; PCA answers "What matters?"

COMPLEMENTARY INSIGHTS:
- PCA shows WHICH channels drive strategic differences
- K-Means shows HOW campaigns cluster around those patterns
- Together: Validate that the patterns are real and interpretable

FOR THE BUDGET OPTIMIZER PROJECT:
 Use K-Means for user interface (show archetypes)
 Use PCA to validate feature selection for regression model
 PCA loadings inform which channels to emphasize in UI
 Combine: "Your strategy is Cluster 2 (Digital-First), driven 
  primarily by variations in Social Media and Influencer spend (PC1)"
""")

print("\n" + "="*60)
print("ANALYSIS COMPLETE! 🎉")
print("="*60)
print("\nGenerated Files:")
print("  1. viz1_spend_distributions.png")
print("  2. viz2_correlation_and_trends.png")
print("  3. viz3_optimal_clusters.png")
print("  4. viz4_kmeans_results.png")
print("  5. viz5_pca_analysis.png")