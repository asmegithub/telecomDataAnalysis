def top_10_handsets(df):
    print("Top 10 handsets:")
    top_handsets = df['Handset Type'].value_counts().head(10)
    return top_handsets

def get_top_3_manufacturers(df):
    """
    Identifies the top 3 handset manufacturers based on frequency in the dataset.

    Args:
        df (DataFrame): The dataset containing the 'Handset Manufacturer' column.

    Returns:
        Series: A Pandas Series containing the top 3 manufacturers and their counts.
    """
    return df['Handset Manufacturer'].value_counts().head(3)

def top_5_handsets_per_manufacturer(df, top_manufacturers):
    results = {}
    for manufacturer in top_manufacturers.index:
        handsets = (
            df[df['Handset Manufacturer'] == manufacturer]['Handset Type']
            .value_counts()
            .head(5)
        )
        results[manufacturer] = handsets
    return results

def top_3_manufacturers(df):
    top_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
    return top_manufacturers

def interpret_and_recommend(top_handsets, top_manufacturers, handsets_per_manufacturer):
    insights = []
    insights.append(f"Top 10 handsets indicate customer preferences for {top_handsets.index.tolist()}.")
    insights.append(f"Top 3 manufacturers dominate the market: {top_manufacturers.index.tolist()}.")
    insights.append("Key handsets driving engagement per manufacturer are:")
    for manufacturer, handsets in handsets_per_manufacturer.items():
        insights.append(f"- {manufacturer}: {handsets.index.tolist()}")
    insights.append("Recommendations:")
    insights.append("- Target marketing campaigns towards customers using popular handsets.")
    insights.append("- Partner with top manufacturers to boost sales and customer engagement.")
    return insights

def interpret_and_recommend(top_handsets, top_manufacturers, handsets_per_manufacturer):
    insights = []
    insights.append(f"Top 10 handsets indicate customer preferences for {top_handsets.index.tolist()}.")
    insights.append(f"Top 3 manufacturers dominate the market: {top_manufacturers.index.tolist()}.")
    insights.append("Key handsets driving engagement per manufacturer are:")
    for manufacturer, handsets in handsets_per_manufacturer.items():
        insights.append(f"- {manufacturer}: {handsets.index.tolist()}")
    insights.append("Recommendations:")
    insights.append("- Target marketing campaigns towards customers using popular handsets.")
    insights.append("- Partner with top manufacturers to boost sales and customer engagement.")
    return insights
