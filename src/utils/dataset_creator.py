import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def create_sample_dataset(size=1000, output_path="data/social_media_data.csv"):
    """
    Create a sample social media dataset for sentiment analysis.
    The dataset includes realistic social media posts about various tech products.
    """

    # Seed for reproducibility
    np.random.seed(42)

    # Sample positive phrases
    positive_phrases = [
        "Love this product! {emoji}",
        "Amazing customer service {emoji}",
        "Best purchase ever {emoji}",
        "Highly recommend {emoji}",
        "Works perfectly {emoji}",
        "Great value for money {emoji}",
        "Excellent quality {emoji}",
        "Very satisfied {emoji}",
        "Fantastic features {emoji}",
        "Impressive performance {emoji}",
    ]

    # Sample negative phrases
    negative_phrases = [
        "Terrible product {emoji}",
        "Poor customer service {emoji}",
        "Waste of money {emoji}",
        "Would not recommend {emoji}",
        "Doesn't work properly {emoji}",
        "Very disappointed {emoji}",
        "Poor quality {emoji}",
        "Not worth the price {emoji}",
        "Frustrating experience {emoji}",
        "Many issues {emoji}",
    ]

    # Sample neutral phrases
    neutral_phrases = [
        "Average product {emoji}",
        "Okay for the price {emoji}",
        "Could be better {emoji}",
        "Not bad, not great {emoji}",
        "Standard features {emoji}",
        "Expected more {emoji}",
        "Decent quality {emoji}",
        "Regular performance {emoji}",
        "Basic functionality {emoji}",
        "Mixed feelings {emoji}",
    ]

    # Product brands and features
    brands = [
        "Apple",
        "Samsung",
        "Google",
        "Microsoft",
        "Amazon",
        "Sony",
        "Dell",
        "HP",
        "Lenovo",
        "LG",
    ]
    products = [
        "phone",
        "laptop",
        "tablet",
        "smartwatch",
        "headphones",
        "speaker",
        "camera",
        "TV",
        "console",
    ]
    features = [
        "battery life",
        "screen",
        "camera",
        "performance",
        "design",
        "software",
        "price",
        "quality",
    ]

    # Emojis
    positive_emojis = ["😊", "😍", "🥰", "👍", "⭐", "🎉", "💯", "🙌"]
    negative_emojis = ["😠", "😡", "👎", "😤", "😒", "😑", "💔", "😫"]
    neutral_emojis = ["🤔", "😐", "😕", "🤷", "😶", "💭", "❓", "💫"]

    # Create lists to store data
    texts = []
    dates = []

    # Generate random dates within the last year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    for _ in range(size):
        # Randomly select sentiment
        sentiment = np.random.choice(
            ["positive", "negative", "neutral"], p=[0.4, 0.3, 0.3]
        )

        if sentiment == "positive":
            template = random.choice(positive_phrases)
            emoji = random.choice(positive_emojis)
        elif sentiment == "negative":
            template = random.choice(negative_phrases)
            emoji = random.choice(negative_emojis)
        else:
            template = random.choice(neutral_phrases)
            emoji = random.choice(neutral_emojis)

        # Create post content
        brand = random.choice(brands)
        product = random.choice(products)
        feature = random.choice(features)

        # Generate text with more context
        text = template.format(emoji=emoji)
        context = f"The {brand} {product}'s {feature} "

        # Combine context and sentiment
        full_text = context + text
        texts.append(full_text)

        # Generate random date
        random_days = random.randint(0, 365)
        post_date = start_date + timedelta(days=random_days)
        dates.append(post_date)

    # Create DataFrame
    df = pd.DataFrame({"text": texts, "date": dates})

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Save to CSV
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Created dataset with {size} samples at: {output_path}")
    return df


if __name__ == "__main__":
    # Create sample dataset
    df = create_sample_dataset()
