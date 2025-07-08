import json
import random
from faker import Faker

fake = Faker()

def generate_business_samples(n):
    samples = []
    for _ in range(n):
        template = random.choice([
            "{company} reported {adj} quarterly earnings driven by {driver}.",
            "{company}'s stock price {action} after announcing {event}.",
            "The {industry} industry is {trend} due to {factor}.",
            "{company} announced a {adj} acquisition deal worth {amount}.",
            "The Federal Reserve {action} interest rates by {amount} to {reason}.",
            "{product} prices {action} amid {factor} concerns.",
            "The {market} market experienced {adj} volatility following {event}.",
            "Major {sector} companies reported {adj} profits despite {challenge}.",
            "{industry} sales grew {percent} year-over-year during {period}.",
            "The {asset} market showed signs of {trend} with {factor}."
        ])
        samples.append(template.format(
            company=fake.company(),
            adj=random.choice(["strong", "weak", "mixed", "record", "disappointing"]),
            driver=random.choice(["consumer demand", "new product lines", "cost cutting", "overseas expansion"]),
            action=random.choice(["surged", "plummeted", "fluctuated", "stabilized"]),
            event=random.choice(["record sales", "a product recall", "regulatory approval", "management changes"]),
            industry=random.choice(["automotive", "tech", "retail", "healthcare", "financial"]),
            trend=random.choice(["expanding", "contracting", "transforming", "consolidating"]),
            factor=random.choice(["supply chain issues", "consumer trends", "regulatory changes", "technological advances"]),
            amount=random.choice(["$1 billion", "$500 million", "0.5%", "25 basis points"]),
            reason=random.choice(["combat inflation", "stimulate growth", "stabilize markets"]),
            product=random.choice(["Oil", "Gasoline", "Commodity", "Agricultural"]),
            market=random.choice(["cryptocurrency", "bond", "stock", "housing"]),
            sector=random.choice(["banking", "investment", "insurance", "financial"]),
            percent=random.choice(["5%", "10%", "15%", "20%"]),
            period=random.choice(["the holiday season", "summer months", "Q2", "the fiscal year"]),
            asset=random.choice(["housing", "commercial real estate", "auto", "luxury goods"])
        ))
    return samples

categories = {
    "business": generate_business_samples,
    # Add similar generator functions for other categories
}

output = []
for category, generator in categories.items():
    output.append({category: generator(1000)})  # Generate 1000 samples per category

with open("expanded_training_samples.json", "w") as f:
    json.dump(output, f, indent=2)