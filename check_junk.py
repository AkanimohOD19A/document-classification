import json
# Samples
with open("./data/training_samples.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

business_texts = samples[0]['business']
technology_texts = samples[1]['technology']
sports_texts = samples[2]['sports']
entertainment_texts = samples[3]['entertainment']
politics_texts = samples[4]['politics']

print(business_texts, "\n")
print(technology_texts, "\n")