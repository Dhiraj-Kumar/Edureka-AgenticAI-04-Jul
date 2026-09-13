import json

DATASET = [
    {
        "prompt": "What is NimbusCloud's Pro tier price?",
        "completion": "NimbusCloud's Pro tier costs $29/month with 500GB storage and unlimited API calls."
    },
    {
        "prompt": "How long is the refund window?",
        "completion": "NimbusCloud offers refunds within 14 days of purchase for any paid plan."
    },
    {
        "prompt": "Is 2FA required for Enterprise accounts?",
        "completion": "Yes, two-factor authentication is mandatory for all NimbusCloud Enterprise accounts."
    },
    {
        "prompt": "What is the Pro tier's API rate limit?",
        "completion": "The Pro tier allows 1000 API requests per minute."
    },
    {
        "prompt": "What does the free tier include?",
        "completion": "The free tier includes 5GB of storage and 100 API calls per day."
    },
] * 4

with open("domain_dataset.jsonl", "w", encoding="utf-8") as file:
    for row in DATASET:
        file.write(json.dumps(row) + "\n")

print(f"Wrote {len(DATASET)} examples to domain_dataset.jsonl")