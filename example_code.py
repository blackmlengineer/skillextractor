import csv
import json
import time

from skillextractor import SkillExtractor

se = SkillExtractor(
    taxonomies=["onet", "ukssc", "esco", "lightcast", "wef"],
    model="blackmlengineer/skill-extractor-v0.1",
    threshold=0.875,
    device="cuda",
)

jobs = []
with open("example_jobs.csv","r") as f:
    csvf = csv.DictReader(f)
    for row in csvf:
        jobs.append({
            "id" : row["id"],
            "title" : row["title"],
            "description" : row["description"]
        })

results = se.batch_extract([job["description"] for job in jobs])