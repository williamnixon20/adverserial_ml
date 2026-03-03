import pandas as pd
import matplotlib.pyplot as plt

# 1. Load data from JSON files
# Assuming files contain list of dicts: [{'label': 'A''value': 10}...]
PESQ_result_elevenlab = pd.read_json('PESQ_result_elevenlab.json')
PESQ_result = pd.read_json('PESQ_result.json')

agg_pesq_score = {"Bit Rate":[], "Sample Rate":[], "Speed with Altered Pitch":[], "Speed with Pitch Adjusted":[]}

for _, d1 in PESQ_result.iterrows():
    if d1["mode"] == "bit_rate":
        agg_pesq_score["Bit Rate"].append(d1["PESQ Score"])
    elif d1["mode"] == "sample_rate":
        agg_pesq_score["Sample Rate"].append(d1["PESQ Score"])
    elif d1["mode"] == "speed_with_pitch":
        agg_pesq_score["Speed with Altered Pitch"].append(d1["PESQ Score"])
    elif d1["mode"] == "speed_no_pitch":
        agg_pesq_score["Speed with Pitch Adjusted"].append(d1["PESQ Score"])


for _, d2 in PESQ_result_elevenlab.iterrows():
    if d2["mode"] == "bit_rate":
        agg_pesq_score["Bit Rate"].append(d2["PESQ Score"])
    elif d2["mode"] == "sample_rate":
        agg_pesq_score["Sample Rate"].append(d2["PESQ Score"])
    elif d2["mode"] == "speed_with_pitch":
        agg_pesq_score["Speed with Altered Pitch"].append(d2["PESQ Score"])
    elif d2["mode"] == "speed_no_pitch":
        agg_pesq_score["Speed with Pitch Adjusted"].append(d2["PESQ Score"])



fig1, axes1 = plt.subplots(figsize=(9, 7))

# First dataset
bars1 = axes1.bar(["Bit Rate", "Sample Rate", "Speed with Altered Pitch", "Speed with Pitch Adjusted"], 
            [sum(agg_pesq_score["Bit Rate"])/len(agg_pesq_score["Bit Rate"]), 
             sum(agg_pesq_score["Sample Rate"])/len(agg_pesq_score["Sample Rate"]),
             sum(agg_pesq_score["Speed with Altered Pitch"])/len(agg_pesq_score["Speed with Altered Pitch"]),
             sum(agg_pesq_score["Speed with Pitch Adjusted"])/len(agg_pesq_score["Speed with Pitch Adjusted"])], color='purple')
axes1.set_title("Average PESQ Scores By Perturbation Methods")
axes1.set_ylabel("Average PESQ Score")

for bar1 in bars1:
    axes1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.02,
            f'{bar1.get_height():.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('pesq_scores.png', dpi=150, bbox_inches='tight')
plt.close()



checker_result = pd.read_json('checker_result.json')


agg_confidence_score = {"Bit Rate":[], "Sample Rate":[], "Speed with Altered Pitch":[], "Speed with Pitch Adjusted":[]}

for _, d3 in checker_result.iterrows():
    if d3["perturbation_mode"] == "bit_rate":
        agg_confidence_score["Bit Rate"].append(d3["detection_confidence"])
    elif d3["perturbation_mode"] == "sample_rate":
        agg_confidence_score["Sample Rate"].append(d3["detection_confidence"])
    elif d3["perturbation_mode"] == "speed_with_pitch":
        agg_confidence_score["Speed with Altered Pitch"].append(d3["detection_confidence"])
    elif d3["perturbation_mode"] == "speed_no_pitch":
        agg_confidence_score["Speed with Pitch Adjusted"].append(d3["detection_confidence"])





fig2, axes2 = plt.subplots(figsize=(9,7))

# First dataset
bars2 = axes2.bar(["Bit Rate", "Sample Rate", "Speed with Altered Pitch", "Speed with Pitch Adjusted"], 
            [sum(agg_confidence_score["Bit Rate"])/len(agg_confidence_score["Bit Rate"]), 
             sum(agg_confidence_score["Sample Rate"])/len(agg_confidence_score["Sample Rate"]),
             sum(agg_confidence_score["Speed with Altered Pitch"])/len(agg_confidence_score["Speed with Altered Pitch"]),
             sum(agg_confidence_score["Speed with Pitch Adjusted"])/len(agg_confidence_score["Speed with Pitch Adjusted"])], color='darkturquoise')
axes2.set_title("Average AudioSeal Confidence Scores By Perturbation Methods")
axes2.set_ylabel("Average AudioSeal Confidence Score")
for bar2 in bars2:
    h = bar2.get_height()
    axes2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.02,
            f'{h:.2e}' if abs(h) < 0.01 else f'{h:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('audio_seal_confidence.png', dpi=150, bbox_inches='tight')
plt.close()