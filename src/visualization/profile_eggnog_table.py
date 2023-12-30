from pathlib import Path
import sys
import requests

# # linear
# annotations = Path("data/visualization/profile_function/linear_c_004_annotations.emapper.annotations.tsv")
# output_file = Path("data/visualization/profile_function/linear_c_004_annotations_cog_kegg_pfam.tsv")

# # transformer
# annotations = Path("data/visualization/profile_function/transformer/top_21_features_freq_adj.emapper.annotations.tsv")
# output_file = Path("data/visualization/profile_function/transformer/top_21_features_freq_adj_annotations_cog_kegg_pfam.tsv")

# all
annotations = Path("data/visualization/profile_function/transformer/all_representatives.emapper.annotations.tsv")
output_file = Path("data/visualization/profile_function/transformer/top_21_features_phage_annotations_cog_kegg_pfam.tsv")

top_phage = [
"IMGVR_UViG_2878445092_000002_30","IMGVR_UViG_2847235940_000003_11","IMGVR_UViG_2869822064_000003_9",
"IMGVR_UViG_2869577513_000004_18","IMGVR_UViG_2791355158_000007_9",
"SSRR4423633Ccontig151_17","SSRR2674234Ccontig439_10","IMGVR_UViG_2541047946_000016_14",
"IMGVR_UViG_2718218150_000001_8","IMGVR_UViG_2548876993_000001_7","IMGVR_UViG_2847550744_000002_29",
"IMGVR_UViG_2541047946_000016_9","IMGVR_UViG_2744054653_000001_39","IMGVR_UViG_2838840603_000004_29",
"IMGVR_UViG_2636415848_000012_26","IMGVR_UViG_2846898933_000001_23","IMGVR_UViG_2724679539_000004_14",
"IMGVR_UViG_2603880015_000004_25","IMGVR_UViG_2681812993_000002_32","IMGVR_UViG_2541047793_000001_36",
"IMGVR_UViG_2671180952_000010_24", "IMGVR_UViG_2690316302_000001_15"
]

# Function to fetch KEGG pathway description
def get_kegg_description(kegg_id):
    kegg_api_url = f"http://rest.kegg.jp/get/{kegg_id}"
    response = requests.get(kegg_api_url)
    try:
        description = " ".join(response.text.split("\n")[2].split()[1:])
    except IndexError:
        description = 'No description available'
    return description

# Function to fetch COG description from NCBI COG API
def get_cog_description(cog_id):
    cog_api_url = f"https://www.ncbi.nlm.nih.gov/research/cog/api/cogdef/?cog={cog_id}"
    response = requests.get(cog_api_url)
    if response.ok:
        data = response.json()
        results = data.get('results', [])
        if results:
            name = results[0].get('name', 'No description available')
            return name
    return 'No description available'

# Function to extract the first COG ID from the eggNOG_OGs column
def extract_first_cog(eggNOG_OGs):
    # Split the string by commas and take the first COG ID
    for item in eggNOG_OGs.split(','):
        if "COG" in item:
            return item.split('@')[0]
    return None

with open(annotations, 'r') as fin, open(output_file, 'w') as fout:
    for line in fin:
        if line.startswith("#query"):
            elements = line.strip().split("\t")
            for n, element in enumerate(elements):
                print(n, element)
        if line.startswith("#"):
            continue
        elements = line.strip().split("\t")
        
        #function = elements[20]
        name = elements[0].replace("|","_")
        if name.replace("|","_") not in top_phage:
            continue

        # if len(name.split('_')) > 2:
        #     name = "_".join(name.split("_")[0:2]) + "|" + name.split("|")[1]
        function = elements[7]
        if len(function.split()) > 15:
            function = " ".join(function.split(" ")[0:1])
        
        pfam = elements[20]
        pfam = ' and '.join(pfam.strip().split(','))
        ogs = elements[4]
        best_cog = extract_first_cog(ogs)
        kegg = elements[11] if elements[11] != "-" else None  
        try:
            kegg = kegg.split(",")[0]
        except AttributeError:
            pass
        if best_cog:
            cog_name = get_cog_description(best_cog)
        else:
            cog_name = '-'
        if kegg:
            kegg_name = get_kegg_description(kegg)
        else:
            kegg_name = '-'
        
        #print(f"name: {name}", f"pfam: {' and '.join(pfam.strip().split(','))}", f"COG: {cog_name}", f"KEGG: {kegg_name}", sep="\t")
        
        if cog_name == '-':
            if pfam == '-':
                id = f"unknown"
            else:
                id = f"Pfam: {pfam}"
        else:
            id = f"COG: {cog_name}"
        print(name, id, sep="\t", file=fout)
        # print(name,
        #       cog_name,
        #       kegg_name,
        #       pfam,
        #       sep="\t",
        #       file=fout)