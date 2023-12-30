from pathlib import Path
import logging
from collections import Counter
from Bio.SeqIO.FastaIO import SimpleFastaParser
import sys

def main():

    outfn = Path("data/visualization/real_data/metasets/parsed_metaset.tsv")
    outfn_inter = Path("data/visualization/real_data/metasets/parsed_metaset_intersect.tsv")

    airways = {'result_id': 'Airways',
               'results_file': Path("data/visualization/real_data/metasets/transformer/cami2_airways_contigs_25kbp/coordinates.tsv"),
               'results_file_2': Path("data/visualization/real_data/metasets/satellite_finder/cami2_airways_contigs_25kbp_results.tsv"),
               'annotations_file': Path("data/raw/05_cami2/annotations/airways_annotations.csv"),
               'input_sequences': Path("data/raw/05_cami2/cami2_airways_contigs_25kbp.fna")}

    gi = {'result_id': 'GI',
            'results_file': Path("data/visualization/real_data/metasets/transformer/cami2_gi_contigs_25kbp/coordinates.tsv"),
            'results_file_2': Path("data/visualization/real_data/metasets/satellite_finder/cami2_gi_contigs_25kbp_results.tsv"),
            'annotations_file': Path("data/raw/05_cami2/annotations/gi_annotations.csv"),
            'input_sequences': Path("data/raw/05_cami2/cami2_gi_contigs_25kbp.fna")}
    
    marine = {'result_id': 'Marine',
                'results_file': Path("data/visualization/real_data/metasets/transformer/cami2_marine_contigs_25kbp/coordinates.tsv"),
                'results_file_2': Path("data/visualization/real_data/metasets/satellite_finder/cami2_marine_contigs_25kbp_results.tsv"),
                'annotations_file': Path("data/raw/05_cami2/annotations/marine_annotations.csv"),
                'input_sequences': Path("data/raw/05_cami2/cami2_marine_contigs_25kbp.fna")}
    oral = {'result_id': 'Oral',
            'results_file': Path("data/visualization/real_data/metasets/transformer/cami2_oral_contigs_25kbp/coordinates.tsv"),
            'results_file_2': Path("data/visualization/real_data/metasets/satellite_finder/cami2_oral_contigs_25kbp_results.tsv"),
            'annotations_file': Path("data/raw/05_cami2/annotations/oral_annotations.csv"),
            'input_sequences': Path("data/raw/05_cami2/cami2_oral_contigs_25kbp.fna")}

    skin = {'result_id': 'Skin',
            'results_file': Path("data/visualization/real_data/metasets/transformer/cami2_skin_contigs_25kbp/coordinates.tsv"),
            'results_file_2': Path("data/visualization/real_data/metasets/satellite_finder/cami2_skin_contigs_25kbp_results.tsv"),
            'annotations_file': Path("data/raw/05_cami2/annotations/skin_annotations.csv"),
            'input_sequences': Path("data/raw/05_cami2/cami2_skin_contigs_25kbp.fna")}
    
    rhizo = {'result_id': 'Rhizosphere',
                'results_file': Path("data/visualization/real_data/metasets/transformer/cami2_rhizo_contigs_25kbp/coordinates.tsv"),
                'results_file_2': Path("data/visualization/real_data/metasets/satellite_finder/cami2_rhizo_contigs_25kbp_results.tsv"),
                'annotations_file': Path("data/raw/05_cami2/annotations/rhizo_annotations.csv"),
                'input_sequences': Path("data/raw/05_cami2/cami2_rhizo_contigs_25kbp.fna")}

    metasets = [airways, gi, marine, oral, rhizo, skin]

    with open(outfn, 'w') as fout, open(outfn_inter, 'w') as fout_inter:
        # Get tranformer results:

        for metaset in metasets:
            print("Processing:", metaset['result_id'])

            result_id = metaset['result_id']
            results_file = metaset['results_file']
            annotations_file = metaset['annotations_file']
            input_sequences = metaset['input_sequences']
            
            contigs = []
            with open(results_file, "r") as fin:
                for line in fin:
                    contigs.append(line.strip().split("\t")[0])
            
            annotations = {}
            with open(annotations_file, "r") as fin:
                fin.readline()
                for line in fin:
                    id = line.strip().split(",")[0]
                    tax = line.strip().split(",")[-1].split(';')
                    annotations[id] = tax

            contig_counts = Counter(contigs)

            missing = 0
            for contig in contig_counts:
                try:
                    print('transformer', result_id, contig, contig_counts[contig], "positive", "\t".join(annotations[contig][-3:]), sep='\t', file=fout)
                except KeyError:
                    print('transformer', result_id, contig, contig_counts[contig], "positive", "\t".join(["unknown"]*3), sep='\t', file=fout)
                    missing += 1

            satellite_finder_counts = {}
            with open(metaset['results_file_2'], "r") as fin:
                for line in fin:
                    contig, count = line.strip().split("\t")[1:]
                    if count == '0':
                        continue
                    satellite_finder_counts[contig] = int(count)
            
            for contig in satellite_finder_counts:
                try:
                    print('satellite_finder', result_id, contig, satellite_finder_counts[contig], "positive", "\t".join(annotations[contig][-3:]), sep='\t', file=fout)
                except KeyError:
                    print('satellite_finder', result_id, contig, satellite_finder_counts[contig], "positive", "\t".join(["unknown"]*3), sep='\t', file=fout)

            # Get negatives:
            all_contigs = set()
            with open(input_sequences, "r") as fin:
                input_sequences = SimpleFastaParser(fin)
                for seq_acc, _ in input_sequences:
                    all_contigs.add(seq_acc)

            for contig in all_contigs:
                if contig not in contig_counts:
                    try:
                        print('background', result_id, contig, 0, "negative", "\t".join(annotations[contig][-3:]), sep='\t', file=fout)
                    except KeyError:
                        print('background', result_id, contig, 0, "negative", "\t".join(["unknown"]*3), sep='\t', file=fout)
                        missing += 1
            
            print("Missing annotations:", missing)
            print("Total contigs:", len(all_contigs))
            print("Total contigs with annotations:", len(annotations))
            print("Total contigs with transformer predictions:", len(contig_counts))
            print("Total contigs with satellite finder predictions:", len(satellite_finder_counts))

            transformer_contig_set = set(contig_counts.keys())
            satellite_finder_contig_set = set(satellite_finder_counts.keys())

            only_transformer = transformer_contig_set - satellite_finder_contig_set
            only_satellite_finder = satellite_finder_contig_set - transformer_contig_set
            intersection = transformer_contig_set.intersection(satellite_finder_contig_set)
            no_prediction = all_contigs - transformer_contig_set - satellite_finder_contig_set
            distribution = {
                'transformer': only_transformer,
                'satellite_finder': only_satellite_finder,
                'both': intersection,
                'negative': no_prediction
            }

            for element in distribution:
                for contig in distribution[element]:
                    try:
                        print(result_id, contig, element, "\t".join(annotations[contig][-3:]), sep='\t', file=fout_inter)
                    except KeyError:
                        print(result_id, contig, element, "\t".join(["unknown"]*3), sep='\t', file=fout_inter)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()