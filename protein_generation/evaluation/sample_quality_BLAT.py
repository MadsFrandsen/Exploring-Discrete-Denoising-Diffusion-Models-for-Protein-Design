import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

from evcouplings.couplings import read_raw_ec_file
from evcouplings.compare import (
    SIFTS, intra_dists,
    multimer_dists, coupling_scores_compared
)

from evcouplings.utils import read_config_file
from evcouplings.utils.pipeline import execute

########################

parser = argparse.ArgumentParser(description='SampleQuality')
parser.add_argument('--align_file', type=str, help='Path to file with samples alignment')
parser.add_argument('--handle', type=str, help='Handle for saving files in "output" folder')
args = parser.parse_args()

########################

config_file = "./config_sample_alignment_to_fold.txt"
config = read_config_file(config_file)

align_file = args.align_file
handle = args.handle

config["global"]["theta"] = 1.0 # Don't do reweighting
config["global"]["sequence_file"] = align_file
config["global"]["input_alignment"] = align_file
config["global"]["prefix"] = "output/" + handle

outcfg = execute(**config)

########################

s = SIFTS("./sifts/pdb_chain_uniprot_plus.csv", "./sifts/pdb_chain_uniprot_plus.fa")
selected_structures = s.by_pdb_id("1xpb")

distmap_intra = intra_dists(selected_structures)
distmap_multimer = multimer_dists(selected_structures)

########################

# show these many long-range ECs
raw_ec_file = "./output/"+handle+"/couplings/"+handle+"_ECs.txt"
ecs = read_raw_ec_file(raw_ec_file) # couplings

L = 263
dist_cutoff = 5 # minimum distance in sequence

cc = coupling_scores_compared(
    ecs, distmap_intra, distmap_multimer,
    dist_cutoff=dist_cutoff,
    output_file="CouplingScoresCompared_"+handle+".csv"
)

########################

top = L
prec = cc.head(top)['precision'].mean()
print("Average precision: %.4f\n\nDetails: top %i (L = %i), distance cutoff %i" %(prec, top, L, dist_cutoff))

top = L//2
prec = cc.head(top)['precision'].mean()
print("Average precision: %.4f\n\nDetails: top %i (L = %i), distance cutoff %i" %(prec, top, L, dist_cutoff))

top = L//4
prec = cc.head(top)['precision'].mean()
print("Average precision: %.4f\n\nDetails: top %i (L = %i), distance cutoff %i" %(prec, top, L, dist_cutoff))