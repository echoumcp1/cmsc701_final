import csv

fasta_path = "data/genomic_species.fa"
name_to_tax_id = {}
with open(fasta_path, 'r') as f:
    for line in f:
        if line.startswith('>'):
            header = line[1:].strip()
            parts = header.split('|')
            name = parts[0]
            taxid = parts[2]
            name_to_tax_id[name] = taxid

for size in range(1000, 3001, 500):
    filepath = f"mutate_and_indel_context_{size}"
    with open("results/" + filepath + "_results", 'r') as res, open("data/" + filepath + ".csv", 'r') as seqs:
        reader = csv.DictReader(seqs)
        matches = 0
        for line, row in zip(res, reader):
            group = row['group']
            genus = row['genus']
            species_epithet = row['species_epithet']

            parts = line.strip().split('\t')
            matches += 1 if parts[2] == name_to_tax_id[f"{group}_{genus}_{species_epithet}"] else 0
    
    print(matches/8500)