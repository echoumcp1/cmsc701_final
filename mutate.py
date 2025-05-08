import math
import random
import csv
import textwrap
from collections import defaultdict
import json

average_mutation_probs = {
    'A': {'C': 0.0004, 'G': 0.0006, 'T': 0.0010},
    'C': {'A': 0.0013, 'G': 0.0004, 'T': 0.0011},
    'G': {'A': 0.0010, 'C': 0.0004, 'T': 0.0011},
    'T': {'A': 0.0009, 'C': 0.0004, 'G': 0.0005},
}

homopolymer_indel_probs = [
    0.017, 0.016, 0.018, 0.023, 0.031, 0.041, 0.068, 0.093, 0.78, 0.945
]

indel_lengths = list(range(1, 11))
indel_probs = [math.exp(-l) for l in indel_lengths]
indel_probs = [p / sum(indel_probs) for p in indel_probs]

p_mismatch = 1 / 13048
p_indel_nonhomo = 1 / 9669
p_indel_homo = 1 / 477
 
substitution_total = {b: sum(average_mutation_probs[b].values()) for b in average_mutation_probs}
sub_prob = sum(substitution_total.values())

def get_homopolymer_length(seq, index):
    base = seq[index]
    i = 0
    while index + i < len(seq) and seq[index + i] == base:
        i += 1
    return i

def get_indel_prob(homopolymer_len):
    return homopolymer_indel_probs[homopolymer_len-2] if homopolymer_len - 2 >= 0 else 0

def random_base():
    return random.choice(['A', 'C', 'G', 'T'])

def mutate_sequence_short(seq, chunk_size=175):
    mutated_chunks = []
    indel_stats = []
    mutation_stats = []

    possible_starts = list(range(0, len(seq) - chunk_size + 1))

    random.shuffle(possible_starts)
    segments = []
    used_ranges = []

    for start in possible_starts:
        end = start + chunk_size
        overlap = any(start < r[1] and end > r[0] for r in used_ranges)
        if not overlap:
            segments.append(seq[start:end])
            used_ranges.append((start, end))
            if len(segments) == 10:
                break
    j = 0
    for segment in segments:
        if j < 3:
            mutated_chunks.append(segment)
            mutation_stats.append(dict())
            indel_stats.append(dict())
        else:
            mutation_count = {
                'A': defaultdict(int),
                'C': defaultdict(int),
                'G': defaultdict(int),
                'T': defaultdict(int),
            }
            indel_count = { 
                'insertions': defaultdict(int),
                'deletions': defaultdict(int),
                'homopolymer': defaultdict(lambda: defaultdict(int))
            }
            current_chunk = ""
            i = 0
            while i < len(segment):
                homopolymer_length = get_homopolymer_length(segment, i) 
                if (homopolymer_length > 1):
                    indel_len = 0
                    for j in range(homopolymer_length):
                        if random.random() < get_indel_prob(homopolymer_length):
                            if random.random() < 0.35:
                                indel_len += 1
                            else:
                                indel_len -= 1
                    inserted = segment[i] * (indel_len + homopolymer_length)
                    current_chunk += inserted
                    i += homopolymer_length
                    indel_count['homopolymer'][homopolymer_length][indel_len] += 1
                else:
                    base = segment[i]
                    rand = random.random()
                    if rand < sub_prob * 0.1:
                        if random.random() < 0.35:
                            indel_len = random.choices(indel_lengths, weights=indel_probs)[0]
                            inserted = ''.join(random_base() for _ in range(indel_len))
                            current_chunk += inserted + base
                            i += 1
                            indel_count['insertions'][indel_len] += 1 
                        else:
                            indel_len = random.choices(indel_lengths, weights=indel_probs)[0]
                            i += indel_len
                            indel_count['deletions'][indel_len] += 1 
                    elif rand < sub_prob:
                        mutations = average_mutation_probs[base]
                        choices = list(mutations.keys())
                        weights = [mutations[b] for b in mutations]
                        change = random.choices(choices, weights=weights)[0]
                        current_chunk += change
                        mutation_count[base][change] += 1
                        i += 1
                    else:
                        current_chunk += base
                        i += 1
            mutated_chunks.append(current_chunk)
            mutation_stats.append(mutation_count)
            indel_stats.append(indel_count)
        j += 1
        

    return mutated_chunks, mutation_stats, indel_stats

def mutate_sequence_long(seq, chunk_size = 1000):
    mutated_chunks = []
    indel_stats = []
    mutation_stats = []

    max_start = len(seq) - chunk_size

    seen = set()
    segments = []

    while len(segments) < 20:
        start = random.randint(0, max_start)
        segment = seq[start:start + chunk_size]
        if segment not in seen:
            seen.add(segment)
            segments.append(segment)

    j = 0
    
    for segment in segments:
        if j < 5:
            mutated_chunks.append(segment)
            mutation_stats.append(dict())
            indel_stats.append(dict())
        else:
            mutation_count = {
                'A': defaultdict(int),
                'C': defaultdict(int),
                'G': defaultdict(int),
                'T': defaultdict(int),
            }
            indel_count = { 
                'insertions': defaultdict(int),
                'deletions': defaultdict(int),
                'homopolymer': defaultdict(lambda: defaultdict(int))
            }
            current_chunk = ""
            i = 0
            while i < len(segment):
                rand = random.random()
                if rand < p_mismatch:
                    change = random.choice([b for b in ['A', 'C', 'G', 'T'] if b != segment[i]])
                    current_chunk += change
                    mutation_count[segment[i]][change] += 1
                    i += 1
                elif rand < p_mismatch + p_indel_homo:
                    homopolymer_length = get_homopolymer_length(segment, i)
                    if homopolymer_length > 1:
                        if random.random() < 0.5:
                            current_chunk += (segment[i] * (homopolymer_length + 1))
                            indel_count['insertions'][1] += 1 
                            indel_count['homopolymer'][homopolymer_length][1] += 1
                        else:
                            current_chunk += (segment[i] * (homopolymer_length - 1))
                            indel_count['deletions'][1] += 1 
                            indel_count['homopolymer'][homopolymer_length][-1] += 1
                        i += homopolymer_length
                    else:
                        current_chunk += segment[i]
                        i += 1
                elif rand < p_mismatch + p_indel_homo + p_indel_nonhomo:
                    homopolymer_length = get_homopolymer_length(segment, i)
                    if homopolymer_length == 1:
                        if random.random() < 0.5:
                            current_chunk += (segment[i] + random_base())
                            indel_count['insertions'][1] += 1 
                    else:
                        current_chunk += segment[i]
                    i += 1 
                else:
                    current_chunk += segment[i]
                    i += 1
            mutated_chunks.append(current_chunk)
            mutation_stats.append(mutation_count)
            indel_stats.append(indel_count)
        j += 1

    return mutated_chunks, mutation_stats, indel_stats


def process(input_path, output_path_base, chunk_size=175):
    with open(input_path, newline='') as infile, open(output_path_base + ".csv", 'w', newline='') as outfile, open(output_path_base + ".fa", 'w') as fa_outfile, open(output_path_base + "_mutation_stats.json", 'w') as mutation_stats_file, open(output_path_base + "_indel_stats.json", 'w') as indel_stats_file:
        reader = csv.DictReader(infile)
        fieldnames = ['', 'label', 'subsequence', 'description', 'group', 'genus', 'species_epithet']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            row_id = row[""]
            sequence = row['sequence']
            description = row['description']
            group = row['group']
            genus = row['genus']
            species_epithet = row['species_epithet']

            chunks, mutation_stats, indel_stats = mutate_sequence_long(sequence, chunk_size)
            for i, chunk in enumerate(chunks):
                is_sub_seq = 1 if i < 5 else 0
                writer.writerow({
                    '': row_id,
                    'label': is_sub_seq,
                    'subsequence': chunk,
                    'description': f"{description} Chunk {i+1}",
                    'group': group,
                    'genus': genus,
                    'species_epithet': species_epithet
                })
                chunk_description = f"{description} chunk {i} - {group}, {genus}, {species_epithet}"
                fasta_header = f">{chunk_description}"
                wrapped_sequence = "\n".join(textwrap.wrap(chunk, width=80))

                fa_outfile.write(f"{fasta_header}\n{wrapped_sequence}\n")

            mutation_stats_file.write(json.dumps(mutation_stats))
            indel_stats_file.write(json.dumps(indel_stats))

for chunk_sz in range(1000, 3001, 500):
    input_csv_path = 'data/genomic_species.csv'
    output_path_base = f'data/mutate_and_indel_context_{chunk_sz}'
    process(input_csv_path, output_path_base, chunk_sz)
