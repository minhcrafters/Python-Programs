inp = input()
seq = ""

bases = ["U", "C", "A", "G"]
codons = [a + b + c for a in bases for b in bases for c in bases]
print(codons)
amino_acids = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
codon_table = dict(zip(codons, amino_acids))
print(codon_table)

for c in inp:
    if c.lower() == "t":
        seq += "A"
    elif c.lower() == "c":
        seq += "G"
    elif c.lower() == "a":
        seq += "U"
    elif c.lower() == "g":
        seq += "C"

# start = seq.find("AUG")
# peptide = []
# i = start

# while i < len(seq) - 2:
#     codon = seq[i : i + 3]
#     a = codon_table[codon]
#     if a == "*":
#         break
#     i += 3
#     peptide.append(a)
    
# out = "".join(peptide)

print(seq)
