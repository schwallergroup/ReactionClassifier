COHORT_SYSTEM_PROMPT = """
Expert Organic Chemistry Reaction Classification Prompt (Cohort Mode)

You are given multiple reactions that share the same core transformation (same reaction template). Your task is to pick a SINGLE two-level hierarchy classification (X.Y) that best represents this transformation for the entire cohort, not per reaction.

Since all reactions in the cohort share the same reaction template, they undergo the same core structural transformation. Analyze this shared transformation to select the most appropriate X.Y classification.

Reaction Hierarchy

1. Heteroatom alkylation and arylation
   1.1 N-substitution with alkyl-X
   1.2 Reductive amination
   1.3 N-arylation with Ar-X
   1.4 Amide N-alkylation
   1.5 Aniline N-alkylation
   1.6 Heteroaryl N-alkylation
   1.7 O-substitution
   1.8 S-substitution

2. Acylation and related processes
   2.1 N-acylation to amide
   2.2 N-sulfonylation
   2.3 N-acylation to urea
   2.4 Carbamate/carbonate formation
   2.5 Amidine formation
   2.6 O-acylation to ester
   2.7 O-sulfonylation
   2.8 Other

3. C–C bond formation
   3.1 Suzuki coupling
   3.2 Heck reaction
   3.3 Sonogashira reaction
   3.4 Stille reaction
   3.5 Other Pd-catalyzed reactions (Negishi, Kumada, etc.)
   3.6 Ester condensation
   3.7 Grignard
   3.8 Wittig olefination
   3.9 Other organometallic (e.g., organolithium)
   3.10 Friedel–Crafts acylation
   3.11 Other

4. Heterocycle formation
   4.1 N-containing
   4.2 O-containing
   4.3 S-containing

5. Protections
   5.1 Deprotections
   5.2 Reductions

6. Reductions
   6.1 NO₂ to amine
   6.2 Amide to amine
   6.3 CN or imine to amine
   6.4 Ester to alcohol
   6.5 Ketone to alcohol
   6.6 Alkene to alkane
   6.7 Alkyne to alkane
   6.8 Aryl/hetaryl to fully sat.
   6.9 Other

7. Oxidations
   7.1 Alcohols to aldehydes
   7.2 At sulfur
   7.3 Alcohols to acids
   7.4 At nitrogen
   7.5 Alkene oxidative cleavage
   7.6 Benzylic/allylic oxidation
   7.7 Alkene oxidation
   7.8 Other

8. Functional group interconversion (FGI)
   8.1 Alcohol to halide
   8.2 Amide to imidoyl chloride
   8.3 Acid to acid chloride
   8.4 Nitrile to acid
   8.5 Dehydration
   8.6 Carbonyl to C=N
   8.7 Other

9. Functional group addition (FGA)
   9.1 Halogenation
   9.2 Nitration
   9.3 Sulfonation
   9.4 Other

Classification Instructions:
1. Identify the key bonds formed/broken and functional groups involved in the shared transformation
2. Focus on the core, reagent-agnostic structural change that all reactions in the cohort undergo
3. Select the most appropriate X.Y sub class from the hierarchy above
4. Do NOT consider reagents/catalysts/conditions at this stage - only the core transformation

Output exactly one line in exactly the following tags:
<reaction_class>X.Y</reaction_class>

Cohort reactions (same transformation):
{{reaction_smiles}}

Retro Template (shared across cohort):
{{retro_template}}
"""
