system_prompt = """
You are given a set of proposed new reaction classes, Your task is to:

Tasks

1) Merge duplicate or overlapping entries: Identify proposals that describe the same or very similar reactions and consolidate them into a single entry. 
2) Assign appropriate hierarchy codes: Place each new class at the correct level in the existing hierarchy, following these principles:

Reaction Hierarchy

Level 1: Super class (e.g., "1. Heteroatom alkylation and arylation")
Level 2: Sub class (e.g., "1.2 Reductive amination")
Level 3: Specific reaction type (e.g., "1.2.1 Reductive Amination with aldehydes")
Level 4: Detailed structural variant (e.g., "1.2.1.1 Aldehyde + Primary Amine to Secondary Amine")
Level 5+: Named reaction conditions or generalized condition sets (e.g., "1.2.1.1.1 cond: Leuckart-Wallach Reaction")

Critical Rules:

- Level 4 must be classifiable without reagent information - a classifier must distinguish at Level 4 using only reactants and products, NOT reagents/conditions.
- **Level 5+ should capture named reactions or general condition sets (prefixed with "cond:") rather than raw reagent strings.**
- Identify and populate missing Level 3 and Level 4 categories.
- Consider functional group variations (primary vs secondary vs tertiary, aliphatic vs aromatic, etc.) at Level 4.
- Preserve legitimate Level 3 categories and add missing detail underneath them.
- Ensure the new classes correctly fit this hierarchy.

Standardize naming: Ensure class names follow consistent nomenclature and are descriptive of the transformation.
Check for duplicates: Do not include classes that already exist in the current hierarchy.
Remove placeholder codes: Replace any "999" codes or conflicting codes with proper sequential numbering that fits the existing hierarchy.
Assign reactions to classes: Map each indexed reaction to its appropriate new or existing class code.

Input:

- A JSON array of proposed new reaction classes (with reaction_smiles, index, suggested_parent_code, new_entry_code, new_entry_name, hierarchy_line, and rationale)
- The existing hierarchy for section 2.1 (Amidation and N-Acylation) or the relevant parents

Additional cohort rule:
- When proposals correspond to cohorts that share a reaction template, ensure there is ONE canonical L3/L4 class for each cohort’s core transformation. Allow L5/L6 variants only when they reflect reagent/condition differences; each variant must sit under the canonical L4. Do not inflate L3/L4.

Output Format:

Your response must be structured with the following tags:

New Hierarchy Classes
<new_hierarchy>
X.Y.Z New Class Name
X.Y.Z.W New Subclass Name
X.Y.Z.W.Q Any details about conditions (use "cond: Name" format). Be sure to preserve condition information if it is provided and correct.
...
</new_hierarchy>
Just put in all the new hierarchies, they will automatically be patched back into the existing by boilerplate code.
You should always include the condition information here, as its important for later stages.

2. Reaction Assignments
<assignments>
{
"0": "X.Y.Z.W",
"1": "X.Y.Z.W.Q",
"2": "X.Y.Z",
...
}
</assignments>

Where:

- Keys are the string representation of the reaction index in the current batch (0-based).
- Values are the assigned hierarchy codes as strings.
- Include ALL reactions from the input with their assigned codes.
- Assignments should use the canonical class code (shared L3/L4) for each cohort core. Do not encode L5/L6 variants in assignments; variants should appear only in <new_hierarchy> when justified by reagents/conditions.
- Never assign conditions to the cohort, but still generate them for the hierarchy.

Key Considerations:

- Multiple proposals may describe the same reaction - merge them.
- Some proposals may already be covered by existing classes - exclude them from <new_hierarchy> but still assign reactions to existing codes.
- Ensure codes don't conflict with existing hierarchy numbering.
- If a code already exists in a hierarchy, DO NOT add one in with the same words or code.
- Maintain sequential numbering within each hierarchy level.
- All reactions must be assigned, even if to existing classes.

Proposals:
{{proposals}}

Relevant Hierarchy:
{{hierarchy}}
"""
