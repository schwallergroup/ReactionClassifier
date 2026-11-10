cohort_system_prompt = """
You are proposing a new reaction class for a cohort of reactions that share the same transformation and the same reaction template.

The verifier has determined that the current hierarchy does not adequately describe this cohort. The failure was due to: {{cause}}

**Understanding the Failure Cause:**
- **core_transformation**: The shared L3/L4 transformation is not in the hierarchy. You must create a new canonical L3/L4 class.
- **reagents**: The L3/L4 exists but L5/6+ variants for specific reagents are missing. You may need to add reagent-specific subclasses.

**Critical Principles:**

1. **"What vs. How" Naming Principle:**
   - **The "What" (Reactants):** Level 3 and 4 names must clearly describe the reagent-agnostic core transformation by specifying the key reactant functional groups (e.g., "Amidation of carboxylic acids with secondary amines to tertiary amide")
   - **The "How" (Reagents):** Level 5/6+ names describe **Named Reactions** or **General Condition Sets** (e.g., "cond: Leuckart-Wallach Reaction", "cond: Acidic Hydrolysis", or "cond: EDC coupling")

2. **One Canonical Class:** Propose ONE canonical L3/L4 class that captures the shared core transformation. This class must be determinable from reactants and products alone, WITHOUT considering reagents.

3. **Reagent Variants:** If reactions in the cohort use different reagents/catalysts/conditions, you SHOULD propose L5/6+ variants. Each variant:
   - Must sit under the canonical L4
   - **Must use the format: "cond: [Name of Reaction]" or "cond: [General Conditions]"**
   - Should identify standard organic chemistry named reactions (e.g., Pictet-Spengler) if applicable.
   - This is important to include.

4. **Functional Group Examination:** When identifying the core transformation:
   - Examine the complete bonding environment of ALL atoms involved in new bonds, including immediate neighbors
   - A functional group is defined by the full local connectivity pattern
   - Pay strong attention to heteroatoms and conjugated systems bonded to the reacting functional groups

5. **Finding the Best Parent:**
   - Identify the most specific existing parent class that correctly describes the general transformation
   - For L3/L4 classes, the parent is typically L2 (X.Y)
   - For L5/6+ variants, the parent is your new canonical L4 class

6. **Do Not Alter L1/L2:** You must NEVER propose a new entry for the first two digits (X.Y). These are set in the upper hierarchy.

**Hierarchy Levels:**
- Level 1: Super class (e.g., "2. Acylation and related processes")
- Level 2: Sub class (e.g., "2.1 N-acylation to amide") [FIXED - do not modify]
- Level 3: Specific reaction type (e.g., "2.1.3 Amidation with carboxylic acids")
- Level 4: Detailed structural variant (e.g., "2.1.3.2 Carboxylic acid + secondary amine to tertiary amide")
- Level 5+: Named conditions or reagent specificity (e.g., "2.1.3.2.1 cond: Leuckart-Wallach Reaction")

Hierarchy Subtree:
{{reaction_hierarchy}}

Cohort reactions (role-labeled):
{{reactions}}

Retro Template (shared across cohort):
{{retro_template}}

Failure Cause:
{{cause}}

Output strictly as the following XML tags, with no extra text:

<rationale>[Justify your proposal by: 1) Identifying the specific reactant functional groups and their complete bonding environment, 2) Explaining the core transformation (the "what"), 3) If proposing L5/6+ variants, justifying them by the named reaction or condition set (the "how")]</rationale>
<suggested_parent_code>[The full code of the most logical parent for the canonical L3/L4 class, e.g., 2.1 or 2.1.3]</suggested_parent_code>
<new_entry_code>[Full code for the new canonical L3/L4 class]</new_entry_code>
<new_entry_name>[Clear, Title Case name for the canonical class, following the "What vs. How" principle - describing reactants and transformation]</new_entry_name>
<hierarchy_line>[The exact line to append: "code new_entry_name"]</hierarchy_line>
<new_parent_lines>
[Zero or more lines for missing parent levels and any L5/6+ reagent variants, one per line in format "code label"]
[If the canonical class is new (e.g., 2.1.3.4), include it here]
[If L5/6+ variants are needed, include them here under the canonical class (e.g., "2.1.3.4.1 cond: Leuckart-Wallach Reaction")]
[If no additional entries needed, leave empty]
</new_parent_lines>
"""
