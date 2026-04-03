"""All LLM prompts for the deep research multi-hop pipeline.

Centralized here for easy tuning and iteration.
"""

# --- Step 1: Constraint Decomposition + Answer Type Detection ---

DECOMPOSE_PROMPT = """You are a research analyst specializing in breaking down complex, multi-constraint questions.

Given a question, do TWO things:

1. Identify the EXPECTED ANSWER TYPE — what kind of thing is the answer?
   Types: person, place, time, date, number, title, organization, event, artifact, other

2. Identify each INDEPENDENT FACTUAL CONSTRAINT that narrows down the answer.
   - Each constraint should be something you could search for independently
   - Constraints should be specific (dates, numbers, names, properties)
   - Order constraints from MOST searchable to LEAST searchable
   - Category each: temporal, person, event, property, location, artifact, organization

Return JSON:
{{"answer_type": "person", "constraints": [{{"text": "soccer match between 1990 and 1994", "category": "temporal"}}, {{"text": "Brazilian referee", "category": "person"}}]}}

Question: {question}

Return ONLY the JSON object, no other text."""


# --- Step 2: Initial Query Generation ---

INITIAL_QUERIES_PROMPT = """You are a web search expert. Given a complex question, its constraints, and the expected answer type,
generate {num_queries} search queries that are most likely to find the answer.

Expected answer type: {answer_type}

Strategy:
1. Start BROAD — short queries targeting the most distinctive constraint
2. Combine 2-3 constraints per query (not all at once)
3. Include queries targeting the ANSWER TYPE directly (e.g., "list of [answer_type] [constraint]")
4. Include at least one query using specific numbers/dates from the constraints
5. Include one site-specific query: "site:wikipedia.org [most distinctive constraint]"
6. NEVER use the full question as a query — it's too long and specific
7. Each query should be 3-8 words maximum

Question: {question}

Constraints:
{constraints}

Return a JSON array of search query strings:
["query 1", "query 2", ...]

Return ONLY the JSON array, no other text."""


# --- Step 3: Refined Query Generation (subsequent hops) ---

REFINE_QUERIES_PROMPT = """You are a web search expert performing iterative research. Previous searches didn't find the answer.

Expected answer type: {answer_type}
Question: {question}

Constraints:
{constraints}

Previous queries tried (DO NOT repeat these):
{search_history}

Current candidates (may be wrong):
{candidates}

Gap analysis from last hop:
{gap_analysis}

Generate {num_queries} NEW search queries using a DIFFERENT ANGLE:
1. If previous queries were too specific, try broader
2. If previous queries were too broad, try combining constraints differently
3. Search for LISTS or DATABASES: "list of [answer_type] [constraint]", "Wikipedia list [topic]"
4. Try site-specific searches: "site:wikipedia.org", "site:imdb.com", "site:transfermarkt.com"
5. Try searching for the INTERSECTION of the two LEAST explored constraints
6. Try reverse searching: instead of the question's framing, search for what the answer would look like
   (e.g., if looking for a person, search for "[role/profession] [distinctive attribute]")

Return a JSON array of search query strings:
["query 1", "query 2", ...]

Return ONLY the JSON array, no other text."""


# --- Step 4: Candidate Extraction ---

EXTRACT_CANDIDATES_PROMPT = """You are extracting potential answers to a complex question from search results.

Question: {question}

The answer is expected to be a: **{answer_type}**
ONLY extract candidates that match this type. For example, if the answer type is "time", only extract times (like "3:50 PM"), not places or names.

Constraints that the answer must satisfy:
{constraints}

Search findings:
{findings}

Existing candidates (already found):
{existing_candidates}

Instructions:
1. Look for SPECIFIC, SHORT answers matching the expected answer type
2. The answer should be something easily verifiable — typically 1-5 words
3. For each candidate, note which constraints it appears to satisfy
4. Rate confidence 0.0-1.0 based on how many constraints are met
5. Do NOT repeat existing candidates
6. If no new candidates found, return empty array
7. Look carefully in the full page content — answers are often buried in details

Return a JSON array:
[
  {{
    "answer": "Ireland v Romania",
    "confidence": 0.7,
    "source_url": "https://...",
    "source_snippet": "relevant text from the source",
    "constraints_met": ["soccer match 1990-1994", "Brazilian referee"]
  }}
]

Return ONLY the JSON array, no other text."""


# --- Step 5: Constraint Verification ---

VERIFY_PROMPT = """You are verifying whether a candidate answer satisfies ALL constraints of a complex question.

Question: {question}

Candidate answer: {candidate_answer}
Expected answer type: {answer_type}

First: Does the candidate match the expected answer type ({answer_type})? If not, it's automatically wrong.

Constraints to verify:
{constraints}

Available evidence:
{evidence}

For EACH constraint, determine if the candidate answer satisfies it based on the evidence.
Be STRICT — if there's no evidence supporting a constraint, mark it as unmet.

Return JSON:
{{
  "type_match": true/false,
  "all_met": true/false,
  "constraints": [
    {{"text": "constraint text", "met": true/false, "evidence": "supporting text or empty"}},
    ...
  ],
  "overall_confidence": 0.0-1.0,
  "notes": "any additional reasoning"
}}

Return ONLY the JSON, no other text."""


# --- Step 6: Gap Analysis ---

GAP_ANALYSIS_PROMPT = """You are analyzing why the current search hasn't found the answer yet.

Expected answer type: {answer_type}
Question: {question}

Constraints:
{constraints}

What we searched:
{search_history}

What we found (candidates):
{candidates}

What's missing or wrong:
1. Which constraints remain UNMET?
2. Which search angles haven't been tried?
3. Are we looking in the wrong type of source? (e.g., need a specific database, archive, or niche site)
4. Should we try a completely different decomposition of the question?
5. Have we tried searching for LISTS of {answer_type}s matching the constraints?

Provide a brief analysis (2-3 sentences) of what to try next.

Return ONLY the analysis text, no JSON."""
