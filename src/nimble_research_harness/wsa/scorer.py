"""Score WSA candidates against a research task."""

from __future__ import annotations

from ..models.discovery import AgentFitScore, WSACandidate


def score_candidate(
    candidate: WSACandidate,
    target_domains: list[str],
    target_verticals: list[str],
    target_entity_types: list[str],
    required_output_fields: list[str],
    available_input_params: dict[str, str],
) -> AgentFitScore:
    """Score a single WSA candidate's fit against task requirements."""

    # Domain match
    domain_score = 0.0
    if candidate.domain and target_domains:
        cd = candidate.domain.lower().replace("www.", "")
        for td in target_domains:
            td = td.lower().replace("www.", "")
            if cd == td or cd in td or td in cd:
                domain_score = 1.0
                break

    # Entity type match
    entity_score = 0.0
    if candidate.entity_type and target_entity_types:
        ce = candidate.entity_type.lower()
        for te in target_entity_types:
            if te.lower() in ce or ce in te.lower():
                entity_score = 1.0
                break

    # Vertical match
    vertical_score = 0.0
    if candidate.vertical and target_verticals:
        cv = candidate.vertical.lower()
        for tv in target_verticals:
            if tv.lower() in cv or cv in tv.lower():
                vertical_score = 1.0
                break

    # Output field coverage
    output_score = 0.0
    if required_output_fields and candidate.output_schema:
        available_fields = set(
            k.lower() for k in candidate.output_schema.keys()
        )
        matched = sum(1 for f in required_output_fields if f.lower() in available_fields)
        output_score = matched / len(required_output_fields) if required_output_fields else 0.0

    # Input feasibility
    input_score = 0.0
    if candidate.input_properties:
        required_inputs = [
            k
            for k, v in candidate.input_properties.items()
            if isinstance(v, dict) and v.get("required")
        ]
        if not required_inputs:
            input_score = 1.0
        else:
            supplied = sum(1 for r in required_inputs if r in available_input_params)
            input_score = supplied / len(required_inputs)
    else:
        input_score = 0.5

    return AgentFitScore(
        agent_name=candidate.name,
        domain_match=domain_score,
        entity_type_match=entity_score,
        vertical_match=vertical_score,
        output_field_coverage=output_score,
        input_feasibility=input_score,
    )
